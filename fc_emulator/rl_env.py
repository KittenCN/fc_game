"""Gymnasium-compatible environment wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np

from .controller import BUTTON_ORDER, ControllerState
from .emulator import NESEmulator
from .renderer import ScreenRenderer

ObservationKind = Literal["rgb", "gray", "ram"]


def _to_controller(action: np.ndarray | list[int]) -> ControllerState:
    values = [bool(v) for v in action]
    return ControllerState(**{name: values[idx] for idx, name in enumerate(BUTTON_ORDER)})


@dataclass(slots=True)
class RewardContext:
    """Runtime data provided to reward shaping callbacks."""

    frame: np.ndarray
    base_reward: float
    info: dict[str, Any]
    ram: np.ndarray
    done: bool
    step: int


@dataclass(slots=True)
class RewardConfig:
    """Allows plugging a custom reward shaping callback."""

    func: Callable[[RewardContext], float]
    on_reset: Callable[[], None] | None = None

    def compute(self, context: RewardContext) -> float:
        return self.func(context)

    def reset(self) -> None:
        if self.on_reset:
            self.on_reset()


class NESGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str,
        *,
        render_mode: ObservationKind | None = None,
        frame_skip: int = 1,
        observation_type: ObservationKind = "rgb",
        reward_config: RewardConfig | None = None,
        auto_start: bool = True,
        auto_start_max_frames: int = 120,
        auto_start_press_frames: int = 6,
        stagnation_max_frames: int | None = 900,
        stagnation_progress_threshold: int = 1,
    ) -> None:
        super().__init__()
        self.emulator = NESEmulator(rom_path)
        self.frame_skip = max(1, frame_skip)
        self.observation_type: ObservationKind = observation_type
        self.reward_config = reward_config
        self._renderer: ScreenRenderer | None = None
        self.render_mode = render_mode
        self._episode_steps = 0
        self.auto_start = auto_start
        self._auto_start_max_frames = max(1, auto_start_max_frames)
        self._auto_start_press_frames = max(1, auto_start_press_frames)
        self._auto_start_timer_threshold = 401
        self._start_press_state = ControllerState(START=True)
        self._noop_state = ControllerState()
        self._last_auto_start_presses = 0
        self._stagnation_max_frames = (
            None
            if stagnation_max_frames is None or stagnation_max_frames <= 0
            else int(stagnation_max_frames)
        )
        self._stagnation_progress_threshold = max(0, int(stagnation_progress_threshold))
        self._stagnation_counter = 0
        self._last_progress_x: int | None = None

        self.action_space = gym.spaces.MultiBinary(len(BUTTON_ORDER))
        self.observation_space = self._make_observation_space(observation_type)

    # Gym API ---------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        if self.reward_config:
            self.reward_config.reset()
        self._episode_steps = 0
        self._stagnation_counter = 0
        self._last_progress_x = None
        frame = self.emulator.reset()
        frame, auto_start_presses = self._auto_start_if_needed(frame)
        obs = self._process_observation(frame)
        ram_snapshot = self.emulator.get_ram()
        metrics = self._extract_metrics_from_ram(ram_snapshot)
        self._initialize_stagnation(metrics, ram_snapshot)
        info: dict[str, Any] = {
            "metrics": metrics
        }
        if auto_start_presses:
            diagnostics = info.setdefault("diagnostics", {})
            diagnostics["auto_start_presses"] = auto_start_presses
        if self.render_mode == "human":
            self._ensure_renderer()
            assert self._renderer is not None
            self._renderer.draw_frame(frame)
        return obs, info

    def step(self, action):
        controller = _to_controller(action)
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        last_frame = None

        for _ in range(self.frame_skip):
            frame, reward, done, step_info = self._step_once(controller)
            total_reward += reward
            last_frame = frame
            if step_info:
                info = step_info
            if done:
                terminated = True
                break

        assert last_frame is not None
        processed = self._process_observation(last_frame)
        self._episode_steps += 1

        ram_snapshot = self.emulator.get_ram().copy()
        metrics = self._extract_metrics_from_ram(ram_snapshot)
        merged_info = dict(info)
        merged_info.setdefault("metrics", {}).update(metrics)
        base_reward_value = total_reward

        stagnation_frames: int | None = None
        if self._stagnation_max_frames is not None and not terminated:
            triggered, frames = self._update_stagnation(metrics, ram_snapshot)
            if triggered:
                truncated = True
                stagnation_frames = frames
                merged_info["stagnation_truncated"] = True
        if stagnation_frames is not None:
            merged_info["stagnation_frames"] = stagnation_frames

        if self.reward_config:
            context = RewardContext(
                frame=last_frame,
                base_reward=base_reward_value,
                info=merged_info,
                ram=ram_snapshot,
                done=terminated or truncated,
                step=self._episode_steps,
            )
            total_reward = self.reward_config.compute(context)
            diagnostics = merged_info.setdefault("diagnostics", {})
            diagnostics["base_reward"] = base_reward_value
            diagnostics["shaped_reward"] = total_reward
            if stagnation_frames is not None:
                diagnostics.setdefault("stagnation_frames", stagnation_frames)
        elif stagnation_frames is not None:
            merged_info.setdefault("diagnostics", {})["stagnation_frames"] = stagnation_frames

        if self.render_mode == "human":
            self._ensure_renderer()
            assert self._renderer is not None
            self._renderer.draw_frame(last_frame)

        return processed, total_reward, terminated, truncated, merged_info
    def render(self):
        if self.render_mode != "human":
            frame = self.emulator.get_screen()
            if self.render_mode == "rgb":
                return frame
            if self.render_mode == "gray":
                return self._rgb_to_gray(frame)
        return None

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None
        self.emulator.close()
        super().close()

    # Internal helpers -----------------------------------------------------
    def _auto_start_if_needed(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        if not self.auto_start:
            self._last_auto_start_presses = 0
            return frame, 0
        presses = 0
        frames_spent = 0
        current_frame = frame
        while frames_spent < self._auto_start_max_frames and self._needs_auto_start_press():
            current_frame, _reward, done, _info = self.emulator.step(self._start_press_state)
            presses += 1
            frames_spent += 1
            if done:
                current_frame = self.emulator.reset()
                frames_spent = 0
                continue
            release_frames = 0
            while (
                release_frames < self._auto_start_press_frames
                and frames_spent < self._auto_start_max_frames
            ):
                current_frame, _reward, done, _info = self.emulator.step(self._noop_state)
                release_frames += 1
                frames_spent += 1
                if done:
                    current_frame = self.emulator.reset()
                    frames_spent = 0
                    break
        self._last_auto_start_presses = presses
        return current_frame, presses

    def _initialize_stagnation(self, metrics: dict[str, int], ram: np.ndarray) -> None:
        if self._stagnation_max_frames is None:
            self._stagnation_counter = 0
            self._last_progress_x = None
            return
        self._stagnation_counter = 0
        self._last_progress_x = self._get_mario_x(metrics, ram)

    def _update_stagnation(
        self, metrics: dict[str, int], ram: np.ndarray
    ) -> tuple[bool, int | None]:
        if self._stagnation_max_frames is None:
            return False, None
        current_x = self._get_mario_x(metrics, ram)
        if self._last_progress_x is None:
            self._last_progress_x = current_x
            self._stagnation_counter = 0
            return False, None
        if current_x > self._last_progress_x:
            delta = current_x - self._last_progress_x
            if delta >= self._stagnation_progress_threshold:
                self._stagnation_counter = 0
            else:
                self._stagnation_counter = max(0, self._stagnation_counter - self.frame_skip)
            self._last_progress_x = max(self._last_progress_x, current_x)
            return False, None
        self._stagnation_counter += self.frame_skip
        if self._stagnation_counter >= self._stagnation_max_frames:
            triggered_frames = self._stagnation_counter
            self._stagnation_counter = 0
            self._last_progress_x = current_x
            return True, triggered_frames
        return False, None

    @staticmethod
    def _get_mario_x(metrics: dict[str, int], ram: np.ndarray) -> int:
        x_pos = metrics.get("mario_x")
        if x_pos is None:
            x_pos = int(ram[0x6D]) * 256 + int(ram[0x86])
        return int(x_pos)

    def _needs_auto_start_press(self) -> bool:
        ram_snapshot = self.emulator.get_ram()
        timer = self._decode_timer_from_ram(ram_snapshot)
        # Super Mario Bros keeps the timer at 401 on the title screen; it drops once gameplay begins.
        if timer is None:
            return False
        if timer <= self._auto_start_timer_threshold - 1:
            return False
        if len(ram_snapshot) > 0x0770:
            # Known NES gameplay state codes when the level is active (SMB heuristic).
            game_mode = int(ram_snapshot[0x0770])
            if game_mode in (0x06, 0x07):
                return False
        return True

    @staticmethod
    def _decode_timer_from_ram(ram: np.ndarray) -> int | None:
        try:
            hundreds = int(ram[0x07F8]) & 0x0F
            tens = int(ram[0x07F9]) & 0x0F
            ones = int(ram[0x07FA]) & 0x0F
        except IndexError:
            return None
        return hundreds * 100 + tens * 10 + ones

    def _step_once(self, controller: ControllerState):
        frame, reward, done, info = self.emulator.step(controller)
        return frame, reward, done, info

    @staticmethod
    def _extract_metrics_from_ram(ram: np.ndarray) -> dict[str, int]:
        metrics: dict[str, int] = {}
        try:
            metrics["mario_x"] = int(ram[0x6D]) * 256 + int(ram[0x86])
            metrics["world"] = int(ram[0x075F]) + 1
            metrics["stage"] = int(ram[0x0760]) + 1
            metrics["area"] = int(ram[0x0761])
            metrics["timer"] = (
                (int(ram[0x07F8]) & 0x0F) * 100
                + (int(ram[0x07F9]) & 0x0F) * 10
                + (int(ram[0x07FA]) & 0x0F)
            )
            score_digits = [int(ram[addr]) & 0x0F for addr in range(0x07DE, 0x07E4)]
            score = 0
            for digit in score_digits:
                score = score * 10 + digit
            metrics["score"] = score
            metrics["player_state"] = int(ram[0x0756])
            metrics["lives"] = int(ram[0x075A]) & 0x0F
        except IndexError:
            pass
        return metrics

    def _process_observation(self, frame: np.ndarray) -> np.ndarray:
        if self.observation_type == "rgb":
            return frame
        if self.observation_type == "gray":
            return self._rgb_to_gray(frame)
        if self.observation_type == "ram":
            return self.emulator.get_ram()
        raise ValueError(f"Unsupported observation type {self.observation_type}")

    def _make_observation_space(self, kind: ObservationKind):
        if kind == "rgb":
            return gym.spaces.Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)
        if kind == "gray":
            return gym.spaces.Box(low=0, high=255, shape=(240, 256, 1), dtype=np.uint8)
        if kind == "ram":
            return gym.spaces.Box(low=0, high=255, shape=(2048,), dtype=np.uint8)
        raise ValueError(kind)

    def _rgb_to_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return gray[..., None]

    def _ensure_renderer(self) -> None:
        if not self._renderer:
            self._renderer = ScreenRenderer()
