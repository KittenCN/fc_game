"""Gymnasium-compatible environment wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np

from .auto_start import AutoStartConfig, AutoStartController
from .controller import BUTTON_ORDER, ControllerState
from .emulator import NESEmulator
from .renderer import ScreenRenderer
from .stagnation import StagnationConfig, StagnationMonitor

ObservationKind = Literal["rgb", "gray", "ram", "rgb_ram", "gray_ram"]


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
        self._last_auto_start_presses = 0

        auto_start_cfg = AutoStartConfig(
            max_frames=auto_start_max_frames,
            press_frames=auto_start_press_frames,
        )
        self._auto_start_controller = AutoStartController(auto_start_cfg, enabled=auto_start)

        base_frames = (
            None
            if stagnation_max_frames is None or stagnation_max_frames <= 0
            else int(stagnation_max_frames)
        )
        progress_threshold = max(0, int(stagnation_progress_threshold))
        self._stagnation_monitor = StagnationMonitor(
            StagnationConfig(base_frames=base_frames, progress_threshold=progress_threshold)
        )

        self.action_space = gym.spaces.MultiBinary(len(BUTTON_ORDER))
        self.observation_space = self._make_observation_space(observation_type)

    @property
    def stagnation_counter(self) -> int:
        """Expose current stagnation frame count for wrappers."""
        return int(self._stagnation_monitor.counter)


    # Gym API ---------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        if self.reward_config:
            self.reward_config.reset()
        self._episode_steps = 0
        frame = self.emulator.reset()
        frame, auto_start_presses = self._auto_start_controller.warmup(
            self.emulator, initial_frame=frame
        )
        self._last_auto_start_presses = auto_start_presses
        ram_snapshot = self.emulator.get_ram()
        metrics = self._extract_metrics_from_ram(ram_snapshot)
        self._stagnation_monitor.reset(metrics, ram_snapshot)
        obs = self._process_observation(frame, ram_snapshot)
        info: dict[str, Any] = {"metrics": metrics}
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
        ram_snapshot = self.emulator.get_ram().copy()
        processed = self._process_observation(last_frame, ram_snapshot)
        self._episode_steps += 1

        metrics = self._extract_metrics_from_ram(ram_snapshot)
        merged_info = dict(info)
        merged_info.setdefault("metrics", {}).update(metrics)
        base_reward_value = total_reward

        status = self._stagnation_monitor.update(metrics, ram_snapshot, frame_skip=self.frame_skip)
        diagnostics = merged_info.setdefault("diagnostics", {})
        metrics_entry = merged_info.setdefault("metrics", {})
        diagnostics["stagnation_counter"] = int(self._stagnation_monitor.counter)
        if status.limit is not None:
            diagnostics["stagnation_limit"] = int(status.limit)
        if status.reason:
            diagnostics["stagnation_reason"] = status.reason
            metrics_entry["stagnation_reason"] = status.reason
        if status.event:
            diagnostics["stagnation_event"] = status.event
            metrics_entry["stagnation_event"] = status.event
        if status.position is not None:
            diagnostics["stagnation_position"] = int(status.position)
            metrics_entry["stagnation_position"] = int(status.position)
        if status.bucket is not None:
            diagnostics["stagnation_bucket"] = int(status.bucket)
            metrics_entry["stagnation_bucket"] = int(status.bucket)
        if status.idle_counter is not None:
            diagnostics["stagnation_idle_frames"] = int(status.idle_counter)
            metrics_entry["stagnation_idle_frames"] = int(status.idle_counter)
        if status.frames is not None:
            merged_info["stagnation_frames"] = status.frames
            diagnostics["stagnation_frames"] = status.frames
        if not terminated and status.triggered:
            truncated = True
            merged_info["stagnation_truncated"] = True


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
            diagnostics["base_reward"] = base_reward_value
            diagnostics["shaped_reward"] = total_reward
        elif status.frames is not None:
            diagnostics["stagnation_frames"] = status.frames

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

    def _process_observation(self, frame: np.ndarray, ram: np.ndarray) -> np.ndarray | dict[str, np.ndarray]:
        if self.observation_type == "rgb":
            return frame
        if self.observation_type == "gray":
            return self._rgb_to_gray(frame)
        if self.observation_type == "ram":
            return ram
        if self.observation_type == "rgb_ram":
            return {"pixels": frame, "ram": ram.copy()}
        if self.observation_type == "gray_ram":
            return {"pixels": self._rgb_to_gray(frame), "ram": ram.copy()}
        raise ValueError(f"Unsupported observation type {self.observation_type}")

    def _make_observation_space(self, kind: ObservationKind):
        pixel_space = gym.spaces.Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)
        gray_space = gym.spaces.Box(low=0, high=255, shape=(240, 256, 1), dtype=np.uint8)
        ram_space = gym.spaces.Box(low=0, high=255, shape=(2048,), dtype=np.uint8)
        if kind == "rgb":
            return pixel_space
        if kind == "gray":
            return gray_space
        if kind == "ram":
            return ram_space
        if kind == "rgb_ram":
            return gym.spaces.Dict({"pixels": pixel_space, "ram": ram_space})
        if kind == "gray_ram":
            return gym.spaces.Dict({"pixels": gray_space, "ram": ram_space})
        raise ValueError(kind)    
    def _rgb_to_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return gray[..., None]

    def _ensure_renderer(self) -> None:
        if not self._renderer:
            self._renderer = ScreenRenderer()
