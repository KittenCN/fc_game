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
class RewardConfig:
    """Allows plugging a custom reward shaping callback."""

    func: Callable[[np.ndarray, float, dict[str, Any]], float]

    def compute(self, frame: np.ndarray, base_reward: float, info: dict[str, Any]) -> float:
        return self.func(frame, base_reward, info)


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
    ) -> None:
        super().__init__()
        self.emulator = NESEmulator(rom_path)
        self.frame_skip = max(1, frame_skip)
        self.observation_type: ObservationKind = observation_type
        self.reward_config = reward_config
        self._renderer: ScreenRenderer | None = None
        self.render_mode = render_mode

        self.action_space = gym.spaces.MultiBinary(len(BUTTON_ORDER))
        self.observation_space = self._make_observation_space(observation_type)

    # Gym API ---------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        frame = self.emulator.reset()
        obs = self._process_observation(frame)
        info: dict[str, Any] = {}
        if self.render_mode == "human":
            self._ensure_renderer()
            assert self._renderer is not None
            self._renderer.draw_frame(frame)
        return obs, info

    def step(self, action):
        controller = _to_controller(action)
        total_reward = 0.0
        terminated = False
        info: dict[str, Any] = {}
        last_frame = None

        for _ in range(self.frame_skip):
            frame, reward, done, info = self._step_once(controller)
            total_reward += reward
            last_frame = frame
            terminated = done
            if done:
                break

        assert last_frame is not None
        processed = self._process_observation(last_frame)
        if self.reward_config:
            total_reward = self.reward_config.compute(last_frame, total_reward, info)

        if self.render_mode == "human":
            self._ensure_renderer()
            assert self._renderer is not None
            self._renderer.draw_frame(last_frame)

        return processed, total_reward, terminated, False, info

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
