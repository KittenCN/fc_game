"""High-level emulator wrapper built on nes-py."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .compat import ensure_numpy_rom_compatibility
from .controller import ControllerState
from .rom import ROM

ensure_numpy_rom_compatibility()

from nes_py import NESEnv


class NESEmulator:
    """Wrap nes-py with convenience helpers for manual or programmatic play."""

    def __init__(self, rom_path: str | Path):
        self.rom = ROM(rom_path)
        self.env = NESEnv(str(Path(rom_path).resolve()))
        self._action_space = self.env.action_space

    def reset(self) -> np.ndarray:
        """Reset the emulator and return the initial frame."""
        outcome = self.env.reset()
        if isinstance(outcome, tuple) and len(outcome) == 2:
            observation, _info = outcome
        else:
            observation = outcome
        return observation

    def step(self, controller: ControllerState | Iterable[int]) -> tuple[np.ndarray, float, bool, dict]:
        """Advance one frame with the provided controller input."""
        if isinstance(controller, ControllerState):
            action = controller.to_action()
        else:
            action = list(controller)
        outcome = self.env.step(action)
        if len(outcome) == 5:
            obs, reward, terminated, truncated, info = outcome
            done = terminated or truncated
        else:
            obs, reward, done, info = outcome
        return obs, reward, done, info

    def press_buttons(self, **buttons: bool) -> tuple[np.ndarray, float, bool, dict]:
        """Helper to step using keyword button arguments."""
        state = ControllerState(**{key.upper(): val for key, val in buttons.items()})
        return self.step(state)

    def run_frame(self, controller: ControllerState | Iterable[int]) -> np.ndarray:
        obs, _, _, _ = self.step(controller)
        return obs

    def get_ram(self) -> np.ndarray:
        return self.env.get_ram()

    def get_screen(self) -> np.ndarray:
        return self.env.screen

    def close(self) -> None:
        self.env.close()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self.env.observation_space
