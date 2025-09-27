"""Utilities for automatically advancing NES title screens."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import ControllerState


@dataclass(frozen=True)
class AutoStartConfig:
    max_frames: int = 120
    press_frames: int = 6
    timer_threshold: int = 401


class AutoStartController:
    """Drive automatic START button presses during environment reset."""

    def __init__(self, config: AutoStartConfig, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.config = AutoStartConfig(
            max_frames=max(1, config.max_frames),
            press_frames=max(1, config.press_frames),
            timer_threshold=max(0, config.timer_threshold),
        )
        self._start_state = ControllerState(START=True)
        self._noop_state = ControllerState()

    @staticmethod
    def _decode_timer(ram: np.ndarray) -> int | None:
        try:
            hundreds = int(ram[0x07F8]) & 0x0F
            tens = int(ram[0x07F9]) & 0x0F
            ones = int(ram[0x07FA]) & 0x0F
        except (IndexError, TypeError):
            return None
        return hundreds * 100 + tens * 10 + ones

    def _needs_press(self, ram: np.ndarray) -> bool:
        timer = self._decode_timer(ram)
        if timer is None:
            return False
        if timer < self.config.timer_threshold:
            return False
        if len(ram) > 0x0770:
            game_mode = int(ram[0x0770])
            if game_mode in (0x06, 0x07):
                return False
        return True

    def warmup(self, emulator, *, initial_frame) -> tuple[np.ndarray, int]:
        if not self.enabled:
            return initial_frame, 0

        current_frame = initial_frame
        presses = 0
        frames_spent = 0

        while frames_spent < self.config.max_frames and self._needs_press(emulator.get_ram()):
            current_frame, _reward, done, _info = emulator.step(self._start_state)
            presses += 1
            frames_spent += 1
            if done:
                current_frame = emulator.reset()
                frames_spent = 0
                continue

            release_frames = 0
            while (
                release_frames < self.config.press_frames
                and frames_spent < self.config.max_frames
            ):
                current_frame, _reward, done, _info = emulator.step(self._noop_state)
                release_frames += 1
                frames_spent += 1
                if done:
                    current_frame = emulator.reset()
                    frames_spent = 0
                    break

        return current_frame, presses


__all__ = ["AutoStartConfig", "AutoStartController"]
