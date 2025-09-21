"""Pygame-based renderer that also captures keyboard input."""
from __future__ import annotations

import pygame
import numpy as np

from .controller import ControllerState, DEFAULT_KEYMAP

WINDOW_SCALE = 3
FPS = 60


class ScreenRenderer:
    """Manage window display and keyboard polling."""

    def __init__(self, title: str = "FC Emulator", scale: int = WINDOW_SCALE):
        pygame.init()
        self.scale = scale
        self.size = (256 * scale, 240 * scale)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self._surface = pygame.Surface((256, 240))

    def poll_inputs(self) -> ControllerState:
        pressed: set[str] = set()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
        keys = pygame.key.get_pressed()
        for name in DEFAULT_KEYMAP:
            try:
                key_code = pygame.key.key_code(name)
            except ValueError:
                continue
            if keys[key_code]:
                pressed.add(name)
        return ControllerState.from_pressed(pressed)

    def draw_frame(self, frame: np.ndarray) -> None:
        if frame.shape != (240, 256, 3):
            raise ValueError(f"Unexpected frame shape {frame.shape}")
        pygame.surfarray.blit_array(self._surface, frame.swapaxes(0, 1))
        scaled_surface = pygame.transform.scale(self._surface, self.size)
        self.screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self) -> None:
        pygame.quit()
