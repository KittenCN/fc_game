"""Controller abstractions mapping inputs to NES actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

DEFAULT_KEYMAP: Mapping[str, str] = {
    "w": "UP",
    "s": "DOWN",
    "a": "LEFT",
    "d": "RIGHT",
    "j": "A",
    "k": "B",
    "enter": "START",
    "right shift": "SELECT",
}


BUTTON_ORDER = ("A", "B", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT")


@dataclass(slots=True)
class ControllerState:
    """Represents the eight NES buttons as booleans."""

    A: bool = False
    B: bool = False
    SELECT: bool = False
    START: bool = False
    UP: bool = False
    DOWN: bool = False
    LEFT: bool = False
    RIGHT: bool = False

    def to_bitfield(self) -> int:
        value = 0
        for idx, button in enumerate(BUTTON_ORDER):
            if getattr(self, button):
                value |= 1 << idx
        return value

    def to_action(self) -> int:
        """Return the nes-py button bitmask."""
        return self.to_bitfield()

    @classmethod
    def from_pressed(cls, pressed: Iterable[str], keymap: Mapping[str, str] | None = None) -> "ControllerState":
        mapping = keymap or DEFAULT_KEYMAP
        normalized = {p.lower() for p in pressed}
        state_kwargs = {button: False for button in BUTTON_ORDER}
        for key, button in mapping.items():
            if key.lower() in normalized:
                state_kwargs[button] = True
        return cls(**state_kwargs)
