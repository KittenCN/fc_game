"""Discrete action presets and helpers for NES controllers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import gymnasium as gym
import numpy as np

from .controller import BUTTON_ORDER


@dataclass(frozen=True)
class ActionPreset:
    """Declarative description of a discrete action library."""

    name: str
    combos: tuple[tuple[str, ...], ...]


def _normalize_combo(combo: Sequence[str]) -> tuple[str, ...]:
    return tuple(button.upper() for button in combo)


def _build_index() -> dict[str, int]:
    return {button: idx for idx, button in enumerate(BUTTON_ORDER)}


_BUTTON_INDEX = _build_index()


DEFAULT_ACTION_SET: tuple[tuple[str, ...], ...] = (
    (),
    ("LEFT",),
    ("RIGHT",),
    ("A", "LEFT"),
    ("A", "RIGHT"),
    ("B", "LEFT"),
    ("B", "RIGHT"),
    ("A", "B", "LEFT"),
    ("A", "B", "RIGHT"),
    ("DOWN",),
    ("UP",),
    ("START",),
)


_ACTION_LIBRARY: Dict[str, ActionPreset] = {
    "default": ActionPreset("default", tuple(map(_normalize_combo, DEFAULT_ACTION_SET))),
    "simple": ActionPreset(
        "simple",
        tuple(
            map(
                _normalize_combo,
                (
                    (),
                    ("A",),
                    ("B",),
                    ("LEFT",),
                    ("RIGHT",),
                    ("A", "RIGHT"),
                    ("A", "LEFT"),
                    ("START",),
                ),
            )
        ),
    ),
    "smb_forward": ActionPreset(
        "smb_forward",
        tuple(
            map(
                _normalize_combo,
                (
                    ("START",),
                    ("RIGHT",),
                    ("A", "RIGHT"),
                    ("B", "RIGHT"),
                    ("A", "B", "RIGHT"),
                    ("RIGHT", "DOWN"),
                    ("RIGHT", "UP"),
                    ("DOWN",),
                    ("A",),
                    ("B",),
                    ("A", "B"),
                    ("LEFT",),
                    ("A", "LEFT"),
                    ("B", "LEFT"),
                    ("A", "B", "LEFT"),
                    (),
                ),
            )
        ),
    ),
}


def register_action_preset(preset: ActionPreset, *, overwrite: bool = False) -> None:
    """Register a new action preset at runtime."""

    key = preset.name.lower()
    if not overwrite and key in _ACTION_LIBRARY:
        raise ValueError(f"Preset '{preset.name}' already exists")
    _ACTION_LIBRARY[key] = preset


def resolve_action_set(name_or_combos: str | Sequence[Sequence[str]] | None) -> tuple[tuple[str, ...], ...]:
    """Resolve user input into a tuple of button combinations."""

    if name_or_combos is None:
        return _ACTION_LIBRARY["default"].combos
    if isinstance(name_or_combos, str):
        preset = _ACTION_LIBRARY.get(name_or_combos.lower())
        if preset is None:
            raise KeyError(f"Unknown action preset '{name_or_combos}'")
        return preset.combos
    combos = tuple(_normalize_combo(combo) for combo in name_or_combos)
    if not combos:
        raise ValueError("Action set must contain at least one combo")
    return combos


def combo_to_multibinary(buttons: Iterable[str]) -> np.ndarray:
    """Convert a combo of button names into a MultiBinary action array."""

    action = np.zeros(len(BUTTON_ORDER), dtype=np.uint8)
    for button in buttons:
        try:
            action[_BUTTON_INDEX[button.upper()]] = 1
        except KeyError as exc:  # pragma: no cover - defensive fallback
            raise ValueError(f"Unknown NES button: {button}") from exc
    return action


class DiscreteActionWrapper(gym.ActionWrapper):
    """Map a discrete index to a predetermined button combination."""

    def __init__(self, env: gym.Env, action_set: Sequence[Sequence[str]] | None = None) -> None:
        super().__init__(env)
        combos = resolve_action_set(action_set)
        self._action_set: tuple[tuple[str, ...], ...] = combos
        self.action_space = gym.spaces.Discrete(len(self._action_set))

    def action(self, action: int) -> np.ndarray:  # type: ignore[override]
        try:
            combo = self._action_set[action]
        except IndexError as exc:
            raise ValueError(f"Action index {action} outside of action set") from exc
        return combo_to_multibinary(combo)

    def reverse_action(self, action: np.ndarray) -> int:
        combo = tuple(btn for idx, btn in enumerate(BUTTON_ORDER) if action[idx])
        return self._action_set.index(combo)


__all__ = [
    "ActionPreset",
    "DEFAULT_ACTION_SET",
    "DiscreteActionWrapper",
    "combo_to_multibinary",
    "register_action_preset",
    "resolve_action_set",
]
