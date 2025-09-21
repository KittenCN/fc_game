"""Environment wrappers and action presets for NES RL training."""
from __future__ import annotations

from typing import Iterable, Sequence

import gymnasium as gym
import numpy as np

from .controller import BUTTON_ORDER

# Minimal yet expressive set of button combinations for action discretization.
DEFAULT_ACTION_SET: tuple[tuple[str, ...], ...] = (
    (),
    ("A",),
    ("B",),
    ("UP",),
    ("DOWN",),
    ("LEFT",),
    ("RIGHT",),
    ("A", "RIGHT"),
    ("A", "LEFT"),
    ("B", "RIGHT"),
    ("B", "LEFT"),
    ("UP", "A"),
    ("DOWN", "A"),
    ("START",),
)

ACTION_PRESETS: dict[str, tuple[tuple[str, ...], ...]] = {
    "default": DEFAULT_ACTION_SET,
    "simple": (
        (),
        ("A",),
        ("B",),
        ("LEFT",),
        ("RIGHT",),
        ("A", "RIGHT"),
        ("A", "LEFT"),
        ("START",),
    ),
}

_INDEX_FOR_BUTTON = {button: idx for idx, button in enumerate(BUTTON_ORDER)}


def combo_to_multibinary(buttons: Iterable[str]) -> np.ndarray:
    """Convert a combo of button names into a MultiBinary action array."""
    action = np.zeros(len(BUTTON_ORDER), dtype=np.uint8)
    for button in buttons:
        try:
            action[_INDEX_FOR_BUTTON[button.upper()]] = 1
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown NES button: {button}") from exc
    return action


class DiscreteActionWrapper(gym.ActionWrapper):
    """Map a discrete action index to a predetermined button combination."""

    def __init__(
        self,
        env: gym.Env,
        action_set: Sequence[Sequence[str]] | None = None,
    ) -> None:
        super().__init__(env)
        combos = action_set or DEFAULT_ACTION_SET
        self._action_set: tuple[tuple[str, ...], ...] = tuple(
            tuple(btn.upper() for btn in combo) for combo in combos
        )
        if not self._action_set:
            raise ValueError("Action set must contain at least one combo")
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
