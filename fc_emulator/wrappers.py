"""Environment wrappers and action presets for NES RL training."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import gymnasium as gym
import numpy as np

from .controller import BUTTON_ORDER

# Optional backends for resizing observations.
try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:  # pragma: no cover - optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

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
    "smb_forward": (
        ("RIGHT",),
        ("A", "RIGHT"),
        ("B", "RIGHT"),
        ("A", "B", "RIGHT"),
        ("RIGHT", "DOWN"),
        ("RIGHT", "UP"),
        ("A",),
        ("B",),
        ("A", "B"),
        (),
    ),
}

_INDEX_FOR_BUTTON = {button: idx for idx, button in enumerate(BUTTON_ORDER)}


def _resize_frame(frame: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    height, width = shape
    if cv2 is not None:  # pragma: no cover - requires optional dep
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    elif Image is not None:  # pragma: no cover - requires optional dep
        if frame.ndim == 2 or frame.shape[-1] == 1:
            img = Image.fromarray(frame.squeeze(), mode="L")
            resized = np.asarray(img.resize((width, height), Image.BILINEAR), dtype=frame.dtype)
            resized = resized[..., None]
        else:
            img = Image.fromarray(frame)
            resized = np.asarray(img.resize((width, height), Image.BILINEAR), dtype=frame.dtype)
    else:  # pragma: no cover - defensive
        raise RuntimeError(
            "Resizing observations requires installing either opencv-python or pillow."
        )

    if resized.dtype != frame.dtype:
        resized = resized.astype(frame.dtype)
    if frame.ndim == 3 and resized.ndim == 2:
        resized = resized[..., None]
    return resized


def combo_to_multibinary(buttons: Iterable[str]) -> np.ndarray:
    """Convert a combo of button names into a MultiBinary action array."""
    action = np.zeros(len(BUTTON_ORDER), dtype=np.uint8)
    for button in buttons:
        try:
            action[_INDEX_FOR_BUTTON[button.upper()]] = 1
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown NES button: {button}") from exc
    return action


class ResizeObservationWrapper(gym.ObservationWrapper):
    """Resize image observations without requiring global gym wrappers."""

    def __init__(self, env: gym.Env, shape: Tuple[int, int]):
        super().__init__(env)
        if len(shape) != 2:
            raise ValueError("shape must be (height, width)")
        self.shape = shape
        orig_space = env.observation_space
        if len(orig_space.shape) == 3:
            channels = orig_space.shape[2]
        else:
            channels = 1
        self.channels = channels
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[0], shape[1], channels),
            dtype=orig_space.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        resized = _resize_frame(observation, self.shape)
        if self.channels == 1 and resized.ndim == 2:
            resized = resized[..., None]
        return resized


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
