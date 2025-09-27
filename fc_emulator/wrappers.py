"""Environment wrappers and action presets for NES RL training."""
from __future__ import annotations

from collections import deque
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
DEFAULT_ACTION_SET: tuple[tuple[str, ...], ...] = (  # More SMB-focused default
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



class EpsilonRandomActionWrapper(gym.ActionWrapper):
    """Inject epsilon-greedy exploration while biasing useful macro actions."""

    def __init__(
        self,
        env: gym.Env,
        epsilon: float,
        *,
        skill_actions: tuple[int, ...] | None = None,
        skill_sequences: tuple[tuple[int, ...], ...] | None = None,
        forward_sequences: tuple[tuple[int, ...], ...] | None = None,
        backward_sequences: tuple[tuple[int, ...], ...] | None = None,
        neutral_sequences: tuple[tuple[int, ...], ...] | None = None,
        skill_bias: float = 0.7,
        stagnation_boost: float = 0.3,
        stagnation_threshold: int = 120,
        hotspot_threshold: int = 3,
        hotspot_window: int = 18,
        hotspot_bucket: int = 32,
        sequence_escalation_bias: float = 0.85,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("EpsilonRandomActionWrapper requires a discrete action space")
        self.epsilon = max(0.0, float(epsilon))
        self._skill_actions: tuple[int, ...] = tuple(skill_actions or ())
        self._skill_sequences: tuple[tuple[int, ...], ...] = tuple(
            seq for seq in (skill_sequences or ()) if seq
        )
        self._forward_sequences: tuple[tuple[int, ...], ...] = tuple(
            seq for seq in (forward_sequences or ()) if seq
        )
        self._backward_sequences: tuple[tuple[int, ...], ...] = tuple(
            seq for seq in (backward_sequences or ()) if seq
        )
        self._neutral_sequences: tuple[tuple[int, ...], ...] = tuple(
            seq for seq in (neutral_sequences or ()) if seq
        )
        self._skill_bias = float(min(max(skill_bias, 0.0), 1.0))
        self._stagnation_boost = max(0.0, float(stagnation_boost))
        self._stagnation_threshold = max(1, int(stagnation_threshold))
        self._macro_queue: deque[int] = deque()
        self._macro_cooldown = 0
        self._hotspot_threshold = max(2, int(hotspot_threshold))
        self._hotspot_window = max(self._hotspot_threshold, int(hotspot_window))
        self._hotspot_bucket = max(4, int(hotspot_bucket))
        self._sequence_escalation_bias = float(
            min(max(sequence_escalation_bias, 0.0), 1.0)
        )
        self._hotspot_history: deque[int] = deque()
        self._hotspot_counts: dict[int, int] = {}
        self._active_hotspot: int | None = None
        self._hotspot_direction: str | None = None
        self._last_mario_x: int | None = None
        self._recent_direction: str | None = None
        self._last_action: int | None = None

    def reset(self, **kwargs):  # type: ignore[override]
        self._macro_queue.clear()
        self._macro_cooldown = 0
        self._hotspot_history.clear()
        self._hotspot_counts.clear()
        self._active_hotspot = None
        self._hotspot_direction = None
        self._last_mario_x = None
        self._recent_direction = None
        self._last_action = None
        return super().reset(**kwargs)

    def _current_stagnation(self) -> int:
        unwrapped = getattr(self.env, "unwrapped", self.env)
        return int(getattr(unwrapped, "stagnation_counter", 0))

    def _queue_macro(self, sequence: tuple[int, ...]) -> int:
        if not sequence:
            choice = int(self.action_space.sample())
            self._last_action = choice
            return choice
        if len(sequence) > 1:
            self._macro_queue.extend(sequence[1:])
        self._macro_cooldown = max(self._macro_cooldown, len(sequence))
        choice = int(sequence[0])
        self._last_action = choice
        return choice

    def _select_sequence(
        self, sequences: tuple[tuple[int, ...], ...]
    ) -> tuple[int, ...] | None:
        if not sequences:
            return None
        idx = int(self.np_random.integers(len(sequences)))
        return sequences[idx]

    def _get_priority_sequences(self) -> tuple[tuple[int, ...], ...]:
        if self._active_hotspot is None:
            return ()
        if self._hotspot_direction == "backward":
            if self._backward_sequences:
                return self._backward_sequences + self._neutral_sequences
        elif self._hotspot_direction == "forward":
            if self._forward_sequences:
                return self._forward_sequences + self._neutral_sequences
        return self._skill_sequences

    def _register_hotspot(self, position: int | None) -> None:
        if position is None:
            return
        bucket = (position // self._hotspot_bucket) * self._hotspot_bucket
        if len(self._hotspot_history) >= self._hotspot_window:
            removed = self._hotspot_history.popleft()
            count = self._hotspot_counts.get(removed)
            if count is not None:
                if count <= 1:
                    self._hotspot_counts.pop(removed, None)
                else:
                    self._hotspot_counts[removed] = count - 1
        self._hotspot_history.append(bucket)
        self._hotspot_counts[bucket] = self._hotspot_counts.get(bucket, 0) + 1
        count = self._hotspot_counts[bucket]
        if count >= self._hotspot_threshold:
            self._active_hotspot = bucket
            self._hotspot_direction = self._recent_direction or "forward"

    def _record_progress(self, info: dict) -> None:
        metrics = info.get("metrics") or {}
        x_pos = metrics.get("mario_x")
        if x_pos is not None:
            if self._last_mario_x is not None:
                if x_pos > self._last_mario_x:
                    self._recent_direction = "forward"
                    if (
                        self._active_hotspot is not None
                        and x_pos >= self._active_hotspot + self._hotspot_bucket * 2
                    ):
                        self._active_hotspot = None
                        self._hotspot_direction = None
                elif x_pos < self._last_mario_x:
                    self._recent_direction = "backward"
            self._last_mario_x = x_pos
        if info.get("stagnation_truncated"):
            self._register_hotspot(x_pos)

    def step(self, action):  # type: ignore[override]
        observation, reward, terminated, truncated, info = super().step(action)
        self._record_progress(info)
        return observation, reward, terminated, truncated, info

    def action(self, action: int) -> int:  # type: ignore[override]
        if self._macro_cooldown > 0:
            self._macro_cooldown = max(0, self._macro_cooldown - 1)

        if self._macro_queue:
            choice = int(self._macro_queue.popleft())
            self._last_action = choice
            return choice

        stagnation = self._current_stagnation()
        prioritized = self._get_priority_sequences()
        if (
            prioritized
            and stagnation >= max(1, self._stagnation_threshold // 2)
            and self._macro_cooldown == 0
            and self.np_random.random() < self._sequence_escalation_bias
        ):
            sequence = self._select_sequence(prioritized)
            if sequence:
                return self._queue_macro(sequence)

        if (
            self._skill_sequences
            and stagnation >= self._stagnation_threshold
            and self._macro_cooldown == 0
        ):
            sequence_bias = 1.0 if stagnation >= self._stagnation_threshold * 2 else min(
                1.0, self._skill_bias + 0.15
            )
            if self.np_random.random() < sequence_bias:
                sequence = self._select_sequence(self._skill_sequences)
                if sequence:
                    return self._queue_macro(sequence)

        effective_epsilon = self.epsilon
        if (
            self.epsilon > 0.0
            and stagnation >= self._stagnation_threshold
            and self._stagnation_boost > 0.0
        ):
            ratio = stagnation / float(self._stagnation_threshold)
            effective_epsilon = min(1.0, self.epsilon + ratio * self._stagnation_boost)
            if stagnation >= self._stagnation_threshold * 3:
                effective_epsilon = 1.0

        if effective_epsilon <= 0.0 or self.np_random.random() >= effective_epsilon:
            self._last_action = int(action)
            return action

        if self._skill_sequences and self.np_random.random() < self._skill_bias:
            sequence = self._select_sequence(self._skill_sequences)
            if sequence:
                return self._queue_macro(sequence)

        if self._skill_actions and self.np_random.random() < self._skill_bias:
            choice = int(self.np_random.choice(self._skill_actions))
            self._last_action = choice
            return choice

        sampled = int(self.action_space.sample())
        self._last_action = sampled
        return sampled

    def set_exploration_epsilon(self, epsilon: float) -> None:
        self.epsilon = max(0.0, float(epsilon))

    def set_skill_actions(self, actions: tuple[int, ...], *, bias: float | None = None) -> None:
        self._skill_actions = tuple(actions)
        if bias is not None:
            self._skill_bias = float(min(max(bias, 0.0), 1.0))

    def set_skill_sequences(
        self,
        sequences: tuple[tuple[int, ...], ...],
        *,
        bias: float | None = None,
    ) -> None:
        self._skill_sequences = tuple(seq for seq in sequences if seq)
        if bias is not None:
            self._skill_bias = float(min(max(bias, 0.0), 1.0))

