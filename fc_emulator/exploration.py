"""Exploration helpers and wrappers for macro-action scheduling."""
from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence, Tuple

import gymnasium as gym
import numpy as np


class MacroSequenceLibrary:
    """Container describing skill actions and macro sequences."""

    def __init__(
        self,
        *,
        skill_actions: Iterable[int] = (),
        skill_sequences: Iterable[Sequence[int]] = (),
        forward_sequences: Iterable[Sequence[int]] = (),
        backward_sequences: Iterable[Sequence[int]] = (),
        neutral_sequences: Iterable[Sequence[int]] = (),
    ) -> None:
        self.skill_actions = tuple(int(a) for a in skill_actions)
        self.skill_sequences = tuple(tuple(int(i) for i in seq) for seq in skill_sequences if seq)
        self.forward_sequences = tuple(tuple(int(i) for i in seq) for seq in forward_sequences if seq)
        self.backward_sequences = tuple(tuple(int(i) for i in seq) for seq in backward_sequences if seq)
        self.neutral_sequences = tuple(tuple(int(i) for i in seq) for seq in neutral_sequences if seq)


class EpsilonRandomActionWrapper(gym.ActionWrapper):
    """Inject epsilon-greedy exploration while biasing useful macro-actions."""

    def __init__(
        self,
        env: gym.Env,
        epsilon: float,
        *,
        sequences: MacroSequenceLibrary | None = None,
        skill_bias: float = 0.7,
        stagnation_boost: float = 0.3,
        stagnation_threshold: int = 120,
        hotspot_threshold: int = 3,
        hotspot_window: int = 18,
        hotspot_bucket: int = 32,
        sequence_escalation_bias: float = 0.85,
        hotspot_sequence_bias: float = 0.9,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("EpsilonRandomActionWrapper requires a discrete action space")
        self.epsilon = max(0.0, float(epsilon))
        library = sequences or MacroSequenceLibrary()
        self._skill_actions: tuple[int, ...] = library.skill_actions
        self._skill_sequences: tuple[tuple[int, ...], ...] = library.skill_sequences
        self._forward_sequences: tuple[tuple[int, ...], ...] = library.forward_sequences
        self._backward_sequences: tuple[tuple[int, ...], ...] = library.backward_sequences
        self._neutral_sequences: tuple[tuple[int, ...], ...] = library.neutral_sequences
        self._skill_bias = float(min(max(skill_bias, 0.0), 1.0))
        self._stagnation_boost = max(0.0, float(stagnation_boost))
        self._stagnation_threshold = max(1, int(stagnation_threshold))
        self._macro_queue: deque[int] = deque()
        self._macro_cooldown = 0
        self._hotspot_threshold = max(2, int(hotspot_threshold))
        self._hotspot_window = max(self._hotspot_threshold, int(hotspot_window))
        self._hotspot_bucket = max(4, int(hotspot_bucket))
        self._sequence_escalation_bias = float(min(max(sequence_escalation_bias, 0.0), 1.0))
        self._hotspot_sequence_bias = float(min(max(hotspot_sequence_bias, 0.0), 1.0))
        self._hotspot_history: deque[int] = deque()
        self._hotspot_counts: dict[int, int] = {}
        self._hotspot_direction_map: dict[int, str] = {}
        self._active_hotspot: int | None = None
        self._hotspot_direction: str | None = None
        self._last_mario_x: int | None = None
        self._recent_direction: str | None = None
        self._last_action: int | None = None

    def reset(self, **kwargs):  # type: ignore[override]
        self._macro_queue.clear()
        self._macro_cooldown = 0
        self._last_mario_x = None
        self._recent_direction = None
        self._last_action = None
        self._refresh_hotspot_target()
        return super().reset(**kwargs)

    # Public API ---------------------------------------------------------
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

    # Internal helpers ---------------------------------------------------
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
        if self._hotspot_direction == "backward" and self._backward_sequences:
            return self._backward_sequences + self._neutral_sequences
        if self._hotspot_direction == "forward" and self._forward_sequences:
            return self._forward_sequences + self._neutral_sequences
        return self._skill_sequences

    def _active_hotspot_intensity(self) -> float:
        if self._active_hotspot is None or not self._hotspot_history:
            return 0.0
        count = self._hotspot_counts.get(self._active_hotspot, 0)
        return count / float(len(self._hotspot_history))

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
                    self._hotspot_direction_map.pop(removed, None)
                else:
                    self._hotspot_counts[removed] = count - 1
        self._hotspot_history.append(bucket)
        self._hotspot_counts[bucket] = self._hotspot_counts.get(bucket, 0) + 1
        count = self._hotspot_counts[bucket]
        if count >= self._hotspot_threshold:
            direction = self._recent_direction or "forward"
            self._hotspot_direction_map[bucket] = direction
            self._active_hotspot = bucket
            self._hotspot_direction = direction
        self._refresh_hotspot_target(prefer_existing=True)

    def _refresh_hotspot_target(self, *, prefer_existing: bool = False) -> None:
        if self._hotspot_counts:
            if prefer_existing and self._active_hotspot is not None:
                current_count = self._hotspot_counts.get(self._active_hotspot, 0)
                if current_count >= self._hotspot_threshold:
                    self._hotspot_direction = self._hotspot_direction_map.get(
                        self._active_hotspot, self._hotspot_direction or self._recent_direction
                    )
                    return

            best_bucket = None
            best_count = 0
            for bucket, count in self._hotspot_counts.items():
                if count < self._hotspot_threshold:
                    continue
                if count > best_count or (count == best_count and bucket == self._active_hotspot):
                    best_bucket = bucket
                    best_count = count

            if best_bucket is not None:
                self._active_hotspot = best_bucket
                self._hotspot_direction = self._hotspot_direction_map.get(
                    best_bucket, self._recent_direction or "forward"
                )
                return

        self._active_hotspot = None
        self._hotspot_direction = None

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
                        self._refresh_hotspot_target()
                elif x_pos < self._last_mario_x:
                    self._recent_direction = "backward"
            self._last_mario_x = x_pos
        if info.get("stagnation_truncated"):
            self._register_hotspot(x_pos)

    # Gym overrides ------------------------------------------------------
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
        hotspot_intensity = self._active_hotspot_intensity()
        if prioritized and self._macro_cooldown == 0:
            threshold = max(1, self._stagnation_threshold // 3)
            if stagnation >= threshold:
                bias = self._sequence_escalation_bias
                if self._active_hotspot is not None:
                    ratio = min(1.0, stagnation / float(max(1, self._stagnation_threshold)))
                    bias = max(bias, self._hotspot_sequence_bias * ratio)
                    if hotspot_intensity >= 0.3:
                        bias = max(bias, min(1.0, 0.85 + hotspot_intensity / 2.0))
                if self.np_random.random() < bias:
                    sequence = self._select_sequence(prioritized)
                    if sequence:
                        return self._queue_macro(sequence)

        if (
            self._skill_sequences
            and stagnation >= self._stagnation_threshold
            and self._macro_cooldown == 0
        ):
            sequence_bias = min(1.0, self._skill_bias + 0.15)
            if stagnation >= self._stagnation_threshold * 2:
                sequence_bias = 1.0
            if self._active_hotspot is not None:
                sequence_bias = max(sequence_bias, self._hotspot_sequence_bias)
                if hotspot_intensity >= 0.3:
                    sequence_bias = 1.0
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
            elif hotspot_intensity >= 0.4:
                effective_epsilon = min(1.0, effective_epsilon + hotspot_intensity * 0.5)

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


__all__ = [
    "EpsilonRandomActionWrapper",
    "MacroSequenceLibrary",
]
