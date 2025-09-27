"""Adaptive stagnation tracking for NES environments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StagnationConfig:
    base_frames: int | None = 900
    progress_threshold: int = 1
    bonus_scale: float = 0.25
    micro_relief_ratio: float = 0.5
    score_relief_base: int = 6
    powerup_relief: int = 45
    hotspot_bucket: int = 32


@dataclass
class StagnationStatus:
    counter: int
    limit: int | None
    triggered: bool
    frames: int | None
    reason: str | None
    event: str | None
    position: int | None
    bucket: int | None


class StagnationMonitor:
    """Track forward progress and decide when to early-terminate episodes."""

    def __init__(self, config: StagnationConfig) -> None:
        self.config = config
        self._counter = 0
        self._limit: int | None = None
        self._max_progress_x = 0
        self._last_progress_x: int | None = None
        self._last_score: int | None = None
        self._last_player_state: int | None = None
        self._last_world: int | None = None
        self._last_stage: int | None = None
        self._last_area: int | None = None
        self._last_reason: str | None = None

    @property
    def counter(self) -> int:
        return self._counter

    @property
    def limit(self) -> int | None:
        return self._limit

    @staticmethod
    def _get_mario_x(metrics: dict[str, Any], ram: np.ndarray) -> int:
        x_pos = metrics.get("mario_x")
        if x_pos is None:
            x_pos = int(ram[0x6D]) * 256 + int(ram[0x86])
        return int(x_pos)

    def _compute_limit(self) -> int | None:
        base = self.config.base_frames
        if base is None or base <= 0:
            return None
        bonus = int(self._max_progress_x * self.config.bonus_scale)
        bonus = max(0, min(base, bonus))
        return int(base + bonus)

    def _bucketise(self, position: int | None) -> int | None:
        if position is None:
            return None
        bucket = max(0, int(position))
        size = max(1, int(self.config.hotspot_bucket))
        return (bucket // size) * size

    def reset(self, metrics: dict[str, Any], ram: np.ndarray) -> None:
        self._counter = 0
        self._max_progress_x = 0
        self._last_progress_x = self._get_mario_x(metrics, ram)
        self._max_progress_x = self._last_progress_x or 0
        self._last_score = metrics.get("score")
        self._last_player_state = metrics.get("player_state")
        self._last_world = metrics.get("world")
        self._last_stage = metrics.get("stage")
        self._last_area = metrics.get("area")
        self._limit = self._compute_limit()
        self._last_reason = None

    def update(self, metrics: dict[str, Any], ram: np.ndarray, *, frame_skip: int) -> StagnationStatus:
        if self.config.base_frames is None or self.config.base_frames <= 0:
            self._counter = 0
            self._limit = None
            self._last_reason = None
            return StagnationStatus(
                counter=0,
                limit=None,
                triggered=False,
                frames=None,
                reason=None,
                event=None,
                position=current_x,
                bucket=self._bucketise(current_x),
            )

        current_x = self._get_mario_x(metrics, ram)
        reason = "no_progress"
        progressed = False

        if self._last_progress_x is None:
            self._last_progress_x = current_x
            self._max_progress_x = max(self._max_progress_x, current_x)
            self._limit = self._compute_limit()
            return StagnationStatus(
                counter=self._counter,
                limit=self._limit,
                triggered=False,
                frames=None,
                reason=None,
                event=None,
                position=current_x,
                bucket=self._bucketise(current_x),
            )

        delta_x = current_x - self._last_progress_x
        if delta_x > 0:
            progressed = True
            if delta_x >= self.config.progress_threshold:
                self._counter = 0
                reason = "forward_progress"
            else:
                relief = int(frame_skip * self.config.micro_relief_ratio)
                self._counter = max(0, self._counter - relief)
                reason = "micro_progress"
            self._last_progress_x = current_x
            self._max_progress_x = max(self._max_progress_x, current_x)
        elif delta_x < 0:
            reason = "backtrack"

        world = metrics.get("world")
        stage = metrics.get("stage")
        area = metrics.get("area")

        if (
            (world is not None and world != self._last_world)
            or (stage is not None and stage != self._last_stage)
            or (area is not None and area != self._last_area)
        ):
            self._counter = 0
            progressed = True
            reason = "level_transition"
            self._last_progress_x = current_x
            self._max_progress_x = max(self._max_progress_x, current_x)

        if world is not None:
            self._last_world = world
        if stage is not None:
            self._last_stage = stage
        if area is not None:
            self._last_area = area

        relief_frames = 0

        score = metrics.get("score")
        if score is not None and self._last_score is not None:
            delta_score = score - self._last_score
            if delta_score > 0 and not progressed:
                relief_frames = max(
                    relief_frames,
                    frame_skip * (self.config.score_relief_base + min(delta_score, 50) // 2),
                )
                reason = "score_event"
        self._last_score = score

        player_state = metrics.get("player_state")
        if player_state is not None and self._last_player_state is not None:
            if player_state > self._last_player_state:
                relief_frames = max(relief_frames, frame_skip * self.config.powerup_relief)
                reason = "powerup"
        self._last_player_state = player_state

        if relief_frames:
            self._counter = max(0, self._counter - relief_frames)
            progressed = True

        if not progressed:
            self._counter += frame_skip

        self._limit = self._compute_limit()
        triggered = False
        frames = None
        event = reason
        if self._limit is not None and self._counter >= self._limit:
            triggered = True
            frames = self._counter
            self._counter = 0
            self._last_progress_x = current_x
            reason = "stagnation"

        self._last_reason = reason
        return StagnationStatus(
            counter=self._counter,
            limit=self._limit,
            triggered=triggered,
            frames=frames,
            reason=reason,
            event=event,
            position=current_x,
            bucket=self._bucketise(current_x),
        )


__all__ = ["StagnationConfig", "StagnationMonitor", "StagnationStatus"]
