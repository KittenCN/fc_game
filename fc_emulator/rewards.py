"""Reward shaping helpers for NES environments."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .rl_env import RewardConfig, RewardContext


def _decode_timer(ram: np.ndarray) -> int | None:
    try:
        hundreds = int(ram[0x07F8]) & 0x0F
        tens = int(ram[0x07F9]) & 0x0F
        ones = int(ram[0x07FA]) & 0x0F
    except IndexError:
        return None
    return hundreds * 100 + tens * 10 + ones


def _decode_score(ram: np.ndarray) -> int | None:
    digits = []
    for addr in range(0x07DE, 0x07E4):
        try:
            digits.append(int(ram[addr]) & 0x0F)
        except IndexError:
            return None
    score = 0
    for digit in digits:
        score = score * 10 + digit
    return score


def make_super_mario_progress_reward(
    *,
    progress_scale: float = 0.05,
    backward_penalty: float = 0.1,
    time_penalty: float = 0.01,
    death_penalty: float = -25.0,
    score_scale: float = 0.01,
) -> RewardConfig:
    """Shaping inspired by popular SMB RL projects.

    Encourages horizontal progress, slight penalty for idling/backtracking,
    and small reward for increasing score (coins, stomps). Death or timeout
    yields an additional penalty to push exploration forward.
    """

    state: Dict[str, float | int | None] = {
        "prev_x": None,
        "prev_timer": None,
        "prev_score": None,
    }

    def on_reset() -> None:
        state["prev_x"] = None
        state["prev_timer"] = None
        state["prev_score"] = None

    def shaper(context: RewardContext) -> float:
        metrics = context.info.get("metrics", {})
        ram = context.ram

        # Horizontal progress ------------------------------------------------
        x_pos = metrics.get("mario_x")
        if x_pos is None:
            x_pos = int(ram[0x6D]) * 256 + int(ram[0x86])
        progress_bonus = 0.0
        if state["prev_x"] is not None:
            delta_x = x_pos - int(state["prev_x"])
            if delta_x >= 0:
                progress_bonus = delta_x * progress_scale
            else:
                progress_bonus = delta_x * backward_penalty
        state["prev_x"] = x_pos

        # Score changes ------------------------------------------------------
        score = metrics.get("score")
        if score is None:
            score = _decode_score(ram)
        score_bonus = 0.0
        if score is not None and state["prev_score"] is not None:
            delta_score = score - int(state["prev_score"])
            if delta_score > 0:
                score_bonus = delta_score * score_scale
        state["prev_score"] = score

        # Timer decay penalizes stalling -------------------------------------
        timer = metrics.get("timer")
        if timer is None:
            timer = _decode_timer(ram)
        time_penalty_value = 0.0
        if timer is not None and state["prev_timer"] is not None:
            elapsed = int(state["prev_timer"]) - timer
            if elapsed > 0:
                time_penalty_value = -elapsed * time_penalty
        state["prev_timer"] = timer

        shaped_reward = context.base_reward + progress_bonus + score_bonus + time_penalty_value

        if context.done and context.base_reward <= 0:
            shaped_reward += death_penalty

        return shaped_reward

    return RewardConfig(func=shaper, on_reset=on_reset)


REWARD_PRESETS: Dict[str, Callable[[], RewardConfig]] = {
    "smb_progress": make_super_mario_progress_reward,
}

__all__ = ["make_super_mario_progress_reward", "REWARD_PRESETS"]
