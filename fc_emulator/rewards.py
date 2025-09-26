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
    stagnation_penalty: float = 10.0,
) -> RewardConfig:
    """Shaping inspired by popular SMB RL projects."""

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

        if context.info.get("stagnation_truncated"):
            shaped_reward -= stagnation_penalty

        if context.done and context.base_reward <= 0:
            shaped_reward += death_penalty

        return shaped_reward

    return RewardConfig(func=shaper, on_reset=on_reset)


def make_super_mario_dense_reward(
    *,
    progress_scale: float = 0.5,
    backward_penalty: float = 1.0,
    milestone_bonus: float = 5.0,
    idle_penalty: float = 1.0,
    idle_threshold: int = 15,
    time_penalty: float = 0.02,
    death_penalty: float = -50.0,
    score_scale: float = 0.02,
    stagnation_penalty: float = 20.0,
) -> RewardConfig:
    """Aggressive shaping that heavily favours forward motion and exploration."""

    state: Dict[str, float | int | None] = {
        "prev_x": None,
        "best_x": 0,
        "prev_timer": None,
        "prev_score": None,
        "idle_frames": 0,
        "last_world": None,
        "last_stage": None,
    }

    def on_reset() -> None:
        state.update({
            "prev_x": None,
            "best_x": 0,
            "prev_timer": None,
            "prev_score": None,
            "idle_frames": 0,
            "last_world": None,
            "last_stage": None,
        })

    def shaper(context: RewardContext) -> float:
        metrics = context.info.get("metrics", {})
        ram = context.ram

        x_pos = metrics.get("mario_x")
        if x_pos is None:
            x_pos = int(ram[0x6D]) * 256 + int(ram[0x86])

        prev_x = state["prev_x"] if state["prev_x"] is not None else x_pos
        delta_x = x_pos - int(prev_x)
        state["prev_x"] = x_pos

        if delta_x > 0:
            progress = delta_x * progress_scale
            state["idle_frames"] = 0
        elif delta_x < 0:
            progress = delta_x * backward_penalty
            state["idle_frames"] += 1
        else:
            progress = 0.0
            state["idle_frames"] += 1

        idle_pen = 0.0
        if state["idle_frames"] >= idle_threshold:
            multiples = state["idle_frames"] // idle_threshold
            idle_pen = -idle_penalty * multiples
            state["idle_frames"] -= idle_threshold * multiples

        best_x = int(state["best_x"])
        milestone = 0.0
        if x_pos > best_x:
            milestone = (x_pos - best_x) * (milestone_bonus / 50.0)
            state["best_x"] = x_pos

        score = metrics.get("score")
        if score is None:
            score = _decode_score(ram)
        score_bonus = 0.0
        if score is not None and state["prev_score"] is not None:
            delta_score = score - int(state["prev_score"])
            if delta_score > 0:
                score_bonus = delta_score * score_scale
        state["prev_score"] = score

        timer = metrics.get("timer")
        if timer is None:
            timer = _decode_timer(ram)
        time_penalty_value = 0.0
        if timer is not None and state["prev_timer"] is not None:
            elapsed = int(state["prev_timer"]) - timer
            if elapsed > 0:
                time_penalty_value = -elapsed * time_penalty
        state["prev_timer"] = timer

        world = metrics.get("world")
        stage = metrics.get("stage")
        level_bonus = 0.0
        if world is not None and stage is not None:
            if state["last_world"] is None:
                state["last_world"] = world
            if state["last_stage"] is None:
                state["last_stage"] = stage
            if world > state["last_world"] or (world == state["last_world"] and stage > state["last_stage"]):
                level_bonus = 100.0
                state["last_world"] = world
                state["last_stage"] = stage

        shaped = context.base_reward + progress + milestone + score_bonus + time_penalty_value + idle_pen + level_bonus

        if context.info.get("stagnation_truncated"):
            shaped -= stagnation_penalty

        if context.done and context.base_reward <= 0:
            shaped += death_penalty

        return shaped

    return RewardConfig(func=shaper, on_reset=on_reset)


REWARD_PRESETS: Dict[str, Callable[[], RewardConfig]] = {
    "smb_progress": make_super_mario_progress_reward,
    "smb_dense": make_super_mario_dense_reward,
}

__all__ = [
    "make_super_mario_progress_reward",
    "make_super_mario_dense_reward",
    "REWARD_PRESETS",
]
