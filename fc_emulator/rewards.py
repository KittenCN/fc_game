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
    progress_scale: float = 0.08,
    backward_penalty: float = 0.04,
    time_penalty: float = 0.0015,
    death_penalty: float = -12.0,
    score_scale: float = 0.015,
    stagnation_penalty: float = 6.0,
    milestone_scale: float = 0.03,
    idle_decay_penalty: float = 0.08,
    stagnation_escape_threshold: int = 180,
    stagnation_escape_bonus: float = 10.0,
    micro_progress_bonus: float = 0.45,
    powerup_bonus: float = 15.0,
    forward_hold_bonus: float = 0.35,
    forward_hold_threshold: int = 18,
    backtrack_streak_penalty: float = 0.05,
    reward_scale: float = 0.12,
) -> RewardConfig:
    """Shaping inspired by popular SMB RL projects."""

    state: Dict[str, float | int | None | set[int]] = {
        "prev_x": None,
        "prev_timer": None,
        "prev_score": None,
        "best_x": 0,
        "stagnation_steps": 0,
        "prev_player_state": None,
        "milestones": set(),
        "forward_streak": 0,
        "backtrack_streak": 0,
    }

    def on_reset() -> None:
        state["prev_x"] = None
        state["prev_timer"] = None
        state["prev_score"] = None
        state["best_x"] = 0
        state["stagnation_steps"] = 0
        state["prev_player_state"] = None
        state["milestones"] = set()
        state["forward_streak"] = 0
        state["backtrack_streak"] = 0

    def shaper(context: RewardContext) -> float:
        metrics = context.info.get("metrics", {})
        ram = context.ram

        x_pos = metrics.get("mario_x")
        if x_pos is None:
            x_pos = int(ram[0x6D]) * 256 + int(ram[0x86])

        progress_bonus = 0.0
        micro_progress_value = 0.0
        escape_bonus_value = 0.0
        forward_hold_reward = 0.0
        backtrack_penalty_value = 0.0
        prev_stagnation = int(state.get("stagnation_steps", 0))
        if state["prev_x"] is not None:
            delta_x = x_pos - int(state["prev_x"])
            if delta_x >= 0:
                progress_bonus = delta_x * progress_scale
            else:
                progress_bonus = delta_x * backward_penalty
            if delta_x > 0:
                if delta_x <= 2:
                    micro_progress_value = micro_progress_bonus
                if prev_stagnation >= stagnation_escape_threshold:
                    escape_scale = 1.0 + min(delta_x, 4) / 4.0
                    escape_bonus_value = stagnation_escape_bonus * escape_scale
                state["stagnation_steps"] = 0
                state["forward_streak"] = min(
                    forward_hold_threshold,
                    int(state.get("forward_streak", 0)) + int(delta_x),
                )
                if state["forward_streak"] >= forward_hold_threshold:
                    forward_hold_reward = forward_hold_bonus
                state["backtrack_streak"] = max(
                    0,
                    int(state.get("backtrack_streak", 0)) - int(delta_x),
                )
            else:
                state["stagnation_steps"] = prev_stagnation + 1
                state["forward_streak"] = max(0, int(state.get("forward_streak", 0)) - 1)
                state["backtrack_streak"] = int(state.get("backtrack_streak", 0)) + abs(delta_x)
        else:
            state["stagnation_steps"] = 0
            state["forward_streak"] = 0
            state["backtrack_streak"] = 0
        state["prev_x"] = x_pos

        if state["backtrack_streak"]:
            backtrack_penalty_value = min(24, int(state["backtrack_streak"])) * backtrack_streak_penalty

        milestone_bonus = 0.0
        best_x = int(state.get("best_x", 0))
        if x_pos > best_x:
            milestone_bonus = (x_pos - best_x) * milestone_scale
            state["best_x"] = x_pos

        milestone_reward = 0.0
        reached: set[int] = state["milestones"]  # type: ignore[assignment]
        milestone_targets = ((512, 5.0), (768, 7.5), (1024, 10.0), (1536, 12.0))
        for threshold, bonus in milestone_targets:
            if x_pos >= threshold and threshold not in reached:
                milestone_reward += bonus
                reached.add(threshold)

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

        player_state = metrics.get("player_state")
        powerup_reward = 0.0
        if (
            player_state is not None
            and state["prev_player_state"] is not None
            and player_state > state["prev_player_state"]
        ):
            powerup_reward = powerup_bonus
        state["prev_player_state"] = player_state

        idle_penalty_value = -idle_decay_penalty * (state["stagnation_steps"] // 120)

        shaped_reward = (
            context.base_reward
            + progress_bonus
            + score_bonus
            + time_penalty_value
            + milestone_bonus
            + idle_penalty_value
            + micro_progress_value
            + escape_bonus_value
            + powerup_reward
            + forward_hold_reward
            + milestone_reward
            - backtrack_penalty_value
        )

        shaped_reward *= reward_scale

        if context.info.get("stagnation_truncated"):
            bucket = metrics.get("stagnation_bucket")
            penalty_scale = 1.0
            if bucket is not None:
                normalised = min(max(int(bucket), 0), 1536) / 1536.0
                penalty_scale = max(0.35, 1.0 - 0.45 * normalised)
            shaped_reward -= stagnation_penalty * penalty_scale

        if context.done and context.base_reward <= 0:
            shaped_reward += death_penalty

        diagnostics = context.info.setdefault("diagnostics", {})
        diagnostics["forward_streak"] = int(state.get("forward_streak", 0))
        diagnostics["backtrack_streak"] = int(state.get("backtrack_streak", 0))
        diagnostics["reward_scale"] = float(reward_scale)

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
    stagnation_escape_threshold: int = 150,
    stagnation_escape_bonus: float = 20.0,
    micro_progress_bonus: float = 0.6,
    powerup_bonus: float = 20.0,
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
        "prev_player_state": None,
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
            "prev_player_state": None,
        })

    def shaper(context: RewardContext) -> float:
        metrics = context.info.get("metrics", {})
        ram = context.ram

        x_pos = metrics.get("mario_x")
        if x_pos is None:
            x_pos = int(ram[0x6D]) * 256 + int(ram[0x86])

        prev_idle = int(state["idle_frames"])
        if state["prev_x"] is None:
            state["prev_x"] = x_pos
        delta_x = x_pos - int(state["prev_x"])
        state["prev_x"] = x_pos

        progress = 0.0
        micro_progress_value = 0.0
        escape_bonus_value = 0.0
        if delta_x > 0:
            progress = delta_x * progress_scale
            if delta_x <= 3:
                micro_progress_value = micro_progress_bonus
            if prev_idle >= stagnation_escape_threshold:
                escape_scale = 1.0 + min(delta_x, 6) / 6.0
                escape_bonus_value = stagnation_escape_bonus * escape_scale
            state["idle_frames"] = 0
        elif delta_x < 0:
            progress = delta_x * backward_penalty
            state["idle_frames"] = prev_idle + 1
        else:
            state["idle_frames"] = prev_idle + 1

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

        player_state = metrics.get("player_state")
        powerup_reward = 0.0
        if (
            player_state is not None
            and state["prev_player_state"] is not None
            and player_state > state["prev_player_state"]
        ):
            powerup_reward = powerup_bonus
        state["prev_player_state"] = player_state

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

        shaped = (
            context.base_reward
            + progress
            + milestone
            + score_bonus
            + time_penalty_value
            + idle_pen
            + level_bonus
            + micro_progress_value
            + escape_bonus_value
            + powerup_reward
        )

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
