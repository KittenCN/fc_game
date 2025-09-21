"""Shared helpers for RL training and inference."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import gymnasium as gym

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv
except ImportError as exc:  # pragma: no cover - user guidance
    raise ImportError(
        "Stable-Baselines3 is required. Install the RL extras via `pip install -e .[rl]`."
    ) from exc

from fc_emulator.rl_env import NESGymEnv
from fc_emulator.wrappers import ACTION_PRESETS, DiscreteActionWrapper, DEFAULT_ACTION_SET

ALGO_MAP = {
    "ppo": PPO,
    "a2c": A2C,
}


def build_env(
    rom_path: str,
    *,
    frame_skip: int,
    observation_type: str,
    action_set: Sequence[Sequence[str]],
    max_episode_steps: int | None,
    render_mode: str | None,
) -> gym.Env:
    env = NESGymEnv(
        rom_path,
        frame_skip=frame_skip,
        observation_type=observation_type,
        render_mode=render_mode,
    )
    env = DiscreteActionWrapper(env, action_set=action_set)
    if max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def make_vector_env(
    rom_path: str,
    *,
    frame_skip: int,
    observation_type: str,
    action_set: Sequence[Sequence[str]] | None = None,
    max_episode_steps: int | None,
    n_envs: int,
    seed: int | None,
    render_mode: str | None = None,
) -> VecEnv:
    chosen_action_set = action_set or DEFAULT_ACTION_SET
    return make_vec_env(
        build_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=dict(
            rom_path=rom_path,
            frame_skip=frame_skip,
            observation_type=observation_type,
            action_set=chosen_action_set,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
        ),
    )


def parse_action_set(value: str | None):
    if value is None:
        return DEFAULT_ACTION_SET
    preset = ACTION_PRESETS.get(value.lower())
    if preset is not None:
        return preset
    combos: list[tuple[str, ...]] = []
    for item in value.split(";"):
        combo = tuple(part.strip().upper() for part in item.split(",") if part.strip())
        combos.append(combo)
    if not combos:
        raise ValueError("Parsed action set is empty")
    return tuple(combos)


def resolve_existing_path(path_str: str, description: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"{description} not found: {path}")
    return path
