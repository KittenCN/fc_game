"""Shared helpers for RL training and inference."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, Tuple

import gymnasium as gym

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv
except ImportError as exc:  # pragma: no cover - user guidance
    raise ImportError(
        "Stable-Baselines3 is required. Install the RL extras via `pip install -e .[rl]`."
    ) from exc

from fc_emulator.rl_env import NESGymEnv, RewardConfig
from fc_emulator.wrappers import (
    ACTION_PRESETS,
    DiscreteActionWrapper,
    DEFAULT_ACTION_SET,
    ResizeObservationWrapper,
    EpsilonRandomActionWrapper,
)

ALGO_MAP = {
    "ppo": PPO,
    "a2c": A2C,
}

def _derive_skill_action_indices(action_set: Sequence[Sequence[str]]) -> tuple[int, ...]:
    skill_indices: list[int] = []
    for idx, combo in enumerate(action_set):
        normalized = {btn.upper() for btn in combo}
        # An empty combo is not a skill
        if not normalized:
            continue
        # Any jump is a skill
        if "A" in normalized:
            skill_indices.append(idx)
        # Running is a skill, especially when combined with other buttons
        elif "B" in normalized and "RIGHT" in normalized:
            skill_indices.append(idx)
    return tuple(skill_indices)

def _derive_skill_sequences(action_set: Sequence[Sequence[str]]) -> tuple[tuple[int, ...], ...]:
    normalized = [tuple(btn.upper() for btn in combo) for combo in action_set]
    index_lookup = {frozenset(combo): idx for idx, combo in enumerate(normalized)}

    def find_index(buttons: tuple[str, ...]) -> int | None:
        return index_lookup.get(frozenset(buttons))

    def first_valid(*values: int | None) -> int | None:
        for value in values:
            if value is not None:
                return value
        return None

    def ensure_sequence(indices: tuple[int | None, ...], store: list[tuple[int, ...]]):
        if any(idx is None for idx in indices):
            return
        typed = tuple(int(idx) for idx in indices)
        if typed and typed not in store:
            store.append(typed)

    sequences: list[tuple[int, ...]] = []

    def combo(*buttons: str) -> tuple[str, ...]:
        return tuple(btn.upper() for btn in buttons if btn)

    run_right = first_valid(find_index(combo("B", "RIGHT")), find_index(combo("RIGHT")))
    run_jump_right = first_valid(find_index(combo("A", "B", "RIGHT")), find_index(combo("A", "RIGHT")))
    short_jump_right = find_index(combo("A", "RIGHT"))
    ensure_sequence((run_right, run_right, run_jump_right, run_jump_right), sequences)
    ensure_sequence((run_right, short_jump_right, short_jump_right), sequences)
    if run_right is not None and run_jump_right is not None:
        ensure_sequence(tuple([run_right] * 3 + [run_jump_right] * 3), sequences)
        ensure_sequence(tuple([run_right] * 5 + [run_jump_right] * 5), sequences)
    if run_right is not None and short_jump_right is not None:
        ensure_sequence(tuple([run_right] * 2 + [short_jump_right] * 3), sequences)

    run_left = first_valid(find_index(combo("B", "LEFT")), find_index(combo("LEFT")))
    run_jump_left = first_valid(find_index(combo("A", "B", "LEFT")), find_index(combo("A", "LEFT")))
    short_jump_left = find_index(combo("A", "LEFT"))
    ensure_sequence((run_left, run_left, run_jump_left, run_jump_left), sequences)
    ensure_sequence((run_left, short_jump_left, short_jump_left), sequences)
    if run_left is not None and run_jump_left is not None:
        ensure_sequence(tuple([run_left] * 3 + [run_jump_left] * 3), sequences)
        ensure_sequence(tuple([run_left] * 5 + [run_jump_left] * 5), sequences)
    if run_left is not None and short_jump_left is not None:
        ensure_sequence(tuple([run_left] * 2 + [short_jump_left] * 3), sequences)

    neutral_jump = first_valid(find_index(combo("A")), short_jump_right, short_jump_left)
    if neutral_jump is not None:
        ensure_sequence((neutral_jump, neutral_jump), sequences)

    down = first_valid(find_index(combo("DOWN")), find_index(combo("RIGHT", "DOWN")))
    if down is not None:
        ensure_sequence((down, down, down, down), sequences)
        ensure_sequence(tuple([down] * 8), sequences)

    return tuple(sequences)







def build_env(
    rom_path: str,
    *,
    frame_skip: int,
    observation_type: str,
    action_set: Sequence[Sequence[str]],
    max_episode_steps: int | None,
    render_mode: str | None,
    resize_shape: tuple[int, int] | None,
    reward_config_factory: Callable[[], RewardConfig | None] | None,
    auto_start: bool,
    auto_start_max_frames: int,
    auto_start_press_frames: int,
    exploration_epsilon: float,
    stagnation_max_frames: int | None,
    stagnation_progress_threshold: int,
) -> gym.Env:
    reward_cfg = reward_config_factory() if reward_config_factory else None
    env = NESGymEnv(
        rom_path,
        frame_skip=frame_skip,
        observation_type=observation_type,
        render_mode=render_mode,
        reward_config=reward_cfg,
        auto_start=auto_start,
        auto_start_max_frames=auto_start_max_frames,
        auto_start_press_frames=auto_start_press_frames,
        stagnation_max_frames=stagnation_max_frames,
        stagnation_progress_threshold=stagnation_progress_threshold,
    )
    if resize_shape and observation_type in {"rgb", "gray"}:
        env = ResizeObservationWrapper(env, resize_shape)
    env = DiscreteActionWrapper(env, action_set=action_set)
    if exploration_epsilon > 0.0:
        skill_actions = _derive_skill_action_indices(action_set)
        skill_sequences = _derive_skill_sequences(action_set)
        stagnation_threshold = 120
        if stagnation_max_frames:
            candidate = max(60, int(stagnation_max_frames) // 4)
            stagnation_threshold = max(60, min(candidate, int(stagnation_max_frames)))
        env = EpsilonRandomActionWrapper(
            env,
            exploration_epsilon,
            skill_actions=skill_actions,
            skill_sequences=skill_sequences,
            stagnation_threshold=stagnation_threshold,
            stagnation_boost=0.5,
        )
    if max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def _select_vec_env_cls(vec_env_type: str, n_envs: int):
    if vec_env_type == "dummy":
        return DummyVecEnv
    if vec_env_type == "subproc":
        return SubprocVecEnv
    if vec_env_type == "auto":
        return SubprocVecEnv if n_envs > 1 else DummyVecEnv
    raise ValueError(f"Unknown vec_env_type: {vec_env_type}")


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
    resize_shape: tuple[int, int] | None = None,
    vec_env_type: str = "auto",
    reward_config_factory: Callable[[], RewardConfig | None] | None = None,
    auto_start: bool = True,
    auto_start_max_frames: int = 120,
    auto_start_press_frames: int = 6,
    exploration_epsilon: float = 0.05,
    stagnation_max_frames: int | None = 900,
    stagnation_progress_threshold: int = 1,
) -> VecEnv:
    chosen_action_set = action_set or DEFAULT_ACTION_SET
    vec_cls = _select_vec_env_cls(vec_env_type, n_envs)
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
            resize_shape=resize_shape,
            reward_config_factory=reward_config_factory,
            auto_start=auto_start,
            auto_start_max_frames=auto_start_max_frames,
            auto_start_press_frames=auto_start_press_frames,
            exploration_epsilon=exploration_epsilon,
            stagnation_max_frames=stagnation_max_frames,
            stagnation_progress_threshold=stagnation_progress_threshold,
        ),
        vec_env_cls=vec_cls,
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
