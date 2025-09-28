"""Shared helpers for RL training and inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import gymnasium as gym

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv
except ImportError as exc:  # pragma: no cover - user guidance
    raise ImportError(
        "Stable-Baselines3 is required. Install the RL extras via pip install -e .[rl]."
    ) from exc

from fc_emulator.actions import DEFAULT_ACTION_SET, DiscreteActionWrapper, resolve_action_set
from fc_emulator.exploration import EpsilonRandomActionWrapper, MacroSequenceLibrary
from fc_emulator.observation import (
    ResizeObservationWrapper,
    VecFrameStackPixelsDictWrapper,
    VecTransposePixelsDictWrapper,
)
from fc_emulator.rl_env import NESGymEnv, RewardConfig

ALGO_MAP = {
    "ppo": PPO,
    "a2c": A2C,
}

try:  # pragma: no cover - optional dependency
    from sb3_contrib import RecurrentPPO
except ImportError:  # pragma: no cover - optional dependency
    RecurrentPPO = None
else:  # pragma: no cover - optional dependency
    ALGO_MAP["rppo"] = RecurrentPPO


@dataclass(frozen=True)
class SkillSequences:
    """Categorised macro action sequences derived from the action set."""

    all: tuple[tuple[int, ...], ...]
    forward: tuple[tuple[int, ...], ...]
    backward: tuple[tuple[int, ...], ...]
    neutral: tuple[tuple[int, ...], ...]


def _derive_skill_action_indices(action_set: Sequence[Sequence[str]]) -> tuple[int, ...]:
    skill_indices: list[int] = []
    for idx, combo in enumerate(action_set):
        normalized = {btn.upper() for btn in combo}
        if not normalized:
            continue
        if "A" in normalized:
            skill_indices.append(idx)
        elif "B" in normalized and "RIGHT" in normalized:
            skill_indices.append(idx)
    return tuple(skill_indices)


def _derive_skill_sequences(action_set: Sequence[Sequence[str]]) -> SkillSequences:
    normalized = [tuple(btn.upper() for btn in combo) for combo in action_set]
    index_lookup = {frozenset(combo): idx for idx, combo in enumerate(normalized)}

    def find_index(buttons: tuple[str, ...]) -> int | None:
        return index_lookup.get(frozenset(buttons))

    def first_valid(*values: int | None) -> int | None:
        for value in values:
            if value is not None:
                return value
        return None

    sequences: list[tuple[int, ...]] = []
    forward_sequences: list[tuple[int, ...]] = []
    backward_sequences: list[tuple[int, ...]] = []
    neutral_sequences: list[tuple[int, ...]] = []

    def ensure_sequence(
        indices: tuple[int | None, ...],
        *,
        direction: str | None,
    ) -> None:
        if any(idx is None for idx in indices):
            return
        typed = tuple(int(idx) for idx in indices)
        if not typed or typed in sequences:
            return
        sequences.append(typed)
        if direction == "forward":
            forward_sequences.append(typed)
        elif direction == "backward":
            backward_sequences.append(typed)
        else:
            neutral_sequences.append(typed)

    def combo(*buttons: str) -> tuple[str, ...]:
        return tuple(btn.upper() for btn in buttons if btn)

    run_right = first_valid(find_index(combo("B", "RIGHT")), find_index(combo("RIGHT")))
    run_jump_right = first_valid(
        find_index(combo("A", "B", "RIGHT")),
        find_index(combo("A", "RIGHT")),
    )
    short_jump_right = find_index(combo("A", "RIGHT"))
    ensure_sequence((run_right, run_right, run_jump_right, run_jump_right), direction="forward")
    ensure_sequence((run_right, short_jump_right, short_jump_right), direction="forward")
    if run_right is not None and run_jump_right is not None:
        ensure_sequence(tuple([run_right] * 3 + [run_jump_right] * 3), direction="forward")
        ensure_sequence(tuple([run_right] * 5 + [run_jump_right] * 5), direction="forward")
        ensure_sequence(tuple([run_right] * 6 + [run_jump_right] * 6), direction="forward")
        ensure_sequence(tuple([run_right] * 4 + [run_jump_right] * 8), direction="forward")
        ensure_sequence(tuple([run_right] * 10 + [run_jump_right] * 6), direction="forward")
    if run_right is not None:
        ensure_sequence(tuple([run_right] * 8), direction="forward")
        ensure_sequence(tuple([run_right] * 12), direction="forward")
    if run_right is not None and short_jump_right is not None:
        ensure_sequence(tuple([run_right] * 2 + [short_jump_right] * 3), direction="forward")
        ensure_sequence(tuple([run_right] * 3 + [short_jump_right] * 4 + [run_right] * 2), direction="forward")

    run_left = first_valid(find_index(combo("B", "LEFT")), find_index(combo("LEFT")))
    run_jump_left = first_valid(
        find_index(combo("A", "B", "LEFT")),
        find_index(combo("A", "LEFT")),
    )
    short_jump_left = find_index(combo("A", "LEFT"))
    ensure_sequence((run_left, run_left, run_jump_left, run_jump_left), direction="backward")
    ensure_sequence((run_left, short_jump_left, short_jump_left), direction="backward")
    if run_left is not None and run_jump_left is not None:
        ensure_sequence(tuple([run_left] * 3 + [run_jump_left] * 3), direction="backward")
        ensure_sequence(tuple([run_left] * 5 + [run_jump_left] * 5), direction="backward")
        ensure_sequence(tuple([run_left] * 6 + [run_jump_left] * 6), direction="backward")
        ensure_sequence(tuple([run_left] * 4 + [run_jump_left] * 8), direction="backward")
        ensure_sequence(tuple([run_left] * 10 + [run_jump_left] * 6), direction="backward")
    if run_left is not None:
        ensure_sequence(tuple([run_left] * 8), direction="backward")
        ensure_sequence(tuple([run_left] * 12), direction="backward")
    if run_left is not None and short_jump_left is not None:
        ensure_sequence(tuple([run_left] * 2 + [short_jump_left] * 3), direction="backward")
        ensure_sequence(tuple([run_left] * 3 + [short_jump_left] * 4 + [run_left] * 2), direction="backward")

    neutral_jump = first_valid(find_index(combo("A")), short_jump_right, short_jump_left)
    if neutral_jump is not None:
        ensure_sequence((neutral_jump, neutral_jump), direction=None)

    down = first_valid(find_index(combo("DOWN")), find_index(combo("RIGHT", "DOWN")))
    if down is not None:
        ensure_sequence((down, down, down, down), direction=None)

    return SkillSequences(
        all=tuple(sequences),
        forward=tuple(forward_sequences),
        backward=tuple(backward_sequences),
        neutral=tuple(neutral_sequences),
    )


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
    stagnation_bonus_scale: float,
    stagnation_idle_multiplier: float,
    stagnation_backtrack_penalty_scale: float,
    stagnation_backtrack_stop_ratio: float,
    stagnation_backtrack_stop_min_progress: int,
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
        stagnation_bonus_scale=stagnation_bonus_scale,
        stagnation_idle_multiplier=stagnation_idle_multiplier,
        stagnation_backtrack_penalty_scale=stagnation_backtrack_penalty_scale,
        stagnation_backtrack_stop_ratio=stagnation_backtrack_stop_ratio,
        stagnation_backtrack_stop_min_progress=stagnation_backtrack_stop_min_progress,
    )
    if resize_shape and observation_type in {"rgb", "gray", "rgb_ram", "gray_ram"}:
        env = ResizeObservationWrapper(env, resize_shape)
    env = DiscreteActionWrapper(env, action_set=action_set)
    if exploration_epsilon > 0.0:
        skill_actions = _derive_skill_action_indices(action_set)
        skill_sequences = _derive_skill_sequences(action_set)
        stagnation_threshold = 120
        if stagnation_max_frames:
            candidate = max(45, int(stagnation_max_frames) // 5)
            stagnation_threshold = max(45, min(candidate, int(stagnation_max_frames)))
        sequence_library = MacroSequenceLibrary(
            skill_actions=skill_actions,
            skill_sequences=skill_sequences.all,
            forward_sequences=skill_sequences.forward,
            backward_sequences=skill_sequences.backward,
            neutral_sequences=skill_sequences.neutral,
        )
        env = EpsilonRandomActionWrapper(
            env,
            exploration_epsilon,
            sequences=sequence_library,
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
    stagnation_bonus_scale: float = 0.15,
    stagnation_idle_multiplier: float = 1.1,
    stagnation_backtrack_penalty_scale: float = 1.0,
    stagnation_backtrack_stop_ratio: float = 0.7,
    stagnation_backtrack_stop_min_progress: int = 128,
    frame_stack: int = 4,
    use_icm: bool = False,
    icm_kwargs: dict | None = None,
) -> VecEnv:
    chosen_action_set = action_set or DEFAULT_ACTION_SET
    vec_cls = _select_vec_env_cls(vec_env_type, n_envs)
    env = make_vec_env(
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
            stagnation_bonus_scale=stagnation_bonus_scale,
            stagnation_idle_multiplier=stagnation_idle_multiplier,
            stagnation_backtrack_penalty_scale=stagnation_backtrack_penalty_scale,
            stagnation_backtrack_stop_ratio=stagnation_backtrack_stop_ratio,
            stagnation_backtrack_stop_min_progress=stagnation_backtrack_stop_min_progress,
        ),
        vec_env_cls=vec_cls,
    )

    if observation_type in {"rgb", "gray"}:
        env = VecTransposeImage(env)
        if frame_stack > 1:
            env = VecFrameStack(env, n_stack=frame_stack, channels_order="first")
    elif observation_type in {"rgb_ram", "gray_ram"}:
        env = VecTransposePixelsDictWrapper(env)
        if frame_stack > 1:
            env = VecFrameStackPixelsDictWrapper(env, n_stack=frame_stack)

    if use_icm:
        from fc_emulator.icm import ICMVecEnvWrapper  # pragma: no cover - optional dependency

        icm_cfg = dict(icm_kwargs or {})
        env = ICMVecEnvWrapper(env, **icm_cfg)

    return env


def parse_action_set(value: str | None):
    if value is None:
        return DEFAULT_ACTION_SET
    if ";" not in value and "," not in value:
        try:
            return resolve_action_set(value)
        except KeyError:
            pass
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
