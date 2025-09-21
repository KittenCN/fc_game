"""Training utilities for teaching an agent to play NES games."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
except ImportError as exc:  # pragma: no cover - user guidance
    raise SystemExit(
        "Stable-Baselines3 is required. Install the RL extras via `pip install -e .[rl]`."
    ) from exc

from .rewards import REWARD_PRESETS
from .rl_utils import (
    ALGO_MAP,
    make_vector_env,
    parse_action_set,
    resolve_existing_path,
)


def _find_latest_checkpoint(log_dir: Path, algo: str) -> Optional[Path]:
    pattern = f"{algo}_agent*.zip"
    checkpoints = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RL agent on an NES ROM")
    parser.add_argument("--rom", required=True, help="Path to the .nes ROM")
    parser.add_argument("--algo", choices=sorted(ALGO_MAP.keys()), default="ppo")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=5_000)
    parser.add_argument("--observation-type", choices=["rgb", "gray"], default="gray")
    parser.add_argument(
        "--action-set",
        help="Preset name (default/simple) or custom combos like 'RIGHT;A,RIGHT;B'",
    )
    parser.add_argument(
        "--vec-env",
        choices=["auto", "dummy", "subproc"],
        default="auto",
        help="Vector environment implementation: auto chooses subproc when n_envs>1",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="Downscale observations before feeding the policy",
    )
    parser.add_argument(
        "--reward-profile",
        choices=["none", *sorted(REWARD_PRESETS.keys())],
        default="none",
        help="Reward shaping preset (default: none).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="torch device spec, e.g. cpu or cuda")
    parser.add_argument("--log-dir", default="runs", help="Directory for checkpoints and TensorBoard logs")
    parser.add_argument("--checkpoint-freq", type=int, default=200_000)
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    args = parser.parse_args()

    rom_path = resolve_existing_path(args.rom, "ROM")

    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    action_set = parse_action_set(args.action_set)
    resize_shape = tuple(args.resize) if args.resize else None
    reward_factory = None if args.reward_profile == "none" else REWARD_PRESETS[args.reward_profile]

    vec_env = make_vector_env(
        str(rom_path),
        frame_skip=args.frame_skip,
        observation_type=args.observation_type,
        action_set=action_set,
        max_episode_steps=args.max_episode_steps,
        n_envs=args.num_envs,
        seed=args.seed,
        resize_shape=resize_shape,
        vec_env_type=args.vec_env,
        reward_config_factory=reward_factory,
    )
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack, channels_order="first")

    algo_cls = ALGO_MAP[args.algo]
    tensorboard_log = str(log_dir) if args.tensorboard else None

    checkpoint_path = _find_latest_checkpoint(log_dir, args.algo)
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path.name}")
        model = algo_cls.load(str(checkpoint_path), env=vec_env, device=args.device)
        if args.tensorboard and hasattr(model, "tensorboard_log"):
            model.tensorboard_log = tensorboard_log
    else:
        model = algo_cls(
            "CnnPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=args.device,
        )

    if args.checkpoint_freq > 0:
        save_freq = max(1, args.checkpoint_freq // args.num_envs)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(log_dir),
            name_prefix=f"{args.algo}_agent",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks = [checkpoint_callback]
    else:
        callbacks = []

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks or None)

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = log_dir / f"{args.algo}_agent_{timestamp}"
    model.save(str(model_path))
    print(f"Saved trained model to {model_path.with_suffix('.zip')}")

    vec_env.close()


if __name__ == "__main__":
    main()

