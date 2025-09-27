"""Training utilities for teaching an agent to play NES games."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    from stable_baselines3.common.callbacks import CheckpointCallback
except ImportError as exc:  # pragma: no cover - user guidance
    raise SystemExit(
        "Stable-Baselines3 is required. Install the RL extras via pip install -e .[rl]."
    ) from exc

from .callbacks import EpisodeLogCallback, ExplorationEpsilonCallback, EntropyCoefficientCallback
from .policies import POLICY_PRESETS
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
    parser.add_argument(
        "--observation-type",
        choices=["rgb", "gray", "rgb_ram", "gray_ram", "ram"],
        default="gray",
    )
    parser.add_argument(
        "--action-set",
        help="Preset name (default/simple/smb_forward) or custom combos like 'RIGHT;A,RIGHT;B'",
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
    parser.add_argument(
        "--disable-auto-start",
        action="store_true",
        help="Disable automatically pressing START after each reset.",
    )
    parser.add_argument(
        "--auto-start-max-frames",
        type=int,
        default=120,
        help="Maximum frames to spend on the start-screen warmup (default: 120).",
    )
    parser.add_argument(
        "--auto-start-press-frames",
        type=int,
        default=6,
        help="Frames to wait after each automatic START press (default: 6).",
    )
    parser.add_argument(
        "--stagnation-frames",
        type=int,
        default=900,
        help="End episodes early if no forward progress for this many frames (0 disables).",
    )
    parser.add_argument(
        "--stagnation-progress",
        type=int,
        default=1,
        help="Minimum forward distance (in mario_x) treated as real progress when tracking stagnation.",
    )
    parser.add_argument(
        "--exploration-epsilon",
        type=float,
        default=0.05,
        help="Probability of forcing a random discrete action (0 disables).",
    )
    parser.add_argument(
        "--exploration-final-epsilon",
        type=float,
        help="Final epsilon after decay (defaults to initial epsilon).",
    )
    parser.add_argument(
        "--exploration-decay-steps",
        type=int,
        default=500_000,
        help="Timesteps over which to decay exploration epsilon (0 keeps it constant).",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        help="Initial entropy regularization coefficient (default mirrors PPO).",
    )
    parser.add_argument(
        "--entropy-final-coef",
        type=float,
        help="Final entropy coefficient after linear decay (defaults to initial).",
    )
    parser.add_argument(
        "--entropy-decay-steps",
        type=int,
        default=0,
        help="Timesteps over which to decay entropy coefficient (0 keeps it constant).",
    )
    parser.add_argument(
        "--icm",
        action="store_true",
        help="Enable intrinsic curiosity module (requires pixel observations).",
    )
    parser.add_argument(
        "--icm-beta",
        type=float,
        default=0.2,
        help="Weighting between inverse and forward curiosity losses (default: 0.2).",
    )
    parser.add_argument(
        "--icm-eta",
        type=float,
        default=0.01,
        help="Scaling factor applied to intrinsic rewards (default: 0.01).",
    )
    parser.add_argument(
        "--icm-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the curiosity module (default: 1e-4).",
    )
    parser.add_argument(
        "--icm-feature-dim",
        type=int,
        default=256,
        help="Latent feature dimension used by the curiosity encoder.",
    )
    parser.add_argument(
        "--icm-hidden-dim",
        type=int,
        default=256,
        help="Hidden layer width for curiosity forward/inverse models.",
    )
    parser.add_argument(
        "--policy-preset",
        choices=sorted(POLICY_PRESETS.keys()),
        default="baseline",
        help="Policy/network configuration preset (mario_large increases GPU load).",
    )
    parser.add_argument("--policy", help="Override policy id (default derived from preset)")
    parser.add_argument("--n-steps", type=int, help="Override rollout length (per env)")
    parser.add_argument("--batch-size", type=int, help="Override mini-batch size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="torch device spec, e.g. cpu or cuda")
    parser.add_argument("--log-dir", default="runs", help="Directory for checkpoints and TensorBoard logs")
    parser.add_argument("--checkpoint-freq", type=int, default=200_000)
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument(
        "--episode-log",
        default="episode_log.jsonl",
        help="Per-episode JSONL filename (relative to log dir). Use 'none' to disable.",
    )
    args = parser.parse_args()

    rom_path = resolve_existing_path(args.rom, "ROM")

    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    action_set = parse_action_set(args.action_set)
    resize_shape = tuple(args.resize) if args.resize else None
    reward_factory = None if args.reward_profile == "none" else REWARD_PRESETS[args.reward_profile]
    auto_start = not args.disable_auto_start
    auto_start_max_frames = max(1, args.auto_start_max_frames)
    auto_start_press_frames = max(1, args.auto_start_press_frames)
    stagnation_max_frames = None if args.stagnation_frames <= 0 else args.stagnation_frames
    stagnation_progress_threshold = max(0, args.stagnation_progress)

    icm_enabled = bool(args.icm)
    if icm_enabled and args.observation_type == "ram":
        raise SystemExit("ICM requires pixel observations (rgb/gray or rgb_ram/gray_ram).")
    icm_kwargs = {
        "beta": float(args.icm_beta),
        "eta": float(args.icm_eta),
        "learning_rate": float(args.icm_lr),
        "feature_dim": int(args.icm_feature_dim),
        "hidden_dim": int(args.icm_hidden_dim),
    }

    exploration_epsilon = max(0.0, args.exploration_epsilon)
    exploration_final_epsilon = args.exploration_final_epsilon
    if exploration_final_epsilon is None:
        exploration_final_epsilon = exploration_epsilon
    exploration_final_epsilon = max(0.0, exploration_final_epsilon)
    exploration_decay_steps = max(0, args.exploration_decay_steps)
    if exploration_decay_steps == 0:
        exploration_epsilon = exploration_final_epsilon

    policy_preset = POLICY_PRESETS[args.policy_preset]
    policy_name = args.policy or policy_preset.policy
    policy_kwargs = dict(policy_preset.policy_kwargs)
    algo_kwargs = dict(policy_preset.algo_kwargs)

    observation_is_multi = args.observation_type in {"rgb_ram", "gray_ram"}
    is_multi_policy = policy_name.startswith("MultiInput")
    if observation_is_multi and not is_multi_policy:
        raise SystemExit(
            "rgb_ram/gray_ram observations require a MultiInput* policy (choose a mario_dual preset or set --policy MultiInputPolicy)."
        )
    if not observation_is_multi and is_multi_policy:
        raise SystemExit(
            "MultiInput policies expect rgb_ram/gray_ram observations. Choose an image-only preset or adjust --observation-type."
        )

    is_lstm_policy = policy_name in {"CnnLstmPolicy", "MultiInputLstmPolicy"}
    if is_lstm_policy and args.algo != "rppo":
        raise SystemExit("LSTM-based presets require --algo rppo (sb3-contrib RecurrentPPO).")

    if args.n_steps:
        algo_kwargs["n_steps"] = args.n_steps
    if args.batch_size:
        algo_kwargs["batch_size"] = args.batch_size

    entropy_initial = args.entropy_coef if args.entropy_coef is not None else algo_kwargs.get("ent_coef")
    if entropy_initial is None:
        entropy_initial = 0.01
    entropy_initial = float(entropy_initial)
    algo_kwargs["ent_coef"] = entropy_initial

    entropy_final = args.entropy_final_coef if args.entropy_final_coef is not None else entropy_initial
    entropy_final = float(entropy_final)

    entropy_decay_steps = max(0, args.entropy_decay_steps)
    if entropy_decay_steps == 0:
        algo_kwargs["ent_coef"] = entropy_final

    episode_log_path: Optional[Path] = None
    if args.episode_log and args.episode_log.lower() != "none":
        ep_path = Path(args.episode_log)
        episode_log_path = ep_path if ep_path.is_absolute() else log_dir / ep_path

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
        auto_start=auto_start,
        auto_start_max_frames=auto_start_max_frames,
        auto_start_press_frames=auto_start_press_frames,
        exploration_epsilon=exploration_epsilon,
        stagnation_max_frames=stagnation_max_frames,
        stagnation_progress_threshold=stagnation_progress_threshold,
        frame_stack=args.frame_stack,
        use_icm=icm_enabled,
        icm_kwargs=icm_kwargs if icm_enabled else None,
    )

    algo_cls = ALGO_MAP[args.algo]
    tensorboard_log = str(log_dir) if args.tensorboard else None

    checkpoint_path = _find_latest_checkpoint(log_dir, args.algo)
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path.name}")
        model = algo_cls.load(str(checkpoint_path), env=vec_env, device=args.device)
        if tensorboard_log and hasattr(model, "tensorboard_log"):
            model.tensorboard_log = tensorboard_log
    else:
        model = algo_cls(
            policy_name,
            vec_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=args.device,
            policy_kwargs=policy_kwargs,
            **algo_kwargs,
        )

    callbacks = []
    if exploration_decay_steps > 0 and abs(exploration_final_epsilon - exploration_epsilon) > 1e-9:
        callbacks.append(
            ExplorationEpsilonCallback(
                initial_epsilon=exploration_epsilon,
                final_epsilon=exploration_final_epsilon,
                decay_steps=exploration_decay_steps,
            )
        )
    if entropy_decay_steps > 0 and abs(entropy_final - entropy_initial) > 1e-9:
        callbacks.append(
            EntropyCoefficientCallback(
                initial_entropy=entropy_initial,
                final_entropy=entropy_final,
                decay_steps=entropy_decay_steps,
            )
        )
    if args.checkpoint_freq > 0:
        save_freq = max(1, args.checkpoint_freq // max(1, args.num_envs))
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(log_dir),
            name_prefix=f"{args.algo}_agent",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)

    if episode_log_path:
        callbacks.append(EpisodeLogCallback(episode_log_path))

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks or None)

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = log_dir / f"{args.algo}_agent_{timestamp}"
    model.save(str(model_path))
    print(f"Saved trained model to {model_path.with_suffix('.zip')}")

    vec_env.close()


if __name__ == "__main__":
    main()