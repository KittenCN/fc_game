"""Training utilities for teaching an agent to play NES games."""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    from stable_baselines3.common.callbacks import CheckpointCallback
except ImportError as exc:  # pragma: no cover - user guidance
    raise SystemExit(
        "Stable-Baselines3 is required. Install the RL extras via pip install -e .[rl]."
    ) from exc

from .callbacks import (
    DiagnosticsLoggingCallback,
    EpisodeLogCallback,
    ExplorationEpsilonCallback,
    EntropyCoefficientCallback,
    BestModelCheckpointCallback,
)
from .policies import POLICY_PRESETS
from .rewards import REWARD_PRESETS
from .rl_utils import (
    ALGO_MAP,
    make_vector_env,
    parse_action_set,
    resolve_existing_path,
)


def _serialise_config(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _serialise_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialise_config(item) for item in value]
    return repr(value)


@dataclass
class TrainingConfig:
    rom: Path
    log_dir: Path
    algo: str
    total_timesteps: int
    num_envs: int
    frame_skip: int
    frame_stack: int
    max_episode_steps: Optional[int]
    observation_type: str
    action_set: tuple[tuple[str, ...], ...]
    resize: Optional[tuple[int, int]]
    reward_profile: Optional[str]
    auto_start: bool
    auto_start_max_frames: int
    auto_start_press_frames: int
    stagnation_max_frames: Optional[int]
    stagnation_progress_threshold: int
    stagnation_bonus_scale: float
    stagnation_idle_multiplier: float
    stagnation_backtrack_penalty_scale: float
    stagnation_backtrack_alert_hits: int = 2
    best_checkpoint: Optional[Path] = None
    best_metric_key: str = "mario_x"
    best_metric_mode: str = "mean"
    best_window: int = 20
    best_patience: int = 5
    best_min_improvement: float = 1.0
    eval_interval: int = 0
    eval_episodes: int = 5
    exploration: dict[str, Any] = field(default_factory=dict)
    exploration_env_initial: float = 0.0
    entropy: dict[str, Any] = field(default_factory=dict)
    icm: dict[str, Any] = field(default_factory=dict)
    use_icm: bool = False
    rnd: dict[str, Any] = field(default_factory=dict)
    use_rnd: bool = False
    policy: str = "CnnPolicy"
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    algo_kwargs: dict[str, Any] = field(default_factory=dict)
    device: str = "auto"
    seed: int = 0
    vec_env: str = "auto"
    tensorboard: bool = False
    checkpoint_freq: int = 0
    episode_log_path: Optional[Path] = None
    diagnostics_log_interval: int = 5000
    diagnostics_recent_window: int = 256
    diagnostics_bucket_size: int = 32

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action_set"] = [list(combo) for combo in self.action_set]
        return _serialise_config(data)


def _find_latest_checkpoint(log_dir: Path, algo: str) -> Optional[Path]:
    pattern = f"{algo}_agent*.zip"
    checkpoints = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def _build_training_config(args: argparse.Namespace) -> TrainingConfig:
    rom_path = resolve_existing_path(args.rom, "ROM")
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    action_set = tuple(parse_action_set(args.action_set))
    resize_shape = tuple(int(x) for x in args.resize) if args.resize else None
    reward_profile = None if args.reward_profile == "none" else args.reward_profile

    auto_start = not args.disable_auto_start
    auto_start_max_frames = max(1, args.auto_start_max_frames)
    auto_start_press_frames = max(1, args.auto_start_press_frames)

    stagnation_max_frames: Optional[int] = None
    if args.stagnation_frames is not None and args.stagnation_frames > 0:
        stagnation_max_frames = int(args.stagnation_frames)
    stagnation_progress_threshold = max(0, int(args.stagnation_progress))

    exploration_initial = max(0.0, args.exploration_epsilon)
    exploration_final = (
        args.exploration_final_epsilon
        if args.exploration_final_epsilon is not None
        else exploration_initial
    )
    exploration_final = max(0.0, exploration_final)
    exploration_decay_steps = max(0, args.exploration_decay_steps)
    env_exploration = exploration_initial if exploration_decay_steps > 0 else exploration_final

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
    algo_kwargs.setdefault("normalize_advantage", True)

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

    icm_enabled = bool(args.icm)
    if icm_enabled and args.observation_type == "ram":
        raise SystemExit("ICM requires pixel observations (rgb/gray or rgb_ram/gray_ram).")
    icm_kwargs = {
        "beta": float(args.icm_beta),
        "eta": float(args.icm_eta),
        "learning_rate": float(args.icm_lr),
        "feature_dim": int(args.icm_feature_dim),
        "hidden_dim": int(args.icm_hidden_dim),
        "device": str(args.icm_device),
    }

    rnd_enabled = bool(args.rnd)
    if icm_enabled and rnd_enabled:
        raise SystemExit("ICM and RND cannot be enabled at the same time.")
    rnd_kwargs = {
        "hidden_dim": int(args.rnd_hidden_dim),
        "learning_rate": float(args.rnd_lr),
        "scale": float(args.rnd_scale),
        "normalize": not args.rnd_disable_norm,
        "device": str(args.rnd_device),
        "clip_norm": float(args.rnd_clip_norm),
    }

    episode_log_path: Optional[Path] = None
    if args.episode_log and args.episode_log.lower() != "none":
        ep_path = Path(args.episode_log)
        episode_log_path = ep_path if ep_path.is_absolute() else log_dir / ep_path

    max_episode_steps = args.max_episode_steps if args.max_episode_steps > 0 else None

    best_checkpoint: Optional[Path] = None
    if args.best_checkpoint and args.best_checkpoint.lower() != "none":
        candidate = Path(args.best_checkpoint)
        best_checkpoint = candidate if candidate.is_absolute() else log_dir / candidate

    return TrainingConfig(
        rom=rom_path,
        log_dir=log_dir,
        algo=args.algo,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
        max_episode_steps=max_episode_steps,
        observation_type=args.observation_type,
        action_set=action_set,
        resize=resize_shape,
        reward_profile=reward_profile,
        auto_start=auto_start,
        auto_start_max_frames=auto_start_max_frames,
        auto_start_press_frames=auto_start_press_frames,
        stagnation_max_frames=stagnation_max_frames,
        stagnation_progress_threshold=stagnation_progress_threshold,
        stagnation_bonus_scale=float(args.stagnation_bonus_scale),
        stagnation_idle_multiplier=float(args.stagnation_idle_multiplier),
        stagnation_backtrack_penalty_scale=float(args.stagnation_backtrack_penalty_scale),
        stagnation_backtrack_alert_hits=2,
        best_checkpoint=best_checkpoint,
        best_metric_key=args.best_metric_key,
        best_metric_mode=args.best_metric_mode,
        best_window=int(args.best_window),
        best_patience=int(args.best_patience),
        best_min_improvement=float(args.best_min_improve),
        eval_interval=max(0, int(args.eval_interval)),
        eval_episodes=max(1, int(args.eval_episodes)),
        exploration={
            "initial": exploration_initial,
            "final": exploration_final,
            "decay_steps": exploration_decay_steps,
        },
        exploration_env_initial=env_exploration,
        entropy={
            "initial": entropy_initial,
            "final": entropy_final,
            "decay_steps": entropy_decay_steps,
        },
        icm=icm_kwargs,
        use_icm=icm_enabled,
        rnd=rnd_kwargs,
        use_rnd=rnd_enabled,
        policy=policy_name,
        policy_kwargs=policy_kwargs,
        algo_kwargs=algo_kwargs,
        device=args.device,
        seed=args.seed,
        vec_env=args.vec_env,
        tensorboard=bool(args.tensorboard),
        checkpoint_freq=args.checkpoint_freq,
        episode_log_path=episode_log_path,
        diagnostics_log_interval=max(1, args.diagnostics_log_interval),
        diagnostics_recent_window=max(1, args.diagnostics_recent_window),
        diagnostics_bucket_size=max(1, args.diagnostics_bucket_size),
    )


def _save_config(config: TrainingConfig) -> Path:
    path = config.log_dir / "run_config.json"
    with path.open("w", encoding="utf-8") as fp:
        json.dump(config.to_json_dict(), fp, ensure_ascii=False, indent=2)
    return path


def _evaluate_progress(model, config: TrainingConfig) -> tuple[float, float]:
    if config.eval_interval <= 0:
        return 0.0, 0.0

    eval_env = make_vector_env(
        str(config.rom),
        frame_skip=config.frame_skip,
        observation_type=config.observation_type,
        action_set=config.action_set,
        max_episode_steps=config.max_episode_steps,
        n_envs=1,
        seed=config.seed,
        resize_shape=config.resize,
        vec_env_type="dummy",
        reward_config_factory=None,
        auto_start=config.auto_start,
        auto_start_max_frames=config.auto_start_max_frames,
        auto_start_press_frames=config.auto_start_press_frames,
        exploration_epsilon=0.0,
        stagnation_max_frames=config.stagnation_max_frames,
        stagnation_progress_threshold=config.stagnation_progress_threshold,
        stagnation_bonus_scale=config.stagnation_bonus_scale,
        stagnation_idle_multiplier=config.stagnation_idle_multiplier,
        stagnation_backtrack_penalty_scale=config.stagnation_backtrack_penalty_scale,
        stagnation_backtrack_alert_hits=config.stagnation_backtrack_alert_hits,
        frame_stack=config.frame_stack,
        use_icm=False,
        use_rnd=False,
    )

    mario_x_values: list[float] = []
    episodes_run = 0
    obs = eval_env.reset()
    while episodes_run < config.eval_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        if dones[0]:
            info = infos[0] if infos else {}
            metrics = info.get("metrics") or {}
            mario_x_values.append(float(metrics.get("mario_x", 0.0)))
            episodes_run += 1
            obs = eval_env.reset()

    eval_env.close()

    if mario_x_values:
        mean_val = float(np.mean(mario_x_values))
        max_val = float(np.max(mario_x_values))
    else:
        mean_val = 0.0
        max_val = 0.0
    return mean_val, max_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RL agent on an NES ROM")
    parser.add_argument("--rom", required=True, help="Path to the .nes ROM")
    parser.add_argument("--algo", choices=sorted(ALGO_MAP.keys()), default="ppo")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=3_200)
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
        default=760,
        help="End episodes early if no forward progress for this many frames (0 disables).",
    )
    parser.add_argument(
        "--stagnation-progress",
        type=int,
        default=1,
        help="Minimum forward distance (in mario_x) treated as real progress when tracking stagnation.",
    )
    parser.add_argument(
        "--stagnation-bonus-scale",
        type=float,
        default=0.15,
        help="Additional allowance (ratio of best mario_x) before stagnation triggers.",
    )
    parser.add_argument(
        "--stagnation-idle-multiplier",
        type=float,
        default=1.1,
        help="Multiplier for idle-based termination relative to stagnation limit.",
    )
    parser.add_argument(
        "--stagnation-backtrack-penalty",
        type=float,
        default=1.5,
        help="Penalty factor applied when mario retreats (scaled by frame_skip).",
        dest="stagnation_backtrack_penalty_scale",
    )
    parser.add_argument(
        "--best-checkpoint",
        help="Enable best-model checkpointing (path to save). Use 'none' to disable.",
    )
    parser.add_argument(
        "--best-metric-key",
        default="mario_x",
        help="Metrics key from info.metrics used to evaluate best model (default: mario_x).",
    )
    parser.add_argument(
        "--best-metric-mode",
        choices=["mean", "max"],
        default="mean",
        help="Aggregation mode for best metric (mean or max of recent window).",
    )
    parser.add_argument(
        "--best-window",
        type=int,
        default=30,
        help="Number of episodes per evaluation window when tracking best model.",
    )
    parser.add_argument(
        "--best-patience",
        type=int,
        default=6,
        help="Number of windows without improvement before reloading the best checkpoint.",
    )
    parser.add_argument(
        "--best-min-improve",
        type=float,
        default=1.0,
        help="Minimum average metric improvement required to treat new model as better.",
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
        default=0.02,
        help="Scaling factor applied to intrinsic rewards (default: 0.02).",
    )
    parser.add_argument(
        "--icm-lr",
        type=float,
        default=5e-5,
        help="Learning rate for the curiosity module (default: 5e-5).",
    )
    parser.add_argument(
        "--icm-feature-dim",
        type=int,
        default=128,
        help="Latent feature dimension used by the curiosity encoder.",
    )
    parser.add_argument(
        "--icm-hidden-dim",
        type=int,
        default=128,
        help="Hidden layer width for curiosity forward/inverse models.",
    )
    parser.add_argument(
        "--icm-device",
        default="auto",
        help="Execution device for the curiosity module (auto/cpu/cuda or cuda:idx).",
    )
    parser.add_argument(
        "--rnd",
        action="store_true",
        help="Enable random network distillation intrinsic reward.",
    )
    parser.add_argument(
        "--rnd-hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension for the RND predictor/target networks (default: 512).",
    )
    parser.add_argument(
        "--rnd-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the RND predictor network (default: 1e-4).",
    )
    parser.add_argument(
        "--rnd-scale",
        type=float,
        default=0.5,
        help="Scaling factor applied to normalized RND intrinsic rewards (default: 0.5).",
    )
    parser.add_argument(
        "--rnd-disable-norm",
        action="store_true",
        help="Disable running-normalization for RND intrinsic rewards.",
    )
    parser.add_argument(
        "--rnd-device",
        default="auto",
        help="Execution device for the RND module (auto/cpu/cuda or cuda:idx).",
    )
    parser.add_argument(
        "--rnd-clip-norm",
        type=float,
        default=5.0,
        help="Gradient clipping value applied to the RND predictor (default: 5.0, 0 disables).",
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
    parser.add_argument(
        "--diagnostics-log-interval",
        type=int,
        default=5_000,
        help="Timesteps between diagnostic logging flushes (default: 5000).",
    )
    parser.add_argument(
        "--diagnostics-recent-window",
        type=int,
        default=256,
        help="Rolling window size for recent mario_x mean diagnostics (default: 256).",
    )
    parser.add_argument(
        "--diagnostics-bucket-size",
        type=int,
        default=32,
        help="Bucket size (in mario_x) when aggregating hotspot diagnostics (default: 32).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=200_000,
        help="Timesteps between deterministic evaluation sweeps (0 disables).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of episodes per evaluation sweep when eval-interval>0.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging (disabled by default)",
    )
    parser.add_argument(
        "--episode-log",
        default="episode_log.jsonl",
        help="Per-episode JSONL filename (relative to log dir). Use 'none' to disable.",
    )
    args = parser.parse_args()

    config = _build_training_config(args)

    reward_factory = (
        None if config.reward_profile is None else REWARD_PRESETS[config.reward_profile]
    )

    vec_env = make_vector_env(
        str(config.rom),
        frame_skip=config.frame_skip,
        observation_type=config.observation_type,
        action_set=config.action_set,
        max_episode_steps=config.max_episode_steps,
        n_envs=config.num_envs,
        seed=config.seed,
        resize_shape=config.resize,
        vec_env_type=config.vec_env,
        reward_config_factory=reward_factory,
        auto_start=config.auto_start,
        auto_start_max_frames=config.auto_start_max_frames,
        auto_start_press_frames=config.auto_start_press_frames,
        exploration_epsilon=config.exploration_env_initial,
        stagnation_max_frames=config.stagnation_max_frames,
        stagnation_progress_threshold=config.stagnation_progress_threshold,
        stagnation_bonus_scale=config.stagnation_bonus_scale,
        stagnation_idle_multiplier=config.stagnation_idle_multiplier,
        stagnation_backtrack_penalty_scale=config.stagnation_backtrack_penalty_scale,
        stagnation_backtrack_alert_hits=config.stagnation_backtrack_alert_hits,
        frame_stack=config.frame_stack,
        use_icm=config.use_icm,
        icm_kwargs=config.icm if config.use_icm else None,
        use_rnd=config.use_rnd,
        rnd_kwargs=config.rnd if config.use_rnd else None,
    )

    algo_cls = ALGO_MAP[config.algo]
    tensorboard_log = str(config.log_dir) if config.tensorboard else None

    checkpoint_path = _find_latest_checkpoint(config.log_dir, config.algo)
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path.name}")
        model = algo_cls.load(str(checkpoint_path), env=vec_env, device=config.device)
        if hasattr(model, "tensorboard_log"):
            if tensorboard_log:
                model.tensorboard_log = tensorboard_log
            else:
                model.tensorboard_log = None
    else:
        model = algo_cls(
            config.policy,
            vec_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=config.device,
            policy_kwargs=config.policy_kwargs,
            **config.algo_kwargs,
        )

    if config.use_rnd and hasattr(vec_env, "set_encoder"):
        encoder = getattr(model.policy, "features_extractor", None)
        if encoder is None:
            raise RuntimeError("Policy does not expose a features_extractor required for RND.")
        feature_dim = getattr(encoder, "features_dim", None)
        if feature_dim is None:
            feature_dim = getattr(model.policy, "features_dim", None)
        vec_env.set_encoder(encoder, feature_dim=feature_dim, device=config.device)

    callbacks: list[Any] = []
    exploration_cfg = config.exploration
    if (
        exploration_cfg.get("decay_steps", 0) > 0
        and abs(exploration_cfg["final"] - exploration_cfg["initial"]) > 1e-9
    ):
        callbacks.append(
            ExplorationEpsilonCallback(
                initial_epsilon=float(exploration_cfg["initial"]),
                final_epsilon=float(exploration_cfg["final"]),
                decay_steps=int(exploration_cfg["decay_steps"]),
            )
        )

    entropy_cfg = config.entropy
    if (
        entropy_cfg.get("decay_steps", 0) > 0
        and abs(entropy_cfg["final"] - entropy_cfg["initial"]) > 1e-9
    ):
        callbacks.append(
            EntropyCoefficientCallback(
                initial_entropy=float(entropy_cfg["initial"]),
                final_entropy=float(entropy_cfg["final"]),
                decay_steps=int(entropy_cfg["decay_steps"]),
            )
        )

    if config.checkpoint_freq > 0:
        save_freq = max(1, config.checkpoint_freq // max(1, config.num_envs))
        callbacks.append(
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(config.log_dir),
                name_prefix=f"{config.algo}_agent",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )

    if config.episode_log_path is not None:
        callbacks.append(EpisodeLogCallback(config.episode_log_path))

    callbacks.append(
        DiagnosticsLoggingCallback(
            log_interval=config.diagnostics_log_interval,
            recent_window=config.diagnostics_recent_window,
            hotspot_bucket_size=config.diagnostics_bucket_size,
        )
    )

    best_checkpoint_callback: BestModelCheckpointCallback | None = None
    if config.best_checkpoint is not None:
        best_checkpoint_callback = BestModelCheckpointCallback(
            save_path=config.best_checkpoint,
            metric_key=config.best_metric_key,
            window=config.best_window,
            min_improvement=config.best_min_improvement,
            patience=config.best_patience,
            mode=config.best_metric_mode,
        )
        callbacks.append(best_checkpoint_callback)

    config_path = _save_config(config)
    print(f"Saved run config to {config_path}")

    total_trained = 0
    chunk_size = config.checkpoint_freq if config.checkpoint_freq > 0 else config.total_timesteps
    next_eval = config.eval_interval if config.eval_interval > 0 else None

    while total_trained < config.total_timesteps:
        remaining = config.total_timesteps - total_trained
        current_chunk = min(chunk_size, remaining)
        before_steps = model.num_timesteps
        model.learn(total_timesteps=current_chunk, callback=callbacks, reset_num_timesteps=False)
        after_steps = model.num_timesteps
        actual_chunk = max(0, after_steps - before_steps)
        total_trained += actual_chunk

        if (
            best_checkpoint_callback
            and next_eval is not None
            and total_trained >= next_eval
        ):
            mean_eval, max_eval = _evaluate_progress(model, config)
            metric_to_track = max_eval if config.best_metric_mode == "max" else mean_eval
            print(
                f"Evaluation at {total_trained} steps: mean mario_x={mean_eval:.2f}, max mario_x={max_eval:.2f}"
            )
            best_checkpoint_callback.consider_metric(metric_to_track)
            next_eval += config.eval_interval

        if best_checkpoint_callback and best_checkpoint_callback.should_reload_best:
            best_checkpoint_callback.reset_trigger()
            best_path = config.best_checkpoint
            if best_path and best_path.exists():
                print(f"Reloading best checkpoint from {best_path}")
                old_model = model
                model = None
                gc.collect()
                _clear_cuda_cache(config.device)
                model = algo_cls.load(str(best_path), env=vec_env, device=config.device)
                if hasattr(model, "tensorboard_log"):
                    if tensorboard_log:
                        model.tensorboard_log = tensorboard_log
                    else:
                        model.tensorboard_log = None
                model.num_timesteps = after_steps
                _refresh_callbacks_model(callbacks, model)
                del old_model
                gc.collect()
                _clear_cuda_cache(config.device)
            else:
                print("Best checkpoint flagged for reload but file not found; continuing without reload.")

        if actual_chunk == 0 and not (best_checkpoint_callback and best_checkpoint_callback.should_reload_best):
            print("Stopping early: no training progress detected in the last chunk.")
            break

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = config.log_dir / f"{config.algo}_agent_{timestamp}"
    model.save(str(model_path))
    print(f"Saved trained model to {model_path.with_suffix('.zip')}")

    vec_env.close()


if __name__ == "__main__":
    main()
def _clear_cuda_cache(device_spec: str) -> None:
    """Release cached CUDA memory when reloading checkpoints."""

    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency
        return

    if device_spec.lower() == "cpu":
        return
    if not torch.cuda.is_available():
        return
    if device_spec.lower() != "auto" and not device_spec.lower().startswith("cuda"):
        return

    torch.cuda.empty_cache()


def _refresh_callbacks_model(callbacks: list[Any], model: Any) -> None:
    """Ensure callbacks reference the latest model instance after reloads."""

    for callback in callbacks:
        try:
            callback.model = model
        except AttributeError:
            continue
        if getattr(callback, "training_env", None) is None and getattr(model, "get_env", None):
            callback.training_env = model.get_env()
