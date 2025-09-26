"""Run inference with a trained NES agent."""
from __future__ import annotations

import argparse

try:  # pragma: no cover - optional dependency
    from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
except ImportError as exc:  # pragma: no cover - user guidance
    raise SystemExit(
        "Stable-Baselines3 is required. Install the RL extras via `pip install -e .[rl]`."
    ) from exc

from .rl_utils import (
    ALGO_MAP,
    make_vector_env,
    parse_action_set,
    resolve_existing_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play an NES ROM with a trained agent")
    parser.add_argument("--rom", required=True, help="Path to the .nes ROM")
    parser.add_argument("--model", required=True, help="Path to the saved Stable-Baselines3 model (.zip)")
    parser.add_argument("--algo", choices=sorted(ALGO_MAP.keys()), default="ppo")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=5_000)
    parser.add_argument("--observation-type", choices=["rgb", "gray"], default="gray")
    parser.add_argument("--action-set", help="Preset name (default/simple) or custom combos")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), help="Downscale observations (must match training)")
    parser.add_argument("--disable-auto-start", action="store_true", help="Disable automatically pressing START after each reset.")
    parser.add_argument("--auto-start-max-frames", type=int, default=120, help="Maximum frames to spend on the start-screen warmup (default: 120).")
    parser.add_argument("--auto-start-press-frames", type=int, default=6, help="Frames to wait after each automatic START press (default: 6).")
    parser.add_argument("--stagnation-frames", type=int, default=600, help="End episodes early if no forward progress for this many frames (0 disables).")
    parser.add_argument("--stagnation-progress", type=int, default=5, help="Minimum forward distance treated as progress when tracking stagnation.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy for inference")
    parser.add_argument("--device", default="auto", help="torch device spec, e.g. cpu or cuda")
    args = parser.parse_args()

    rom_path = resolve_existing_path(args.rom, "ROM")
    model_path = resolve_existing_path(args.model, "Model")

    action_set = parse_action_set(args.action_set)
    resize_shape = tuple(args.resize) if args.resize else None
    auto_start = not args.disable_auto_start
    auto_start_max_frames = max(1, args.auto_start_max_frames)
    auto_start_press_frames = max(1, args.auto_start_press_frames)
    stagnation_max_frames = None if args.stagnation_frames <= 0 else args.stagnation_frames
    stagnation_progress_threshold = max(0, args.stagnation_progress)


    vec_env = make_vector_env(
        str(rom_path),
        frame_skip=args.frame_skip,
        observation_type=args.observation_type,
        action_set=action_set,
        max_episode_steps=args.max_episode_steps,
        n_envs=1,
        seed=None,
        render_mode="human",
        resize_shape=resize_shape,
        auto_start=auto_start,
        auto_start_max_frames=auto_start_max_frames,
        auto_start_press_frames=auto_start_press_frames,
        exploration_epsilon=0.0,
        stagnation_max_frames=stagnation_max_frames,
        stagnation_progress_threshold=stagnation_progress_threshold,
    )
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack, channels_order="first")

    algo_cls = ALGO_MAP[args.algo]
    model = algo_cls.load(str(model_path), env=vec_env, device=args.device)

    for episode in range(args.episodes):
        obs = vec_env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, _infos = vec_env.step(action)
            total_reward += float(rewards[0])
            done = bool(dones[0])
            steps += 1
        print(f"Episode {episode + 1}: reward={total_reward:.2f} steps={steps}")

    vec_env.close()


if __name__ == "__main__":
    main()
