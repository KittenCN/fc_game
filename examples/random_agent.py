"""Run a random agent for smoke testing."""
from __future__ import annotations

import argparse
import time

import numpy as np

from fc_emulator.rl_env import NESGymEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Random policy demo")
    parser.add_argument("--rom", required=True, help="Path to the .nes ROM")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = NESGymEnv(args.rom, render_mode="human", observation_type="rgb")
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            for _ in range(args.max_steps):
                action = env.action_space.sample()
                obs, reward, done, _truncated, info = env.step(action)
                total_reward += reward
                time.sleep(0.01)
                if done:
                    break
            print(f"Episode {episode + 1}: reward={total_reward:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
