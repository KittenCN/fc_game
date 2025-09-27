"""Utilities for analysing episode logs produced during training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class EpisodeAggregate:
    """Summary statistics extracted from an episode JSONL file."""

    episodes: int
    mean_mario_x: float
    median_mario_x: float
    max_mario_x: int
    mean_episode_length: float
    mean_episode_reward: float
    stagnation_episodes: int
    stagnation_frames_mean: float | None
    hotspots: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, float | int | list[tuple[int, int]] | None]:
        return {
            "episodes": self.episodes,
            "mean_mario_x": self.mean_mario_x,
            "median_mario_x": self.median_mario_x,
            "max_mario_x": self.max_mario_x,
            "mean_episode_length": self.mean_episode_length,
            "mean_episode_reward": self.mean_episode_reward,
            "stagnation_episodes": self.stagnation_episodes,
            "stagnation_frames_mean": self.stagnation_frames_mean,
            "hotspots": list(self.hotspots),
        }


def _iter_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def summarise_episode_log(
    path: Path,
    *,
    bucket_size: int = 32,
    top_n: int = 10,
) -> EpisodeAggregate:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    mario_positions: list[int] = []
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    stagnation_frames: list[int] = []
    stagnation_count = 0
    hotspots = Counter()

    for record in _iter_records(path):
        metrics = record.get("metrics") or {}
        x_pos = metrics.get("mario_x")
        if isinstance(x_pos, (int, float)):
            mario_positions.append(int(x_pos))
            bucket = (int(x_pos) // bucket_size) * bucket_size
            hotspots[bucket] += 1

        episode_lengths.append(int(record.get("episode_length", 0)))
        episode_rewards.append(float(record.get("episode_reward", 0.0)))

        if record.get("stagnation_truncated"):
            stagnation_count += 1
            frames = record.get("stagnation_frames")
            if isinstance(frames, (int, float)):
                stagnation_frames.append(int(frames))

    if not mario_positions:
        raise ValueError(f"No episode records found in {path}")

    sorted_positions = sorted(mario_positions)
    middle = len(sorted_positions) // 2
    if len(sorted_positions) % 2:
        median_x = float(sorted_positions[middle])
    else:
        median_x = (sorted_positions[middle - 1] + sorted_positions[middle]) / 2.0

    frames_mean = mean(stagnation_frames) if stagnation_frames else None
    top_hotspots = tuple(hotspots.most_common(top_n))

    return EpisodeAggregate(
        episodes=len(mario_positions),
        mean_mario_x=mean(mario_positions),
        median_mario_x=median_x,
        max_mario_x=max(mario_positions),
        mean_episode_length=mean(episode_lengths) if episode_lengths else 0.0,
        mean_episode_reward=mean(episode_rewards) if episode_rewards else 0.0,
        stagnation_episodes=stagnation_count,
        stagnation_frames_mean=frames_mean,
        hotspots=top_hotspots,
    )


def _format_hotspot(bucket: int, count: int, bucket_size: int) -> str:
    start = bucket
    end = bucket + bucket_size
    return f"[{start:>4}-{end:>4}): {count}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise an episode JSONL log")
    parser.add_argument("path", type=Path, help="Path to episode_log.jsonl")
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=32,
        help="Bucket width for mario_x hotspot aggregation (default: 32)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of hotspots to display (default: 10)",
    )
    args = parser.parse_args()

    summary = summarise_episode_log(args.path, bucket_size=args.bucket_size, top_n=args.top)
    data = summary.as_dict()

    print(f"Episodes            : {data['episodes']}")
    print(f"Mean mario_x        : {data['mean_mario_x']:.2f}")
    print(f"Median mario_x      : {data['median_mario_x']:.2f}")
    print(f"Max mario_x         : {data['max_mario_x']}")
    print(f"Mean episode length : {data['mean_episode_length']:.2f}")
    print(f"Mean episode reward : {data['mean_episode_reward']:.2f}")
    print(f"Stagnation episodes : {data['stagnation_episodes']}")
    frames_mean = data['stagnation_frames_mean']
    if frames_mean is not None:
        print(f"Mean stagnation frames: {frames_mean:.2f}")
    print("Hotspots (bucket start -> count):")
    for bucket, count in summary.hotspots:
        print("  ", _format_hotspot(bucket, count, args.bucket_size))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
