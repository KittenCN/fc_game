"""Utilities for analysing episode logs produced during training."""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
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
    mean_shaped_reward: float
    stagnation_episodes: int
    stagnation_frames_mean: float | None
    hotspots: tuple[tuple[int, int], ...]
    negative_reward_ratio: float
    mean_intrinsic_reward: float | None
    intrinsic_episodes: int
    termination_counts: tuple[tuple[str, int], ...]
    stagnation_reason_counts: tuple[tuple[str, int], ...]
    recent_mario_x_mean: float | None
    recent_window: int

    def as_dict(self) -> dict[str, float | int | list[tuple[int, int]] | None | list[tuple[str, int]]]:
        return {
            "episodes": self.episodes,
            "mean_mario_x": self.mean_mario_x,
            "median_mario_x": self.median_mario_x,
            "max_mario_x": self.max_mario_x,
            "mean_episode_length": self.mean_episode_length,
            "mean_episode_reward": self.mean_episode_reward,
            "mean_shaped_reward": self.mean_shaped_reward,
            "stagnation_episodes": self.stagnation_episodes,
            "stagnation_frames_mean": self.stagnation_frames_mean,
            "hotspots": list(self.hotspots),
            "negative_reward_ratio": self.negative_reward_ratio,
            "mean_intrinsic_reward": self.mean_intrinsic_reward,
            "termination_counts": list(self.termination_counts),
            "stagnation_reason_counts": list(self.stagnation_reason_counts),
            "intrinsic_episodes": self.intrinsic_episodes,
            "recent_mario_x_mean": self.recent_mario_x_mean,
            "recent_window": self.recent_window,
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
    hotspots = Counter()
    termination_counter = Counter()
    stagnation_reason_counter = Counter()

    recent_window = 200
    recent_positions: deque[int] = deque(maxlen=recent_window)

    total_mario_x = 0.0
    max_mario_x = 0
    total_episode_length = 0.0
    total_episode_reward = 0.0
    total_shaped_reward = 0.0
    stagnation_frames_sum = 0.0
    stagnation_count = 0
    intrinsic_sum = 0.0
    intrinsic_count = 0
    negative_count = 0
    episodes = 0
    for record in _iter_records(path):
        episodes += 1
        metrics = record.get("metrics") or {}
        x_pos = metrics.get("mario_x")
        if isinstance(x_pos, (int, float)):
            mario_positions.append(int(x_pos))
            bucket = (int(x_pos) // bucket_size) * bucket_size
            hotspots[bucket] += 1
            total_mario_x += float(x_pos)
            max_mario_x = max(max_mario_x, int(x_pos))
            recent_positions.append(int(x_pos))

        length_value = int(record.get("episode_length", 0))
        total_episode_length += length_value

        reward_value = float(record.get("episode_reward", 0.0))
        total_episode_reward += reward_value
        if reward_value < 0.0:
            negative_count += 1

        shaped_reward = record.get("shaped_reward")
        if isinstance(shaped_reward, (int, float)):
            total_shaped_reward += float(shaped_reward)

        if record.get("stagnation_truncated"):
            stagnation_count += 1
            frames = record.get("stagnation_frames")
            if isinstance(frames, (int, float)):
                stagnation_frames_sum += float(frames)

        intrinsic = record.get("intrinsic_reward")
        if isinstance(intrinsic, (int, float)):
            intrinsic_sum += float(intrinsic)
            intrinsic_count += 1

        termination = record.get("termination_reason")
        if not isinstance(termination, str) or not termination:
            if record.get("stagnation_truncated"):
                termination = "stagnation"
            elif record.get("time_limit_truncated"):
                termination = "time_limit"
            else:
                termination = "unknown"
        termination_counter[termination] += 1

        reason = metrics.get("stagnation_reason")
        if isinstance(reason, str) and reason:
            stagnation_reason_counter[reason] += 1

    if episodes == 0 or not mario_positions:
        raise ValueError(f"No episode records found in {path}")

    sorted_positions = sorted(mario_positions)
    middle = len(sorted_positions) // 2
    if len(sorted_positions) % 2:
        median_x = float(sorted_positions[middle])
    else:
        median_x = (sorted_positions[middle - 1] + sorted_positions[middle]) / 2.0

    frames_mean = (
        stagnation_frames_sum / stagnation_count if stagnation_count > 0 else None
    )
    top_hotspots = tuple(hotspots.most_common(top_n))

    negative_ratio = negative_count / float(episodes)
    intrinsic_mean = intrinsic_sum / intrinsic_count if intrinsic_count > 0 else None
    termination_stats = tuple(termination_counter.most_common())
    stagnation_reason_stats = tuple(stagnation_reason_counter.most_common())

    recent_mean = None
    if recent_positions:
        recent_mean = sum(recent_positions) / len(recent_positions)

    mean_mario_x = total_mario_x / episodes if episodes else 0.0
    mean_length = total_episode_length / episodes if episodes else 0.0
    mean_reward = total_episode_reward / episodes if episodes else 0.0
    mean_shaped = total_shaped_reward / episodes if episodes else 0.0

    return EpisodeAggregate(
        episodes=episodes,
        mean_mario_x=mean_mario_x,
        median_mario_x=median_x,
        max_mario_x=max_mario_x,
        mean_episode_length=mean_length,
        mean_episode_reward=mean_reward,
        mean_shaped_reward=mean_shaped,
        stagnation_episodes=stagnation_count,
        stagnation_frames_mean=frames_mean,
        hotspots=top_hotspots,
        negative_reward_ratio=negative_ratio,
        mean_intrinsic_reward=intrinsic_mean,
        intrinsic_episodes=intrinsic_count,
        termination_counts=termination_stats,
        stagnation_reason_counts=stagnation_reason_stats,
        recent_mario_x_mean=recent_mean,
        recent_window=recent_window,
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the aggregate summary as JSON instead of human-readable text",
    )
    args = parser.parse_args()

    summary = summarise_episode_log(args.path, bucket_size=args.bucket_size, top_n=args.top)
    data = summary.as_dict()

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    print(f"Episodes            : {data['episodes']}")
    print(f"Mean mario_x        : {data['mean_mario_x']:.2f}")
    print(f"Median mario_x      : {data['median_mario_x']:.2f}")
    print(f"Max mario_x         : {data['max_mario_x']}")
    print(f"Mean episode length : {data['mean_episode_length']:.2f}")
    print(f"Mean episode reward : {data['mean_episode_reward']:.2f}")
    print(f"Mean shaped reward  : {data['mean_shaped_reward']:.2f}")
    print(f"Negative reward ratio: {data['negative_reward_ratio'] * 100:.1f}%")
    intrinsic_mean = data['mean_intrinsic_reward']
    if intrinsic_mean is not None:
        print(f"Mean intrinsic reward : {intrinsic_mean:.2f}")
        print(f"Episodes with intrinsic: {data['intrinsic_episodes']}")
    print(f"Stagnation episodes : {data['stagnation_episodes']}")
    frames_mean = data['stagnation_frames_mean']
    if frames_mean is not None:
        print(f"Mean stagnation frames: {frames_mean:.2f}")
    if data['recent_mario_x_mean'] is not None:
        print(
            f"Recent mario_x mean ({data['recent_window']} eps): {data['recent_mario_x_mean']:.2f}"
        )
    print("Terminations by reason:")
    for reason, count in summary.termination_counts:
        print(f"  {reason:>10}: {count}")
    if summary.stagnation_reason_counts:
        print("Stagnation reasons:")
        for reason, count in summary.stagnation_reason_counts:
            print(f"  {reason:>10}: {count}")
    print("Hotspots (bucket start -> count):")
    for bucket, count in summary.hotspots:
        print("  ", _format_hotspot(bucket, count, args.bucket_size))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
