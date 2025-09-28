"""监控训练日志中的热点分布与停滞信息。"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
from typing import Iterable

from .analysis import EpisodeAggregate, summarise_episode_log


def _format_hotspots(
    hotspots: Iterable[tuple[int, int]], *, bucket_size: int, total_episodes: int
) -> str:
    lines: list[str] = []
    for rank, (bucket, count) in enumerate(hotspots, start=1):
        start = bucket
        end = bucket + bucket_size
        ratio = count / total_episodes if total_episodes else 0.0
        lines.append(f"#{rank}: 区间[{start}, {end}) 次数 {count} (占比 {ratio:.2%})")
    return "\n".join(lines) if lines else "无热点数据"


def _format_terminations(termination_counts: Iterable[tuple[str, int]], *, total: int) -> str:
    entries: list[str] = []
    for reason, count in termination_counts:
        ratio = count / total if total else 0.0
        entries.append(f"{reason}: {count} ({ratio:.2%})")
    return "，".join(entries) if entries else "无终止记录"


def _render_summary(summary: EpisodeAggregate, *, path: Path, bucket_size: int, delta_episodes: int) -> str:
    stagnation_ratio = summary.stagnation_episodes / summary.episodes if summary.episodes else 0.0
    intrinsic_ratio = summary.intrinsic_episodes / summary.episodes if summary.episodes else 0.0
    recent_mean = "—" if summary.recent_mario_x_mean is None else f"{summary.recent_mario_x_mean:.1f}"
    intrinsic_mean = "—" if summary.mean_intrinsic_reward is None else f"{summary.mean_intrinsic_reward:.4f}"
    lines = [
        f"[{dt.datetime.now().strftime('%H:%M:%S')}] {path.name} 共 {summary.episodes} 条记录 (新增 {delta_episodes})",
        f"平均 mario_x {summary.mean_mario_x:.1f}，中位数 {summary.median_mario_x:.1f}，最大值 {summary.max_mario_x}",
        f"近期窗口均值 {recent_mean} (窗口 {summary.recent_window})",
        f"停滞终止 {summary.stagnation_episodes} 次，占比 {stagnation_ratio:.2%}，负回报占比 {summary.negative_reward_ratio:.2%}",
        f"内在奖励回合 {summary.intrinsic_episodes} 次，占比 {intrinsic_ratio:.2%}，平均值 {intrinsic_mean}",
        "热点 Top:",
        _format_hotspots(summary.hotspots, bucket_size=bucket_size, total_episodes=summary.episodes),
        f"终止原因统计：{_format_terminations(summary.termination_counts, total=summary.episodes)}",
        f"停滞原因统计：{_format_terminations(summary.stagnation_reason_counts, total=summary.episodes)}",
    ]
    return "\n".join(lines)


def _monitor(path: Path, *, bucket_size: int, top_n: int, poll_interval: float) -> None:
    last_episodes = 0
    while True:
        try:
            summary = summarise_episode_log(path, bucket_size=bucket_size, top_n=top_n)
        except FileNotFoundError:
            print(f"找不到日志文件：{path}")
            time.sleep(max(1.0, poll_interval))
            continue
        except ValueError as exc:
            print(f"读取日志失败：{exc}")
            time.sleep(max(1.0, poll_interval))
            continue

        delta = summary.episodes - last_episodes
        print(_render_summary(summary, path=path, bucket_size=bucket_size, delta_episodes=max(delta, 0)))
        print("-" * 72)
        last_episodes = summary.episodes
        time.sleep(max(1.0, poll_interval))


def _oneshot(path: Path, *, bucket_size: int, top_n: int) -> None:
    summary = summarise_episode_log(path, bucket_size=bucket_size, top_n=top_n)
    print(_render_summary(summary, path=path, bucket_size=bucket_size, delta_episodes=summary.episodes))


def main() -> None:
    parser = argparse.ArgumentParser(description="监控 NES 训练热点分布与停滞统计")
    parser.add_argument("--log", default="runs/episode_log.jsonl", help="Episode JSONL 日志路径")
    parser.add_argument("--bucket-size", type=int, default=32, help="热点统计桶宽度")
    parser.add_argument("--top", type=int, default=10, help="展示的热点数量")
    parser.add_argument("--poll-interval", type=float, default=15.0, help="轮询间隔秒数")
    parser.add_argument("--oneshot", action="store_true", help="仅输出一次统计后退出")
    args = parser.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    if args.oneshot:
        _oneshot(log_path, bucket_size=args.bucket_size, top_n=args.top)
        return
    print(f"开始监控 {log_path}，每 {args.poll_interval:.0f} 秒刷新一次……")
    _monitor(log_path, bucket_size=args.bucket_size, top_n=args.top, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
