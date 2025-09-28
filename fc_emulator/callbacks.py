"""Custom callbacks for training."""
from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class BestModelCheckpointCallback(BaseCallback):
    """Persist the best-performing model and signal reloads when progress stalls."""

    def __init__(
        self,
        *,
        save_path: Path,
        metric_key: str = "mario_x",
        window: int = 20,
        min_improvement: float = 1.0,
        patience: int = 5,
        mode: str = "mean",
    ) -> None:
        super().__init__()
        self.save_path = Path(save_path)
        self.metric_path = self.save_path.with_suffix(".json")
        self.metric_key = metric_key
        self.window = max(1, int(window))
        self.min_improvement = float(min_improvement)
        self.patience = max(1, int(patience))
        mode_normalized = mode.lower()
        if mode_normalized not in {"mean", "max"}:
            raise ValueError("mode must be 'mean' or 'max'")
        self.mode = mode_normalized
        self._recent_metrics: deque[float] = deque(maxlen=self.window)
        self._best_metric: float | None = None
        self._no_improve_windows = 0
        self.should_reload_best = False
        if self.metric_path.exists():
            try:
                data = json.loads(self.metric_path.read_text())
                best_val = float(data.get("best_metric"))
                stored_mode = data.get("mode")
            except Exception:
                best_val = None
                stored_mode = None
            if best_val is not None and (stored_mode in {None, self.mode}):
                self._best_metric = best_val

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            episode = info.get("episode")
            if not episode:
                continue
            metrics = info.get("metrics") or {}
            value = metrics.get(self.metric_key)
            if value is None:
                continue
            try:
                metric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not self._consider_metric(metric_value):
                return False
        return True

    def consider_metric(self, value: float) -> None:
        self._consider_metric(value, force_window=True)

    def _consider_metric(self, value: float, force_window: bool = False) -> bool:
        if force_window:
            self._recent_metrics.clear()
            for _ in range(self.window):
                self._recent_metrics.append(value)
        else:
            self._recent_metrics.append(value)

        if len(self._recent_metrics) < self.window:
            return True

        if self.mode == "mean":
            current_metric = sum(self._recent_metrics) / len(self._recent_metrics)
        else:
            current_metric = max(self._recent_metrics)

        if self._best_metric is None or current_metric > self._best_metric + self.min_improvement:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.save_path))
            try:
                payload = {"best_metric": current_metric, "mode": self.mode}
                self.metric_path.write_text(json.dumps(payload, indent=2))
            except Exception:
                pass
            self._best_metric = current_metric
            self._no_improve_windows = 0
        else:
            self._no_improve_windows += 1

        if self._no_improve_windows >= self.patience:
            self.should_reload_best = True
            self._no_improve_windows = 0
            self._recent_metrics.clear()
            return False

        return True

    def reset_trigger(self) -> None:
        self.should_reload_best = False


class EpisodeLogCallback(BaseCallback):
    """Write per-episode statistics to a JSONL file for offline analysis."""

    def __init__(
        self,
        log_path: Path,
        *,
        flush_every: int = 50,
    ) -> None:
        super().__init__()
        self.log_path = log_path
        self.flush_every = max(1, flush_every)
        self._buffer: list[dict[str, Any]] = []
        self._running_totals = defaultdict(
            lambda: {
                "base": 0.0,
                "shaped": 0.0,
                "auto_start_presses": None,
                "stagnation_frames": None,
                "stagnation_limit": None,
                "stagnation_idle_frames": None,
                "intrinsic": 0.0,
            }
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _coerce_float(value: Any) -> float:
        if isinstance(value, (list, tuple)):
            if not value:
                return 0.0
            value = value[0]
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True

        rewards = self.locals.get("rewards")

        for idx, raw_info in enumerate(infos):
            info = raw_info or {}
            totals = self._running_totals[idx]

            if rewards is not None:
                try:
                    step_reward = rewards[idx]
                except Exception:
                    step_reward = rewards
                totals["shaped"] += self._coerce_float(step_reward)

            diagnostics = info.get("diagnostics") or {}
            base_value = diagnostics.get("base_reward")
            if base_value is not None:
                totals["base"] += self._coerce_float(base_value)

            intrinsic_value = diagnostics.get("intrinsic_reward")
            if intrinsic_value is None:
                intrinsic_value = info.get("intrinsic_reward")
            if intrinsic_value is not None:
                totals["intrinsic"] += self._coerce_float(intrinsic_value)

            if "auto_start_presses" in diagnostics and totals["auto_start_presses"] is None:
                try:
                    totals["auto_start_presses"] = int(diagnostics["auto_start_presses"])
                except (TypeError, ValueError):
                    totals["auto_start_presses"] = None

            for candidate in (diagnostics.get("stagnation_frames"), info.get("stagnation_frames")):
                if candidate is None:
                    continue
                try:
                    value_int = int(candidate)
                except (TypeError, ValueError):
                    continue
                previous = totals["stagnation_frames"]
                totals["stagnation_frames"] = value_int if previous is None else max(previous, value_int)

            limit_value = diagnostics.get("stagnation_limit")
            if limit_value is None:
                limit_value = info.get("stagnation_limit")
            if limit_value is not None:
                try:
                    totals["stagnation_limit"] = float(limit_value)
                except (TypeError, ValueError):
                    pass

            idle_candidate = diagnostics.get("stagnation_idle_frames")
            if idle_candidate is None:
                idle_candidate = info.get("stagnation_idle_frames")
            if idle_candidate is not None:
                try:
                    idle_int = int(idle_candidate)
                except (TypeError, ValueError):
                    idle_int = None
                if idle_int is not None:
                    previous_idle = totals["stagnation_idle_frames"]
                    totals["stagnation_idle_frames"] = (
                        idle_int if previous_idle is None else max(previous_idle, idle_int)
                    )

            episode = info.get("episode")
            if episode is None:
                continue

            termination_reason = "terminated"
            if info.get("stagnation_truncated"):
                termination_reason = "stagnation"
            elif info.get("TimeLimit.truncated", False):
                termination_reason = "time_limit"

            metrics = info.get("metrics", {})
            record = {
                "timesteps": int(self.num_timesteps),
                "episode_reward": float(episode.get("r", 0.0)),
                "episode_length": int(episode.get("l", 0)),
                "wall_time": float(episode.get("t", 0.0)),
                "shaped_reward": float(totals["shaped"]),
                "base_reward": float(totals["base"]),
                "metrics": metrics,
                "time_limit_truncated": bool(info.get("TimeLimit.truncated", False)),
                "stagnation_truncated": bool(info.get("stagnation_truncated", False)),
                "termination_reason": termination_reason,
            }

            if totals["auto_start_presses"] is not None:
                record["auto_start_presses"] = int(totals["auto_start_presses"])
            if totals["stagnation_frames"] is not None:
                record["stagnation_frames"] = int(totals["stagnation_frames"])
            if totals["stagnation_idle_frames"] is not None:
                record["stagnation_idle_frames"] = int(totals["stagnation_idle_frames"])
            if totals["stagnation_limit"] is not None:
                record["stagnation_limit"] = float(totals["stagnation_limit"])

            if abs(totals["intrinsic"]) > 1e-9:
                record["intrinsic_reward"] = float(totals["intrinsic"])

            self._buffer.append(record)
            self._running_totals.pop(idx, None)

        if len(self._buffer) >= self.flush_every:
            self._flush()
        return True

    def _on_training_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        with self.log_path.open("a", encoding="utf-8") as fp:
            for entry in self._buffer:
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._buffer.clear()


__all__ = [
    "DiagnosticsLoggingCallback",
    "EpisodeLogCallback",
    "ExplorationEpsilonCallback",
    "EntropyCoefficientCallback",
]


class ExplorationEpsilonCallback(BaseCallback):
    """Linearly anneal epsilon for EpsilonRandomActionWrapper with hotspot boosts."""

    def __init__(
        self,
        *,
        initial_epsilon: float,
        final_epsilon: float,
        decay_steps: int,
        boost_epsilon: float = 0.05,
        boost_threshold: int = 3,
        boost_duration: int = 100_000,
    ) -> None:
        super().__init__()
        self.initial_epsilon = max(0.0, float(initial_epsilon))
        self.final_epsilon = max(0.0, float(final_epsilon))
        self.decay_steps = max(0, int(decay_steps))
        self.boost_epsilon = max(self.final_epsilon, float(boost_epsilon))
        self.boost_threshold = max(1, int(boost_threshold))
        self.boost_duration = max(0, int(boost_duration))
        self._last_value: float | None = None
        self._boost_steps_remaining = 0
        self._repeat_bucket: int | None = None
        self._repeat_count = 0
        self._boost_events = {"stagnation", "backtrack", "no_progress", "backtrack_warning", "score_loop"}
        self._boost_reasons = {"stagnation", "backtrack", "no_progress", "score_loop"}

    def _on_training_start(self) -> None:
        self._boost_steps_remaining = 0
        self._repeat_bucket = None
        self._repeat_count = 0
        self._apply(self.initial_epsilon)

    def _on_step(self) -> bool:
        if self.decay_steps <= 0:
            target = self.final_epsilon
        else:
            fraction = min(1.0, self.num_timesteps / float(self.decay_steps))
            target = self.initial_epsilon + fraction * (self.final_epsilon - self.initial_epsilon)

        infos = self.locals.get("infos") or []
        for info in infos:
            if not info.get("episode"):
                continue
            metrics = info.get("metrics") or {}
            bucket = metrics.get("stagnation_bucket")
            event = metrics.get("stagnation_event")
            reason = metrics.get("stagnation_reason")
            truncated = bool(info.get("stagnation_truncated"))
            if (
                isinstance(bucket, int)
                and truncated
                and (
                    (isinstance(event, str) and event in self._boost_events)
                    or (isinstance(reason, str) and reason in self._boost_reasons)
                )
            ):
                if bucket == self._repeat_bucket:
                    self._repeat_count += 1
                else:
                    self._repeat_bucket = bucket
                    self._repeat_count = 1
                if self._repeat_count >= self.boost_threshold:
                    self._boost_steps_remaining = self.boost_duration
                    self._repeat_bucket = None
                    self._repeat_count = 0
                    break
            else:
                self._repeat_bucket = None
                self._repeat_count = 0

        effective = target
        if self._boost_steps_remaining > 0 and self.boost_duration > 0:
            ratio = min(1.0, max(0.0, self._boost_steps_remaining / float(self.boost_duration)))
            boost_level = self.final_epsilon + ratio * (self.boost_epsilon - self.final_epsilon)
            effective = max(effective, boost_level)
            self._boost_steps_remaining = max(0, self._boost_steps_remaining - 1)
        elif self._boost_steps_remaining > 0:
            effective = max(effective, self.boost_epsilon)
            self._boost_steps_remaining = 0

        self._apply(effective)
        return True

    def _apply(self, value: float) -> None:
        if self.training_env is None:
            return
        if self._last_value is not None and abs(self._last_value - value) < 1e-6:
            return
        try:
            self.training_env.env_method("set_exploration_epsilon", float(value))
        except Exception:
            pass
        else:
            self._last_value = float(value)




class EntropyCoefficientCallback(BaseCallback):
    """Linearly anneal the policy entropy coefficient (A3C-style exploration)."""

    def __init__(
        self,
        *,
        initial_entropy: float,
        final_entropy: float,
        decay_steps: int,
    ) -> None:
        super().__init__()
        self.initial_entropy = float(initial_entropy)
        self.final_entropy = float(final_entropy)
        self.decay_steps = max(0, int(decay_steps))
        self._last_value: float | None = None

    def _on_training_start(self) -> None:
        self._apply(self.initial_entropy)

    def _on_step(self) -> bool:
        if self.decay_steps <= 0:
            target = self.final_entropy
        else:
            fraction = min(1.0, self.num_timesteps / float(self.decay_steps))
            target = self.initial_entropy + fraction * (self.final_entropy - self.initial_entropy)
        self._apply(target)
        return True

    def _apply(self, value: float) -> None:
        if not hasattr(self.model, "ent_coef"):
            return
        if self._last_value is not None and abs(self._last_value - value) < 1e-6:
            return
        self.model.ent_coef = float(value)
        self._last_value = float(value)


class DiagnosticsLoggingCallback(BaseCallback):
    """Stream environment diagnostics (progress, hotspots, stagnation) to SB3 logger."""

    def __init__(
        self,
        *,
        log_interval: int = 5000,
        recent_window: int = 256,
        hotspot_bucket_size: int = 32,
    ) -> None:
        super().__init__()
        self.log_interval = max(1, int(log_interval))
        self.recent_window = max(1, int(recent_window))
        self.hotspot_bucket_size = max(1, int(hotspot_bucket_size))
        self._last_logged = 0
        self._recent_positions: deque[float] = deque(maxlen=self.recent_window)
        self._positions_since_log: list[float] = []
        self._rewards_since_log: list[float] = []
        self._reason_counts: Counter[str] = Counter()
        self._hotspot_counts: Counter[int] = Counter()
        self._stagnation_hotspots: Counter[int] = Counter()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            metrics = info.get("metrics") or {}
            x_pos = metrics.get("mario_x")
            if isinstance(x_pos, (int, float)):
                position = float(x_pos)
                self._positions_since_log.append(position)
                self._recent_positions.append(position)
                bucket = int(x_pos // self.hotspot_bucket_size) * self.hotspot_bucket_size
                self._hotspot_counts[bucket] += 1
                if info.get("stagnation_truncated"):
                    self._stagnation_hotspots[bucket] += 1
            intrinsic = info.get("intrinsic_reward")
            if intrinsic is None:
                intrinsic = metrics.get("intrinsic_reward")
            if isinstance(intrinsic, (int, float)):
                self._rewards_since_log.append(float(intrinsic))
            reason = metrics.get("stagnation_reason")
            if isinstance(reason, str) and reason:
                self._reason_counts[reason] += 1

        if self.num_timesteps - self._last_logged < self.log_interval:
            return True

        logger = getattr(self.model, "logger", None)
        if logger is not None:
            if self._positions_since_log:
                mean_x = sum(self._positions_since_log) / len(self._positions_since_log)
                logger.record("diagnostics/mario_x_mean", mean_x)
                logger.record("diagnostics/mario_x_max", max(self._positions_since_log))
            if self._recent_positions:
                recent_mean = sum(self._recent_positions) / len(self._recent_positions)
                logger.record("diagnostics/mario_x_recent_mean", recent_mean)
            if self._rewards_since_log:
                mean_intrinsic = sum(self._rewards_since_log) / len(self._rewards_since_log)
                logger.record("diagnostics/intrinsic_mean", mean_intrinsic)
            total_reasons = sum(self._reason_counts.values())
            if total_reasons:
                for reason, count in self._reason_counts.items():
                    ratio = count / float(total_reasons)
                    logger.record(f"diagnostics/stagnation_{reason}", ratio)
            if self._hotspot_counts:
                top_hotspots = self._hotspot_counts.most_common(3)
                for idx, (bucket, count) in enumerate(top_hotspots, start=1):
                    logger.record(f"diagnostics/hotspot_{idx}_bucket", float(bucket))
                    logger.record(f"diagnostics/hotspot_{idx}_ratio", count / max(1.0, len(self._positions_since_log)))
            if self._stagnation_hotspots:
                total_stagnations = sum(self._stagnation_hotspots.values())
                for idx, (bucket, count) in enumerate(self._stagnation_hotspots.most_common(3), start=1):
                    logger.record(f"diagnostics/stagnation_hotspot_{idx}", float(bucket))
                    logger.record(
                        f"diagnostics/stagnation_hotspot_{idx}_ratio",
                        count / float(total_stagnations),
                    )
            logger.record("diagnostics/last_log_step", float(self.num_timesteps))
            logger.dump(self.num_timesteps)

        self._positions_since_log.clear()
        self._rewards_since_log.clear()
        self._reason_counts.clear()
        self._hotspot_counts.clear()
        self._stagnation_hotspots.clear()
        self._last_logged = self.num_timesteps
        return True
