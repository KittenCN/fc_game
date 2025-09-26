"""Custom callbacks for training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


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
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True

        for info in infos:
            if not info:
                continue
            episode = info.get("episode")
            if episode is None:
                continue
            diagnostics = info.get("diagnostics", {})
            metrics = info.get("metrics", {})
            record = {
                "timesteps": int(self.num_timesteps),
                "episode_reward": float(episode.get("r", 0.0)),
                "episode_length": int(episode.get("l", 0)),
                "wall_time": float(episode.get("t", 0.0)),
                "shaped_reward": float(diagnostics.get("shaped_reward", 0.0)),
                "base_reward": float(diagnostics.get("base_reward", 0.0)),
                "metrics": metrics,
                "time_limit_truncated": bool(info.get("TimeLimit.truncated", False)),
                "stagnation_truncated": bool(info.get("stagnation_truncated", False)),
            }
            if "auto_start_presses" in diagnostics:
                record["auto_start_presses"] = int(diagnostics["auto_start_presses"])
            if "stagnation_frames" in diagnostics:
                record["stagnation_frames"] = int(diagnostics["stagnation_frames"])
            self._buffer.append(record)

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


__all__ = ["EpisodeLogCallback", "ExplorationEpsilonCallback", "EntropyCoefficientCallback"]


class ExplorationEpsilonCallback(BaseCallback):
    """Linearly anneal epsilon for EpsilonRandomActionWrapper."""

    def __init__(
        self,
        *,
        initial_epsilon: float,
        final_epsilon: float,
        decay_steps: int,
    ) -> None:
        super().__init__()
        self.initial_epsilon = max(0.0, float(initial_epsilon))
        self.final_epsilon = max(0.0, float(final_epsilon))
        self.decay_steps = max(0, int(decay_steps))
        self._last_value: float | None = None

    def _on_training_start(self) -> None:
        self._apply(self.initial_epsilon)

    def _on_step(self) -> bool:
        if self.decay_steps <= 0:
            target = self.final_epsilon
        else:
            fraction = min(1.0, self.num_timesteps / float(self.decay_steps))
            target = self.initial_epsilon + fraction * (self.final_epsilon - self.initial_epsilon)
        self._apply(target)
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
