"""Random Network Distillation wrapper for VecEnv."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.vec_env import VecEnvWrapper


def _as_torch_device(device: str | torch.device | None) -> torch.device:
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class _RunningMeanStd:
    """Track running standard deviation for reward normalization."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = torch.tensor(0.0)
        self.var = torch.tensor(1.0)
        self.count = torch.tensor(epsilon)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        batch_mean = values.mean()
        batch_var = values.var(unbiased=False)
        batch_count = torch.tensor(float(values.numel()), device=values.device)

        delta = batch_mean - self.mean.to(values.device)
        total_count = self.count.to(values.device) + batch_count

        new_mean = self.mean.to(values.device) + delta * batch_count / total_count
        m_a = self.var.to(values.device) * self.count.to(values.device)
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count.to(values.device) * batch_count / total_count

        self.mean = new_mean.detach().cpu()
        self.var = (m2 / total_count).detach().cpu()
        self.count = total_count.detach().cpu()

    @property
    def std(self) -> float:
        return float(torch.sqrt(self.var + 1e-8))


@dataclass
class RNDConfig:
    hidden_dim: int = 512
    learning_rate: float = 1e-4
    scale: float = 1.0
    normalize: bool = True
    device: str | torch.device = "auto"
    clip_norm: float = 5.0


class RNDVecEnvWrapper(VecEnvWrapper):
    """Augment environment rewards with Random Network Distillation."""

    def __init__(self, venv, **kwargs: Any) -> None:
        super().__init__(venv)
        config = RNDConfig(**{k: v for k, v in kwargs.items() if k in RNDConfig.__annotations__})
        self.config = config
        self.device = _as_torch_device(config.device)
        self.encoder: nn.Module | None = None
        self.encoder_device: torch.device | None = None
        self.feature_dim: int | None = None
        self.predictor: nn.Module | None = None
        self.target: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.running_stats = _RunningMeanStd()

    def set_encoder(
        self,
        encoder: nn.Module,
        *,
        feature_dim: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Attach a feature encoder shared with the policy."""

        self.encoder = encoder
        self.encoder_device = next((p.device for p in encoder.parameters() if p.requires_grad), None)
        if feature_dim is None:
            feature_dim = int(getattr(encoder, "features_dim", 0))
        if not feature_dim or feature_dim <= 0:
            raise ValueError("Unable to determine feature dimension for RND.")
        self.feature_dim = feature_dim
        self.device = _as_torch_device(device or self.encoder_device or self.config.device)
        self._build_networks(feature_dim)

    def _build_networks(self, feature_dim: int) -> None:
        target = nn.Sequential(
            nn.Linear(feature_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.hidden_dim, feature_dim),
        )
        predictor = nn.Sequential(
            nn.Linear(feature_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.hidden_dim, feature_dim),
        )
        self.target = target.to(self.device)
        for param in self.target.parameters():
            param.requires_grad_(False)
        self.predictor = predictor.to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.config.learning_rate)

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        if self.encoder is None or self.predictor is None or self.target is None:
            return obs, rewards, dones, infos

        obs_tensor = self._obs_to_tensor(obs)
        features = self._encode(obs_tensor)

        with torch.enable_grad():
            pred = self.predictor(features)
            target = self.target(features.detach())
            losses = F.mse_loss(pred, target.detach(), reduction="none").mean(dim=1)
            loss = losses.mean()
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.config.clip_norm)
            self.optimizer.step()

        intrinsic = losses.detach()

        if self.config.normalize:
            self.running_stats.update(intrinsic.cpu())
            std = max(self.running_stats.std, 1e-6)
            intrinsic = intrinsic / std

        intrinsic = intrinsic * self.config.scale
        intrinsic_np = intrinsic.cpu().numpy()
        intrinsic_np = intrinsic_np * (1.0 - dones.astype(np.float32))

        rewards = rewards + intrinsic_np
        for idx, info in enumerate(infos):
            diagnostics = info.setdefault("diagnostics", {})
            diagnostics["intrinsic_reward"] = float(intrinsic_np[idx])

        return obs, rewards, dones, infos

    def _obs_to_tensor(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, dict):
            pixels = obs.get("pixels")
            if pixels is None:
                raise ValueError("RND wrapper requires 'pixels' entry for dict observations")
            array = pixels
        else:
            array = obs
        tensor = torch.as_tensor(array, device=self.encoder_device or self.device, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor

    def _encode(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("RND encoder has not been attached.")
        was_training = getattr(self.encoder, "training", False)
        try:
            if was_training:
                self.encoder.eval()
            with torch.no_grad():
                features = self.encoder(obs_tensor).detach()
        finally:
            if was_training:
                self.encoder.train()
        if self.device != features.device:
            features = features.to(self.device)
        return features


__all__ = ["RNDVecEnvWrapper", "RNDConfig"]
