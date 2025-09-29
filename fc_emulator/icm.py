"""Intrinsic Curiosity Module utilities for augmenting VecEnv rewards."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.vec_env import VecEnvWrapper


def _compute_flatten_size(module: nn.Module, channels: int, height: int, width: int) -> int:
    with torch.no_grad():
        sample = torch.zeros(1, channels, height, width)
        return module(sample).view(1, -1).shape[1]


@dataclass
class ICMConfig:
    feature_dim: int = 128
    hidden_dim: int = 128
    beta: float = 0.2
    eta: float = 0.02
    learning_rate: float = 5e-5
    device: str = "auto"
    pixel_key: str = "pixels"


class _ICMEncoder(nn.Module):
    """Convolutional encoder that maps images to latent features."""

    def __init__(self, input_channels: int, feature_dim: int, height: int, width: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        n_flatten = _compute_flatten_size(self.cnn, input_channels, height, width)
        self.head = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.cnn(obs)
        x = torch.flatten(x, 1)
        return self.head(x)


class ICMNetwork(nn.Module):
    """Full ICM network containing encoder, inverse and forward models."""

    def __init__(self, input_channels: int, action_dim: int, height: int, width: int, config: ICMConfig) -> None:
        super().__init__()
        self.encoder = _ICMEncoder(input_channels, config.feature_dim, height, width)
        self.inverse_model = nn.Sequential(
            nn.Linear(config.feature_dim * 2, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, action_dim),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(config.feature_dim + action_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.feature_dim),
        )
        self.beta = float(config.beta)
        self.eta = float(config.eta)
        self.action_dim = action_dim

    def compute(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        one_hot_actions = F.one_hot(actions, num_classes=self.action_dim).float()
        phi_s = self.encoder(observations)
        phi_next = self.encoder(next_observations)
        pred_next = self.forward_model(torch.cat([phi_s, one_hot_actions], dim=1))
        pred_action_logits = self.inverse_model(torch.cat([phi_s, phi_next], dim=1))
        forward_loss = F.mse_loss(pred_next, phi_next.detach(), reduction="none").mean(dim=1)
        inverse_loss = F.cross_entropy(pred_action_logits, actions, reduction="none")
        intrinsic_reward = self.eta * forward_loss.detach()
        return intrinsic_reward, forward_loss, inverse_loss


class ICMVecEnvWrapper(VecEnvWrapper):
    """Augment environment rewards with intrinsic curiosity signals."""

    def __init__(self, venv, **kwargs: Any):
        config = ICMConfig(**{k: v for k, v in kwargs.items() if k in ICMConfig.__annotations__})
        if config.device == "auto":
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        if config.device not in {"cpu", "cuda"} and not config.device.startswith("cuda"):
            raise ValueError("ICM device must be 'auto', 'cpu' or a cuda string such as 'cuda'/'cuda:0'")
        self.config = config
        observation_space = venv.observation_space
        if hasattr(observation_space, "spaces") and config.pixel_key in getattr(observation_space, "spaces", {}):
            pixel_space = observation_space.spaces[config.pixel_key]
            channels, height, width = pixel_space.shape
            self._pixels_key = config.pixel_key
        else:
            pixel_space = observation_space
            channels, height, width = pixel_space.shape
            self._pixels_key = None
        if channels <= 0:
            raise ValueError("ICM requires image observations")
        super().__init__(venv)
        self.device = torch.device(config.device)
        self.model = ICMNetwork(channels, int(self.action_space.n), height, width, config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self._last_obs: Any = None
        self._last_actions: np.ndarray | None = None

    def reset(self):
        obs = self.venv.reset()
        self._last_obs = obs
        self._last_actions = None
        return obs

    def step_async(self, actions):
        self._last_actions = np.array(actions, copy=True)
        super().step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        if self._last_obs is None or self._last_actions is None:
            self._last_obs = obs
            return obs, rewards, dones, infos

        prev_tensor = self._obs_to_tensor(self._last_obs)
        next_tensor = self._obs_to_tensor(obs)
        actions_tensor = torch.as_tensor(self._last_actions, device=self.device, dtype=torch.long)
        intrinsic, forward_loss, inverse_loss = self.model.compute(prev_tensor, next_tensor, actions_tensor)
        mask = torch.as_tensor(1.0 - dones.astype(np.float32), device=self.device)
        intrinsic = intrinsic * mask
        loss = (1.0 - self.model.beta) * inverse_loss.mean() + self.model.beta * forward_loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        intrinsic_np = intrinsic.detach().cpu().numpy()
        rewards = rewards + intrinsic_np
        for idx, info in enumerate(infos):
            diagnostics = info.setdefault("diagnostics", {})
            diagnostics["intrinsic_reward"] = float(intrinsic_np[idx])
        self._last_obs = obs
        return obs, rewards, dones, infos

    def _obs_to_tensor(self, obs: Any) -> torch.Tensor:
        if self._pixels_key is not None:
            pixels = obs[self._pixels_key]
        else:
            pixels = obs
        if pixels.dtype != np.float32:
            pixels = pixels.astype(np.float32)
        tensor = torch.as_tensor(pixels, device=self.device) / 255.0
        return tensor
