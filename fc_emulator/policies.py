"""Custom CNN feature extractors and policy presets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MarioFeatureExtractor(BaseFeaturesExtractor):
    """Deeper CNN backbone for NES observations."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        *,
        channels: Tuple[int, ...] = (32, 64, 128, 128),
        dense: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(observation_space, features_dim=dense)
        n_input_channels = observation_space.shape[0]
        layers = []
        for idx, out_channels in enumerate(channels):
            stride = 2 if idx < len(channels) - 1 else 1
            layers.extend(
                [
                    torch.nn.Conv2d(n_input_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.BatchNorm2d(out_channels),
                ]
            )
            n_input_channels = out_channels
        if dropout > 0:
            layers.append(torch.nn.Dropout2d(dropout))
        self.cnn = torch.nn.Sequential(*layers)

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).view(1, -1).shape[1]
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, dense),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations / 255.0)
        x = torch.flatten(x, 1)
        return self.linear(x)


@dataclass(frozen=True)
class PolicyPreset:
    policy: str
    policy_kwargs: dict
    algo_kwargs: dict


POLICY_PRESETS: Dict[str, PolicyPreset] = {
    "baseline": PolicyPreset(
        policy="CnnPolicy",
        policy_kwargs={},
        algo_kwargs={},
    ),
    "mario_large": PolicyPreset(
        policy="CnnPolicy",
        policy_kwargs={
            "features_extractor_class": MarioFeatureExtractor,
            "features_extractor_kwargs": {
                "channels": (64, 128, 128, 256),
                "dense": 1024,
                "dropout": 0.1,
            },
            "normalize_images": False,
        },
        algo_kwargs={
            "n_steps": 4096,
            "batch_size": 512,
            "learning_rate": 2.5e-4,
        },
    ),
}


__all__ = ["MarioFeatureExtractor", "PolicyPreset", "POLICY_PRESETS"]
