"""Custom feature extractors and policy presets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ImpalaResidualBlock(torch.nn.Module):
    """Residual block used in IMPALA-style CNN backbones."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(channels)
        self.norm2 = torch.nn.BatchNorm2d(channels)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return self.activation(x)


class MarioFeatureExtractor(BaseFeaturesExtractor):
    """Deeper CNN backbone for NES observations (channel-first)."""

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
        x = self.cnn(observations.float() / 255.0)
        x = torch.flatten(x, 1)
        return self.linear(x)


class MarioDualFeatureExtractor(BaseFeaturesExtractor):
    """Process dict observations containing image pixels and raw RAM bytes."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        channels: Tuple[int, ...] = (32, 64, 128, 128),
        dense: int = 512,
        ram_hidden: int = 256,
        dropout: float = 0.0,
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("MarioDualFeatureExtractor expects a Dict observation space")
        if "pixels" not in observation_space.spaces or "ram" not in observation_space.spaces:
            raise ValueError("Observation space must contain 'pixels' and 'ram' entries")
        pixel_space = observation_space.spaces["pixels"]
        ram_space = observation_space.spaces["ram"]
        if len(pixel_space.shape) != 3:
            raise ValueError("'pixels' must be channel-first (C, H, W)")
        if len(ram_space.shape) != 1:
            raise ValueError("'ram' must be a flat vector")
        self.pixel_channels = pixel_space.shape[0]
        self.ram_dim = ram_space.shape[0]

        super().__init__(observation_space, features_dim=dense)

        cnn_layers: list[torch.nn.Module] = []
        n_input_channels = self.pixel_channels
        for idx, out_channels in enumerate(channels):
            stride = 2 if idx < len(channels) - 1 else 1
            cnn_layers.extend(
                [
                    torch.nn.Conv2d(n_input_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.BatchNorm2d(out_channels),
                ]
            )
            n_input_channels = out_channels
        if dropout > 0:
            cnn_layers.append(torch.nn.Dropout2d(dropout))
        self.cnn = torch.nn.Sequential(*cnn_layers)

        with torch.no_grad():
            sample_pixels = torch.as_tensor(pixel_space.sample()[None]).float()
            n_flatten = self.cnn(sample_pixels).view(1, -1).shape[1]
        self.ram_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.ram_dim, ram_hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(ram_hidden, ram_hidden),
            torch.nn.ReLU(inplace=True),
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten + ram_hidden, dense),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        pixels = observations["pixels"].float() / 255.0
        ram = observations["ram"].float() / 255.0
        image_feat = torch.flatten(self.cnn(pixels), 1)
        ram_feat = self.ram_mlp(ram)
        combined = torch.cat([image_feat, ram_feat], dim=1)
        return self.linear(combined)


class ImpalaResidualFeatureExtractor(BaseFeaturesExtractor):
    """IMPALA-style residual CNN backbone producing dense features."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        *,
        channels: Tuple[int, ...] = (16, 32, 32),
        blocks_per_stage: int = 2,
        dense: int = 512,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(observation_space, features_dim=dense)
        if len(observation_space.shape) != 3:
            raise ValueError("ImpalaResidualFeatureExtractor expects channel-first image observations")
        in_channels = observation_space.shape[0]

        stages: list[torch.nn.Module] = []
        for out_channels in channels:
            stages.append(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            stages.append(torch.nn.BatchNorm2d(out_channels))
            stages.append(torch.nn.ReLU(inplace=True))
            for _ in range(blocks_per_stage):
                stages.append(ImpalaResidualBlock(out_channels))
            stages.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        if dropout > 0:
            stages.append(torch.nn.Dropout2d(dropout))
        self.cnn = torch.nn.Sequential(*stages)

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float() / 255.0
            conv_out = self.cnn(sample)
            n_flatten = conv_out.view(1, -1).shape[1]

        self.linear = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_flatten, dense),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float() / 255.0
        x = self.cnn(x)
        return self.linear(x)


@dataclass(frozen=True)
class PolicyPreset:
    policy: str
    policy_kwargs: dict
    algo_kwargs: dict


POLICY_PRESETS: Dict[str, PolicyPreset] = {
    "baseline": PolicyPreset(
        policy="CnnPolicy",
        policy_kwargs={"normalize_images": False},
        algo_kwargs={
            "n_steps": 768,
            "batch_size": 192,
        },
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
    "mario_dual": PolicyPreset(
        policy="MultiInputPolicy",
        policy_kwargs={
            "features_extractor_class": MarioDualFeatureExtractor,
            "features_extractor_kwargs": {
                "channels": (32, 64, 128, 128),
                "dense": 768,
                "ram_hidden": 256,
                "dropout": 0.05,
            },
        },
        algo_kwargs={
            "n_steps": 4096,
            "batch_size": 512,
        },
    ),
    "mario_dual_large": PolicyPreset(
        policy="MultiInputPolicy",
        policy_kwargs={
            "features_extractor_class": MarioDualFeatureExtractor,
            "features_extractor_kwargs": {
                "channels": (64, 128, 128, 256),
                "dense": 1024,
                "ram_hidden": 512,
                "dropout": 0.1,
            },
        },
        algo_kwargs={
            "n_steps": 4096,
            "batch_size": 512,
            "learning_rate": 2e-4,
        },
    ),
    "mario_large_lstm": PolicyPreset(
        policy="CnnLstmPolicy",
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
            "n_steps": 1024,
            "batch_size": 256,
            "learning_rate": 2.5e-4,
        },
    ),
    "mario_dual_lstm": PolicyPreset(
        policy="MultiInputLstmPolicy",
        policy_kwargs={
            "features_extractor_class": MarioDualFeatureExtractor,
            "features_extractor_kwargs": {
                "channels": (64, 128, 128, 256),
                "dense": 1024,
                "ram_hidden": 512,
                "dropout": 0.1,
            },
        },
        algo_kwargs={
            "n_steps": 1024,
            "batch_size": 256,
            "learning_rate": 2.5e-4,
        },
    ),
    "impala_lstm": PolicyPreset(
        policy="CnnLstmPolicy",
        policy_kwargs={
            "features_extractor_class": ImpalaResidualFeatureExtractor,
            "features_extractor_kwargs": {
                "channels": (16, 32, 32),
                "blocks_per_stage": 2,
                "dense": 512,
                "dropout": 0.05,
            },
            "normalize_images": False,
        },
        algo_kwargs={
            "n_steps": 512,
            "batch_size": 128,
            "learning_rate": 3e-4,
        },
    ),
}


__all__ = [
    "MarioFeatureExtractor",
    "MarioDualFeatureExtractor",
    "ImpalaResidualFeatureExtractor",
    "PolicyPreset",
    "POLICY_PRESETS",
]
