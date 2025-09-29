"""Observation utilities for NES Gym environments."""
from __future__ import annotations

from collections.abc import MutableMapping
from typing import Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:  # pragma: no cover - optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:  # pragma: no cover - optional dependency
    from stable_baselines3.common.vec_env import VecEnvWrapper
except ImportError:  # pragma: no cover - optional dependency
    VecEnvWrapper = None  # type: ignore


def _resize_frame(frame: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    height, width = shape
    if cv2 is not None:  # pragma: no cover - requires optional dependency
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    elif Image is not None:  # pragma: no cover - requires optional dependency
        if frame.ndim == 2 or frame.shape[-1] == 1:
            img = Image.fromarray(frame.squeeze(), mode="L")
            resized = np.asarray(img.resize((width, height), Image.BILINEAR), dtype=frame.dtype)
            resized = resized[..., None]
        else:
            img = Image.fromarray(frame)
            resized = np.asarray(img.resize((width, height), Image.BILINEAR), dtype=frame.dtype)
    else:  # pragma: no cover - defensive path
        raise RuntimeError("Resizing observations requires installing either opencv-python or pillow.")

    if resized.dtype != frame.dtype:
        resized = resized.astype(frame.dtype)
    if frame.ndim == 3 and resized.ndim == 2:
        resized = resized[..., None]
    return resized


class ResizeObservationWrapper(gym.ObservationWrapper):
    """Resize image observations without global gym wrappers."""

    def __init__(self, env: gym.Env, shape: Tuple[int, int]):
        super().__init__(env)
        if len(shape) != 2:
            raise ValueError("shape must be (height, width)")
        self.shape = shape
        orig_space = env.observation_space
        self._dict_mode = isinstance(orig_space, spaces.Dict)
        if self._dict_mode:
            if "pixels" not in orig_space.spaces:
                raise ValueError("ResizeObservationWrapper requires a 'pixels' key in Dict observations")
            pixel_space = orig_space.spaces["pixels"]
            channels = pixel_space.shape[2] if len(pixel_space.shape) == 3 else 1
            self.channels = channels
            self._pixel_dtype = pixel_space.dtype
            spaces_dict = dict(orig_space.spaces)
            spaces_dict["pixels"] = spaces.Box(
                low=0,
                high=255,
                shape=(shape[0], shape[1], channels),
                dtype=pixel_space.dtype,
            )
            self.observation_space = spaces.Dict(spaces_dict)
        else:
            channels = orig_space.shape[2] if len(orig_space.shape) == 3 else 1
            self.channels = channels
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(shape[0], shape[1], channels),
                dtype=orig_space.dtype,
            )
            self._pixel_dtype = orig_space.dtype

    def observation(self, observation: np.ndarray | MutableMapping[str, np.ndarray]):
        if self._dict_mode:
            obs_dict = dict(observation)
            pixels = obs_dict.get("pixels")
            if pixels is None:
                return obs_dict
            resized = _resize_frame(pixels, self.shape)
            if self.channels == 1 and resized.ndim == 2:
                resized = resized[..., None]
            if resized.dtype != self._pixel_dtype:
                resized = resized.astype(self._pixel_dtype, copy=False)
            obs_dict["pixels"] = resized
            return obs_dict
        resized = _resize_frame(observation, self.shape)
        if self.channels == 1 and resized.ndim == 2:
            resized = resized[..., None]
        if resized.dtype != self._pixel_dtype:
            resized = resized.astype(self._pixel_dtype, copy=False)
        return resized


class VecTransposePixelsDictWrapper(VecEnvWrapper):
    """Transpose dict observations so the 'pixels' entry is channel-first."""

    def __init__(self, venv, *, key: str = "pixels"):
        if VecEnvWrapper is None:  # pragma: no cover - optional dependency
            raise ImportError("VecTransposePixelsDictWrapper requires stable-baselines3")
        observation_space = venv.observation_space
        if not isinstance(observation_space, spaces.Dict) or key not in observation_space.spaces:
            raise ValueError("VecTransposePixelsDictWrapper expects a Dict observation with a 'pixels' key")
        pixel_space = observation_space.spaces[key]
        if len(pixel_space.shape) != 3:
            raise ValueError("'pixels' entry must be an image with shape (H, W, C)")
        height, width, channels = pixel_space.shape
        transposed_space = spaces.Box(
            low=pixel_space.low.min() if isinstance(pixel_space.low, np.ndarray) else 0,
            high=pixel_space.high.max() if isinstance(pixel_space.high, np.ndarray) else 255,
            shape=(channels, height, width),
            dtype=pixel_space.dtype,
        )
        new_spaces = dict(observation_space.spaces)
        new_spaces[key] = transposed_space
        super().__init__(venv, observation_space=spaces.Dict(new_spaces))
        self.key = key

    def _transpose_batch(self, pixels: np.ndarray) -> np.ndarray:
        return np.transpose(pixels, (0, 3, 1, 2))

    @staticmethod
    def _transpose_single(pixels: np.ndarray) -> np.ndarray:
        if pixels.ndim == 3:
            return np.transpose(pixels, (2, 0, 1))
        if pixels.ndim == 4:
            return np.transpose(pixels, (0, 3, 1, 2))
        raise ValueError("Unexpected pixels terminal observation shape")

    def reset(self):
        obs = self.venv.reset()
        obs_dict = dict(obs)
        obs_dict[self.key] = self._transpose_batch(obs_dict[self.key])
        return obs_dict

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs_dict = dict(obs)
        obs_dict[self.key] = self._transpose_batch(obs_dict[self.key])
        for info in infos:
            terminal = info.get("terminal_observation")
            if terminal is None:
                continue
            if isinstance(terminal, dict):
                term_copy = dict(terminal)
                if self.key in term_copy:
                    term_copy[self.key] = self._transpose_single(term_copy[self.key])
                info["terminal_observation"] = term_copy
            else:
                info["terminal_observation"] = self._transpose_single(terminal)
        return obs_dict, rewards, dones, infos


class VecFrameStackPixelsDictWrapper(VecEnvWrapper):
    """Frame-stack only the 'pixels' entry inside dict observations."""

    def __init__(self, venv, *, key: str = "pixels", n_stack: int = 4):
        if VecEnvWrapper is None:  # pragma: no cover - optional dependency
            raise ImportError("VecFrameStackPixelsDictWrapper requires stable-baselines3")
        if n_stack <= 0:
            raise ValueError("n_stack must be >= 1")
        observation_space = venv.observation_space
        if not isinstance(observation_space, spaces.Dict) or key not in observation_space.spaces:
            raise ValueError("VecFrameStackPixelsDictWrapper expects a Dict observation with a 'pixels' key")
        pixel_space = observation_space.spaces[key]
        if len(pixel_space.shape) != 3:
            raise ValueError("'pixels' entry must be channel-first (C, H, W)")
        channels, height, width = pixel_space.shape
        stacked_space = spaces.Box(
            low=pixel_space.low.min() if isinstance(pixel_space.low, np.ndarray) else pixel_space.low,
            high=pixel_space.high.max() if isinstance(pixel_space.high, np.ndarray) else pixel_space.high,
            shape=(channels * n_stack, height, width),
            dtype=pixel_space.dtype,
        )
        new_spaces = dict(observation_space.spaces)
        new_spaces[key] = stacked_space
        super().__init__(venv, observation_space=spaces.Dict(new_spaces))
        self.key = key
        self.n_stack = int(n_stack)
        self.channels = channels
        self._dtype = pixel_space.dtype
        self._height = height
        self._width = width
        self.stacked_obs = np.zeros((self.num_envs, channels * n_stack, height, width), dtype=self._dtype)
        self._prev_stacked = np.empty_like(self.stacked_obs)
        self._output_buffer = np.empty_like(self.stacked_obs)

    def _append(self, pixels: np.ndarray) -> None:
        self.stacked_obs[:, :-self.channels] = self.stacked_obs[:, self.channels:]
        self.stacked_obs[:, -self.channels:] = pixels

    def reset(self):
        obs = self.venv.reset()
        obs_dict = dict(obs)
        self.stacked_obs.fill(0)
        self._append(obs_dict[self.key])
        np.copyto(self._output_buffer, self.stacked_obs)
        obs_dict[self.key] = self._output_buffer
        return obs_dict

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs_dict = dict(obs)
        pixels = obs_dict[self.key]
        np.copyto(self._prev_stacked, self.stacked_obs)
        self._append(pixels)
        for idx, done in enumerate(dones):
            if not done:
                continue
            term = infos[idx].get("terminal_observation")
            if isinstance(term, dict):
                term_copy = dict(term)
            else:
                term_copy = {}
            term_copy[self.key] = self._prev_stacked[idx].copy()
            infos[idx]["terminal_observation"] = term_copy
            self.stacked_obs[idx].fill(0)
            self.stacked_obs[idx, -self.channels:] = pixels[idx]
        np.copyto(self._output_buffer, self.stacked_obs)
        obs_dict[self.key] = self._output_buffer
        return obs_dict, rewards, dones, infos


__all__ = [
    "ResizeObservationWrapper",
    "VecFrameStackPixelsDictWrapper",
    "VecTransposePixelsDictWrapper",
]
