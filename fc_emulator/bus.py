"""Lightweight helpers for interacting with the NES memory map."""
from __future__ import annotations

import numpy as np


class MemoryBus:
    """Provides convenient read helpers on top of nes-py RAM dumps."""

    def __init__(self, ram_provider):
        """ram_provider must expose a get_ram() -> np.ndarray method."""
        self._provider = ram_provider

    def read_byte(self, address: int) -> int:
        ram = self._provider.get_ram()
        return int(ram[address % len(ram)])

    def read_range(self, start: int, length: int) -> np.ndarray:
        ram = self._provider.get_ram()
        end = start + length
        return np.array([ram[i % len(ram)] for i in range(start, end)], dtype=np.uint8)

    def snapshot(self) -> np.ndarray:
        return self._provider.get_ram().copy()
