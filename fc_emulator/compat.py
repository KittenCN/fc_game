"""Compatibility helpers keeping third-party deps working together."""
from __future__ import annotations

from typing import Any

_PATCHED = False


def ensure_numpy_rom_compatibility() -> None:
    """Patch nes-py ROM helpers that break with NumPy 2.x scalars."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import numpy as np
        from numpy.lib import NumpyVersion
        from nes_py import _rom
    except Exception:  # pragma: no cover - defensive import guard
        return

    if NumpyVersion(np.__version__) < NumpyVersion("2.0.0"):
        return

    def _as_int(value: Any) -> int:
        if hasattr(value, "item"):
            value = value.item()
        return int(value)

    prg_doc = _rom.ROM.prg_rom_size.__doc__
    chr_doc = _rom.ROM.chr_rom_size.__doc__
    ram_doc = _rom.ROM.prg_ram_size.__doc__

    def _prg_rom_size(self):  # type: ignore[override]
        return 16 * _as_int(self.header[4])

    def _chr_rom_size(self):  # type: ignore[override]
        return 8 * _as_int(self.header[5])

    def _prg_ram_size(self):  # type: ignore[override]
        size = _as_int(self.header[8])
        if size == 0:
            size = 1
        return 8 * size

    _rom.ROM.prg_rom_size = property(_prg_rom_size, doc=prg_doc)
    _rom.ROM.chr_rom_size = property(_chr_rom_size, doc=chr_doc)
    _rom.ROM.prg_ram_size = property(_prg_ram_size, doc=ram_doc)

    _PATCHED = True
