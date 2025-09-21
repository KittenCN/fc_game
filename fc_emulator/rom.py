"""ROM loading utilities for NES iNES files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

INES_MAGIC = b"NES\x1a"
HEADER_SIZE = 16
TRAINER_SIZE = 512


@dataclass(slots=True)
class ROMMetadata:
    """Structured view of the iNES header."""

    prg_rom_banks: int
    chr_rom_banks: int
    mapper: int
    mirroring: str
    battery_backed_ram: bool
    trainer_present: bool
    four_screen_vram: bool
    vs_unisystem: bool
    playchoice10: bool

    @property
    def prg_rom_size(self) -> int:
        return self.prg_rom_banks * 16 * 1024

    @property
    def chr_rom_size(self) -> int:
        return self.chr_rom_banks * 8 * 1024


class ROMLoadError(RuntimeError):
    """Raised when ROM parsing fails."""


class ROM:
    """In-memory representation of an NES ROM."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise ROMLoadError(f"ROM file not found: {self.path}")

        with self.path.open("rb") as fp:
            self.metadata = _parse_header(fp)
            self.prg_rom, self.chr_rom, self.trainer = _read_banks(fp, self.metadata)

    def __repr__(self) -> str:  # pragma: no cover - simple debug helper
        return (
            f"ROM(path={self.path!s}, prg={len(self.prg_rom)} bytes, "
            f"chr={len(self.chr_rom)} bytes, mapper={self.metadata.mapper})"
        )


def _parse_header(fp: BinaryIO) -> ROMMetadata:
    header = fp.read(HEADER_SIZE)
    if len(header) != HEADER_SIZE or header[:4] != INES_MAGIC:
        raise ROMLoadError("Invalid iNES header")

    prg_rom_banks = header[4]
    chr_rom_banks = header[5]

    flags6 = header[6]
    flags7 = header[7]

    mapper_low = flags6 >> 4
    mapper_high = flags7 & 0xF0
    mapper = mapper_low | mapper_high

    mirroring = "vertical" if (flags6 & 0x01) else "horizontal"
    battery_backed_ram = bool(flags6 & 0x02)
    trainer_present = bool(flags6 & 0x04)
    four_screen_vram = bool(flags6 & 0x08)
    vs_unisystem = bool(flags7 & 0x01)
    playchoice10 = bool(flags7 & 0x02)

    return ROMMetadata(
        prg_rom_banks=prg_rom_banks,
        chr_rom_banks=chr_rom_banks,
        mapper=mapper,
        mirroring=mirroring,
        battery_backed_ram=battery_backed_ram,
        trainer_present=trainer_present,
        four_screen_vram=four_screen_vram,
        vs_unisystem=vs_unisystem,
        playchoice10=playchoice10,
    )


def _read_banks(fp: BinaryIO, metadata: ROMMetadata) -> tuple[bytes, bytes, bytes | None]:
    trainer = None
    if metadata.trainer_present:
        trainer = fp.read(TRAINER_SIZE)
        if len(trainer) != TRAINER_SIZE:
            raise ROMLoadError("Incomplete trainer data")

    prg_size = metadata.prg_rom_size
    chr_size = metadata.chr_rom_size

    prg_rom = fp.read(prg_size)
    if len(prg_rom) != prg_size:
        raise ROMLoadError("Incomplete PRG ROM")

    chr_rom = fp.read(chr_size)
    if chr_size and len(chr_rom) != chr_size:
        raise ROMLoadError("Incomplete CHR ROM")

    return prg_rom, chr_rom, trainer
