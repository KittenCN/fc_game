"""High-level public API for the FC emulator toolkit.

This module intentionally avoids importing heavy optional dependencies (such as nes-py or numpy) at import time so that lightweight utilities like the
log analysis helpers remain usable in environments where those packages are not
installed.  The public objects are exposed lazily via the __getattr__ hook.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "EpisodeAggregate",
    "NESEmulator",
    "NESGymEnv",
    "summarise_episode_log",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "EpisodeAggregate": ("fc_emulator.analysis", "EpisodeAggregate"),
    "summarise_episode_log": ("fc_emulator.analysis", "summarise_episode_log"),
    "NESEmulator": ("fc_emulator.emulator", "NESEmulator"),
    "NESGymEnv": ("fc_emulator.rl_env", "NESGymEnv"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - simple helper
    return sorted(set(__all__ + list(globals().keys())))


if TYPE_CHECKING:  # pragma: no cover - type checkers need eager defs
    from .analysis import EpisodeAggregate, summarise_episode_log
    from .emulator import NESEmulator
    from .rl_env import NESGymEnv
