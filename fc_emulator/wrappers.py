"""Backward-compatible façade for refactored wrapper utilities."""
from __future__ import annotations

from .actions import (
    DEFAULT_ACTION_SET,
    DiscreteActionWrapper,
    combo_to_multibinary,
    register_action_preset,
    resolve_action_set,
)
from .exploration import EpsilonRandomActionWrapper, MacroSequenceLibrary
from .observation import (
    ResizeObservationWrapper,
    VecFrameStackPixelsDictWrapper,
    VecTransposePixelsDictWrapper,
)

ACTION_PRESETS = {
    "default": DEFAULT_ACTION_SET,
    "simple": (
        (),
        ("A",),
        ("B",),
        ("LEFT",),
        ("RIGHT",),
        ("A", "RIGHT"),
        ("A", "LEFT"),
        ("START",),
    ),
    "smb_forward": (
        ("START",),
        ("RIGHT",),
        ("A", "RIGHT"),
        ("B", "RIGHT"),
        ("A", "B", "RIGHT"),
        ("RIGHT", "DOWN"),
        ("RIGHT", "UP"),
        ("DOWN",),
        ("A",),
        ("B",),
        ("A", "B"),
        ("LEFT",),
        ("A", "LEFT"),
        ("B", "LEFT"),
        ("A", "B", "LEFT"),
        (),
    ),
}


__all__ = [
    "ACTION_PRESETS",
    "DEFAULT_ACTION_SET",
    "DiscreteActionWrapper",
    "EpsilonRandomActionWrapper",
    "MacroSequenceLibrary",
    "ResizeObservationWrapper",
    "VecFrameStackPixelsDictWrapper",
    "VecTransposePixelsDictWrapper",
    "combo_to_multibinary",
    "register_action_preset",
    "resolve_action_set",
]
