"""Serialization helpers shared by configuration and checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def convert_to_serializable(value: Any) -> Any:
    """Recursively convert mappings, paths, and NumPy scalar values."""
    if isinstance(value, Mapping):
        return {str(key): convert_to_serializable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(convert_to_serializable(item) for item in value)
    if isinstance(value, list):
        return [convert_to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def compact_epoch_statistics(
    statistics: Mapping[str, object],
) -> dict[str, object]:
    """Remove large prediction and target vectors before checkpointing."""
    return {
        key: value for key, value in statistics.items() if key not in ('predictions', 'targets')
    }
