"""Path-based interface for built and restored retrieval caches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

SPLIT_NAMES = ('train', 'validation', 'test')
SPACE_NAMES = ('structural', 'spatial', 'market')


@dataclass(frozen=True, slots=True)
class InitialNeighborCache:
    """Paths for the unfiltered nearest-neighbour search output."""

    indices_path: Path
    distances_path: Path
    shape: tuple[int, int] | None = None

    def load_indices(self, mmap_mode: str | None = 'r') -> np.ndarray:
        return np.load(self.indices_path, mmap_mode=mmap_mode, allow_pickle=False)

    def load_distances(self, mmap_mode: str | None = 'r') -> np.ndarray:
        return np.load(self.distances_path, mmap_mode=mmap_mode, allow_pickle=False)


@dataclass(frozen=True, slots=True)
class FinalCandidateCache:
    """Paths for filtered, padded candidates consumed by a collator."""

    indices_path: Path
    distances_path: Path
    time_gaps_path: Path
    masks_path: Path
    counts_path: Path
    shape: tuple[int, int] | None = None

    def load_indices(self, mmap_mode: str | None = 'r') -> np.ndarray:
        return np.load(self.indices_path, mmap_mode=mmap_mode, allow_pickle=False)

    def load_distances(self, mmap_mode: str | None = 'r') -> np.ndarray:
        return np.load(self.distances_path, mmap_mode=mmap_mode, allow_pickle=False)

    def load_time_gaps(self, mmap_mode: str | None = 'r') -> np.ndarray:
        return np.load(self.time_gaps_path, mmap_mode=mmap_mode, allow_pickle=False)

    def load_masks(self, mmap_mode: str | None = 'r') -> np.ndarray:
        return np.load(self.masks_path, mmap_mode=mmap_mode, allow_pickle=False)

    def load_counts(self, mmap_mode: str | None = 'r') -> np.ndarray:
        return np.load(self.counts_path, mmap_mode=mmap_mode, allow_pickle=False)


def restore_initial_neighbor_caches(
    cache_dir: str | Path,
    *,
    split_names: tuple[str, ...] = SPLIT_NAMES,
    space_names: tuple[str, ...] = SPACE_NAMES,
) -> Mapping[str, Mapping[str, InitialNeighborCache]]:
    """Restore the same types returned by initial-cache construction."""
    root = Path(cache_dir)
    restored = {
        split_name: MappingProxyType(
            {
                space_name: InitialNeighborCache(
                    indices_path=root / f'{split_name}_{space_name}_indices.npy',
                    distances_path=root / f'{split_name}_{space_name}_distances.npy',
                )
                for space_name in space_names
            }
        )
        for split_name in split_names
    }
    return MappingProxyType(restored)


def restore_final_candidate_caches(
    cache_dir: str | Path,
    *,
    split_names: tuple[str, ...] = SPLIT_NAMES,
    space_names: tuple[str, ...] = SPACE_NAMES,
) -> Mapping[str, Mapping[str, FinalCandidateCache]]:
    """Restore the same types returned by final-cache construction."""
    root = Path(cache_dir)
    restored = {
        split_name: MappingProxyType(
            {
                space_name: FinalCandidateCache(
                    indices_path=root / f'{split_name}_{space_name}_indices.npy',
                    distances_path=root / f'{split_name}_{space_name}_distances.npy',
                    time_gaps_path=root / f'{split_name}_{space_name}_time_gaps.npy',
                    masks_path=root / f'{split_name}_{space_name}_masks.npy',
                    counts_path=root / f'{split_name}_{space_name}_counts.npy',
                )
                for space_name in space_names
            }
        )
        for split_name in split_names
    }
    return MappingProxyType(restored)
