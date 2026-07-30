"""Worker-safe lazy collation of path-based candidate caches"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from hdb_compnet.config import HDBCompNetConfig
from hdb_compnet.data.cache import FinalCandidateCache
from hdb_compnet.data.datasets import ModelSplitData, TransactionIndexDataset
from hdb_compnet.data.reference_memory import ReferenceMemory


@dataclass(frozen=True, slots=True)
class CandidateCacheArrays:
    """Memory-mapped arrays opened lazily inside a DataLoader worker"""

    indices: np.ndarray
    distances: np.ndarray
    time_gaps: np.ndarray
    masks: np.ndarray
    counts: np.ndarray


class HDBCompNetCollator:
    """Materialize candidates for one split using from space to cache mapping"""

    def __init__(
        self,
        split_data: ModelSplitData,
        reference_memory: ReferenceMemory,
        candidate_caches: Mapping[str, FinalCandidateCache],
    ):
        self.split_data = split_data
        self.reference_numeric_features = reference_memory.numeric_features
        self.reference_categorical_features = reference_memory.categorical_features
        self.reference_targets = reference_memory.targets
        self.candidate_caches = dict(candidate_caches)
        self._candidate_cache_arrays: dict[str, CandidateCacheArrays] | None = None

    def _load_candidate_cache_arrays(self) -> dict[str, CandidateCacheArrays]:
        if self._candidate_cache_arrays is None:
            self._candidate_cache_arrays = {
                space_name: CandidateCacheArrays(
                    indices=cache.load_indices(),
                    distances=cache.load_distances(),
                    time_gaps=cache.load_time_gaps(),
                    masks=cache.load_masks(),
                    counts=cache.load_counts(),
                )
                for space_name, cache in self.candidate_caches.items()
            }
        return self._candidate_cache_arrays

    def __call__(self, batch_indices: Sequence[int]) -> dict[str, object]:
        cache_arrays = self._load_candidate_cache_arrays()
        batch_positions = np.asarray(batch_indices, dtype=np.int64)
        batch: dict[str, object] = {
            'numeric': torch.from_numpy(
                np.ascontiguousarray(self.split_data.numeric_features[batch_positions])
            ),
            'categorical': torch.from_numpy(
                np.ascontiguousarray(self.split_data.categorical_features[batch_positions])
            ),
            'target': torch.from_numpy(
                np.ascontiguousarray(self.split_data.targets[batch_positions])
            ),
            'candidates': {},
        }
        candidate_batches: dict[str, dict[str, torch.Tensor]] = {}
        for space_name, cache in cache_arrays.items():
            candidate_indices = np.asarray(cache.indices[batch_positions], dtype=np.int64)
            candidate_masks = np.asarray(cache.masks[batch_positions], dtype=np.bool_)
            safe_indices = np.where(candidate_masks, candidate_indices, 0)
            candidate_numeric = (
                self.reference_numeric_features[safe_indices] * candidate_masks[..., None]
            )
            candidate_categorical = np.where(
                candidate_masks[..., None],
                self.reference_categorical_features[safe_indices],
                0,
            )
            candidate_targets = np.where(
                candidate_masks, self.reference_targets[safe_indices], 0
            )
            candidate_distances = np.where(
                candidate_masks, cache.distances[batch_positions], 0
            ).astype(np.float32, copy=False)
            candidate_time_gaps = np.where(
                candidate_masks, cache.time_gaps[batch_positions], 0
            ).astype(np.float32, copy=False)
            candidate_counts = np.asarray(cache.counts[batch_positions], dtype=np.float32)
            candidate_batches[space_name] = {
                'numeric': torch.from_numpy(
                    np.ascontiguousarray(candidate_numeric, dtype=np.float32)
                ),
                'categorical': torch.from_numpy(
                    np.ascontiguousarray(candidate_categorical, dtype=np.int64)
                ),
                'targets': torch.from_numpy(
                    np.ascontiguousarray(candidate_targets, dtype=np.float32)
                ),
                'distances': torch.from_numpy(np.ascontiguousarray(candidate_distances)),
                'time_gaps': torch.from_numpy(np.ascontiguousarray(candidate_time_gaps)),
                'masks': torch.from_numpy(np.ascontiguousarray(candidate_masks)),
                'counts': torch.from_numpy(np.ascontiguousarray(candidate_counts)),
            }
        batch['candidates'] = candidate_batches
        return batch


def create_hdb_compnet_dataloaders(
    config: HDBCompNetConfig,
    *,
    train_split_data: ModelSplitData,
    validation_split_data: ModelSplitData,
    test_split_data: ModelSplitData,
    reference_memory: ReferenceMemory,
    final_candidate_caches: Mapping[str, Mapping[str, FinalCandidateCache]],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    common: dict[str, object] = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'pin_memory': config.pin_memory,
        'drop_last': False,
        'persistent_workers': config.persistent_workers and config.num_workers > 0,
    }
    if config.num_workers > 0:
        common['prefetch_factor'] = 2
    split_data = {
        'train': train_split_data,
        'validation': validation_split_data,
        'test': test_split_data,
    }
    collators = {
        split_name: HDBCompNetCollator(
            data,
            reference_memory,
            final_candidate_caches[split_name],
        )
        for split_name, data in split_data.items()
    }
    return (
        DataLoader(
            TransactionIndexDataset(len(train_split_data)),
            shuffle=True,
            collate_fn=collators['train'],
            generator=generator,
            **common,
        ),
        DataLoader(
            TransactionIndexDataset(len(validation_split_data)),
            shuffle=False,
            collate_fn=collators['validation'],
            **common,
        ),
        DataLoader(
            TransactionIndexDataset(len(test_split_data)),
            shuffle=False,
            collate_fn=collators['test'],
            **common,
        ),
    )
