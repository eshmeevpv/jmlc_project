"""Grouped nearest-neighbour search and filtering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from hdb_compnet.data.cache import FinalCandidateCache, InitialNeighborCache
from hdb_compnet.data.reference_memory import ReferenceMemory


@dataclass(frozen=True, slots=True)
class NeighborGroupIndex:
    """Nearest-neighbour estimator and its positions in ReferenceMemory."""

    model: NearestNeighbors
    memory_positions: np.ndarray

    def __len__(self) -> int:
        return self.memory_positions.shape[0]


def build_retrieval_indices(
    reference_memory: ReferenceMemory,
    *,
    k_search: int = 128,
) -> Mapping[str, Mapping[int, NeighborGroupIndex]]:
    """Build separate brute-force Euclidean indices for each flat type and space."""
    retrieval_indices: dict[str, Mapping[int, NeighborGroupIndex]] = {}
    flat_type_indices = np.unique(reference_memory.flat_type_indices)
    flat_type_indices = flat_type_indices[flat_type_indices != 0]
    for space_name, retrieval_features in reference_memory.retrieval_features.items():
        group_indices: dict[int, NeighborGroupIndex] = {}
        for flat_type_index in flat_type_indices:
            memory_positions = np.flatnonzero(
                reference_memory.flat_type_indices == flat_type_index
            ).astype(np.int64)
            model = NearestNeighbors(
                n_neighbors=k_search,
                algorithm='brute',
                metric='euclidean',
                n_jobs=-1,
            )
            model.fit(retrieval_features[memory_positions])
            group_indices[int(flat_type_index)] = NeighborGroupIndex(
                model=model, memory_positions=memory_positions
            )
        retrieval_indices[space_name] = MappingProxyType(group_indices)
    return MappingProxyType(retrieval_indices)


def build_initial_neighbor_caches(
    retrieval_indices: Mapping[str, Mapping[int, NeighborGroupIndex]],
    retrieval_features_by_split: Mapping[str, Mapping[str, np.ndarray | csr_matrix]],
    category_indices_by_split: Mapping[str, pd.DataFrame],
    cache_dir: str | Path,
    *,
    k_search: int = 128,
    batch_size: int = 1024,
) -> Mapping[str, Mapping[str, InitialNeighborCache]]:
    """Search before temporal filtering."""
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Mapping[str, InitialNeighborCache]] = {}
    for split_name, retrieval_features in retrieval_features_by_split.items():
        split_result: dict[str, InitialNeighborCache] = {}
        flat_type_indices = category_indices_by_split[split_name]['flat_type'].to_numpy(
            dtype=np.int64, copy=False
        )
        for space_name, space_indices in retrieval_indices.items():
            indices_path = root / f'{split_name}_{space_name}_indices.npy'
            distances_path = root / f'{split_name}_{space_name}_distances.npy'
            candidate_indices = open_memmap(
                indices_path,
                mode='w+',
                dtype=np.int32,
                shape=(len(flat_type_indices), k_search),
            )
            candidate_distances = open_memmap(
                distances_path,
                mode='w+',
                dtype=np.float32,
                shape=(len(flat_type_indices), k_search),
            )
            candidate_indices[:] = -1
            candidate_distances[:] = np.inf
            for flat_type_index, group_index in space_indices.items():
                query_positions = np.flatnonzero(flat_type_indices == flat_type_index)
                if query_positions.size == 0:
                    continue
                n_neighbors = min(k_search, len(group_index))
                for start in range(0, query_positions.size, batch_size):
                    batch_positions = query_positions[start : start + batch_size]
                    distances, local_indices = group_index.model.kneighbors(
                        retrieval_features[space_name][batch_positions],
                        n_neighbors=n_neighbors,
                        return_distance=True,
                    )
                    global_indices = group_index.memory_positions[local_indices]
                    candidate_indices[batch_positions, :n_neighbors] = global_indices.astype(
                        np.int32, copy=False
                    )
                    candidate_distances[batch_positions, :n_neighbors] = distances.astype(
                        np.float32, copy=False
                    )
            candidate_indices.flush()
            candidate_distances.flush()
            del candidate_indices, candidate_distances
            split_result[space_name] = InitialNeighborCache(
                indices_path=indices_path,
                distances_path=distances_path,
                shape=(len(flat_type_indices), k_search),
            )
        result[split_name] = MappingProxyType(split_result)
    return MappingProxyType(result)


def build_final_candidate_caches(
    initial_neighbor_caches: Mapping[str, Mapping[str, InitialNeighborCache]],
    reference_memory: ReferenceMemory,
    reference_by_split: Mapping[str, pd.DataFrame],
    cache_dir: str | Path,
    *,
    k_candidates: int = 32,
    batch_size: int = 8192,
) -> Mapping[str, Mapping[str, FinalCandidateCache]]:
    """Apply self, strict-history, and 15/20/25% area-backoff filtering."""
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Mapping[str, FinalCandidateCache]] = {}
    for split_name, split_reference in reference_by_split.items():
        split_result: dict[str, FinalCandidateCache] = {}
        query_resale_ids = split_reference['resale_id'].to_numpy(dtype=np.int64, copy=False)
        query_month_indices = split_reference['month_index'].to_numpy(
            dtype=np.int32, copy=False
        )
        query_floor_areas = split_reference['floor_area_sqm'].to_numpy(
            dtype=np.float32, copy=False
        )
        for space_name, initial_cache in initial_neighbor_caches[split_name].items():
            indices_path = root / f'{split_name}_{space_name}_indices.npy'
            distances_path = root / f'{split_name}_{space_name}_distances.npy'
            time_gaps_path = root / f'{split_name}_{space_name}_time_gaps.npy'
            masks_path = root / f'{split_name}_{space_name}_masks.npy'
            counts_path = root / f'{split_name}_{space_name}_counts.npy'
            shape = (len(split_reference), k_candidates)
            final_indices = open_memmap(indices_path, mode='w+', dtype=np.int32, shape=shape)
            final_distances = open_memmap(
                distances_path, mode='w+', dtype=np.float32, shape=shape
            )
            final_time_gaps = open_memmap(
                time_gaps_path, mode='w+', dtype=np.int16, shape=shape
            )
            final_masks = open_memmap(masks_path, mode='w+', dtype=np.bool_, shape=shape)
            final_counts = open_memmap(
                counts_path,
                mode='w+',
                dtype=np.uint8,
                shape=(len(split_reference),),
            )
            final_indices[:] = -1
            final_distances[:] = 0
            final_time_gaps[:] = 0
            final_masks[:] = False
            final_counts[:] = 0
            initial_indices = initial_cache.load_indices()
            initial_distances = initial_cache.load_distances()
            for start in range(0, len(split_reference), batch_size):
                end = min(start + batch_size, len(split_reference))
                candidate_indices = np.asarray(initial_indices[start:end])
                candidate_distances = np.asarray(initial_distances[start:end])
                valid = candidate_indices >= 0
                safe_indices = np.where(valid, candidate_indices, 0)
                candidate_resale_ids = reference_memory.resale_ids[safe_indices]
                candidate_month_indices = reference_memory.month_indices[safe_indices]
                candidate_floor_areas = reference_memory.floor_areas[safe_indices]
                valid &= candidate_resale_ids != query_resale_ids[start:end, None]
                valid &= candidate_month_indices < query_month_indices[start:end, None]
                relative_area_difference = np.abs(
                    candidate_floor_areas / query_floor_areas[start:end, None] - 1
                )
                priority = np.full(candidate_indices.shape, 3, dtype=np.int8)
                priority[valid & (relative_area_difference <= 0.25)] = 2
                priority[valid & (relative_area_difference <= 0.20)] = 1
                priority[valid & (relative_area_difference <= 0.15)] = 0
                selection_order = np.argsort(priority, axis=1, kind='stable')[:, :k_candidates]
                selected_priority = np.take_along_axis(priority, selection_order, axis=1)
                selected_masks = selected_priority < 3
                selected_indices = np.take_along_axis(
                    candidate_indices, selection_order, axis=1
                )
                selected_distances = np.take_along_axis(
                    candidate_distances, selection_order, axis=1
                )
                selected_month_indices = np.take_along_axis(
                    candidate_month_indices, selection_order, axis=1
                )
                selected_time_gaps = (
                    query_month_indices[start:end, None] - selected_month_indices
                )
                final_indices[start:end] = np.where(
                    selected_masks, selected_indices, -1
                ).astype(np.int32, copy=False)
                final_distances[start:end] = np.where(
                    selected_masks, selected_distances, 0
                ).astype(np.float32, copy=False)
                final_time_gaps[start:end] = np.where(
                    selected_masks, selected_time_gaps, 0
                ).astype(np.int16, copy=False)
                final_masks[start:end] = selected_masks
                final_counts[start:end] = selected_masks.sum(axis=1).astype(np.uint8)
            for array in (
                final_indices,
                final_distances,
                final_time_gaps,
                final_masks,
                final_counts,
            ):
                array.flush()
            del (
                final_indices,
                final_distances,
                final_time_gaps,
                final_masks,
                final_counts,
                initial_indices,
                initial_distances,
            )
            split_result[space_name] = FinalCandidateCache(
                indices_path=indices_path,
                distances_path=distances_path,
                time_gaps_path=time_gaps_path,
                masks_path=masks_path,
                counts_path=counts_path,
                shape=shape,
            )
        result[split_name] = MappingProxyType(split_result)
    return MappingProxyType(result)
