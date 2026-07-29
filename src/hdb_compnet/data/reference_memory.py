"""Immutable train-only reference memory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from hdb_compnet.config import CATEGORY_COLUMNS


@dataclass(frozen=True, slots=True)
class ReferenceMemory:
    """Arrays indexed by train-memory positions, never DataFrame indices."""

    resale_ids: np.ndarray
    month_indices: np.ndarray
    floor_areas: np.ndarray
    flat_type_indices: np.ndarray
    targets: np.ndarray
    numeric_features: np.ndarray
    categorical_features: np.ndarray
    retrieval_features: Mapping[str, np.ndarray | csr_matrix]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]

    def __len__(self) -> int:
        return self.resale_ids.shape[0]


def build_reference_memory(
    train_reference: pd.DataFrame,
    train_numeric: pd.DataFrame,
    train_categorical_indices: pd.DataFrame,
    train_retrieval: Mapping[str, np.ndarray | csr_matrix],
) -> ReferenceMemory:
    """Build memory exclusively from the train split."""
    retrieval_features: dict[str, np.ndarray | csr_matrix] = {}
    for space_name in ('structural', 'spatial'):
        retrieval_features[space_name] = (
            train_retrieval[space_name].astype(np.float32, copy=False).tocsr(copy=False)
        )
    retrieval_features['market'] = np.ascontiguousarray(
        train_retrieval['market'], dtype=np.float32
    )
    return ReferenceMemory(
        resale_ids=np.ascontiguousarray(
            train_reference['resale_id'].to_numpy(dtype=np.int64, copy=False)
        ),
        month_indices=np.ascontiguousarray(
            train_reference['month_index'].to_numpy(dtype=np.int32, copy=False)
        ),
        floor_areas=np.ascontiguousarray(
            train_reference['floor_area_sqm'].to_numpy(dtype=np.float32, copy=False)
        ),
        flat_type_indices=np.ascontiguousarray(
            train_categorical_indices['flat_type'].to_numpy(dtype=np.int64, copy=False)
        ),
        targets=np.ascontiguousarray(
            train_reference['target_scaled'].to_numpy(dtype=np.float32, copy=False)
        ),
        numeric_features=np.ascontiguousarray(
            train_numeric.to_numpy(dtype=np.float32, copy=False)
        ),
        categorical_features=np.ascontiguousarray(
            train_categorical_indices[list(CATEGORY_COLUMNS)].to_numpy(
                dtype=np.int64, copy=False
            )
        ),
        retrieval_features=MappingProxyType(retrieval_features),
        numeric_columns=tuple(train_numeric.columns),
        categorical_columns=tuple(CATEGORY_COLUMNS),
    )
