"""Lightweight model split containers and index-only datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from hdb_compnet.config import CATEGORY_COLUMNS


@dataclass(frozen=True, slots=True)
class ModelSplitData:
    """In-memory target features. Candidate arrays remain path-based caches."""

    numeric_features: np.ndarray
    categorical_features: np.ndarray
    targets: np.ndarray
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]

    def __len__(self) -> int:
        return self.targets.shape[0]


def build_model_split_data(
    numeric_features: pd.DataFrame,
    categorical_features: pd.DataFrame,
    reference: pd.DataFrame,
) -> ModelSplitData:
    """Build the compact arrays required by a split's collator."""
    return ModelSplitData(
        numeric_features=np.ascontiguousarray(
            numeric_features.to_numpy(dtype=np.float32, copy=False)
        ),
        categorical_features=np.ascontiguousarray(
            categorical_features[list(CATEGORY_COLUMNS)].to_numpy(dtype=np.int64, copy=False)
        ),
        targets=np.ascontiguousarray(
            reference['target_scaled'].to_numpy(dtype=np.float32, copy=False)
        ),
        numeric_columns=tuple(numeric_features.columns),
        categorical_columns=tuple(CATEGORY_COLUMNS),
    )


class TransactionIndexDataset(Dataset[int]):
    """Return only row positions. Collator performs candidate materialization."""

    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index
