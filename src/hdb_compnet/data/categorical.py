"""Train-only category indices and retrieval one-hot encoders"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

from hdb_compnet.config import CATEGORY_COLUMNS, RETRIEVAL_SPACES


@dataclass(frozen=True, slots=True)
class CategoricalArtifacts:
    """Train-fitted mappings for embeddings and retrieval encoders"""

    category_mappings: Mapping[str, Mapping[object, int]]
    category_cardinalities: Mapping[str, int]
    embedding_dimensions: Mapping[str, int]
    retrieval_one_hot_encoders: Mapping[str, OneHotEncoder]
    unknown_index: int = 0

    def encode_indices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                column: (
                    dataframe[column]
                    .map(self.category_mappings[column])
                    .fillna(self.unknown_index)
                    .astype('int64')
                )
                for column in CATEGORY_COLUMNS
            },
            index=dataframe.index,
        )

    def encode_retrieval(self, dataframe: pd.DataFrame) -> dict[str, csr_matrix]:
        return {
            space_name: encoder.transform(
                dataframe[list(RETRIEVAL_SPACES[space_name]['categorical'])]
            ).tocsr()
            for space_name, encoder in self.retrieval_one_hot_encoders.items()
        }


def fit_categorical_artifacts(train: pd.DataFrame) -> CategoricalArtifacts:
    """Fit category mechanisms using train only"""
    mappings = {
        column: MappingProxyType(
            {
                category: index
                for index, category in enumerate(
                    sorted(train[column].dropna().unique()), start=1
                )
            }
        )
        for column in CATEGORY_COLUMNS
    }
    cardinalities = {column: len(mapping) + 1 for column, mapping in mappings.items()}
    dimensions = {
        column: min(16, max(2, int(np.ceil(np.sqrt(cardinality)))))
        for column, cardinality in cardinalities.items()
    }
    encoders: dict[str, OneHotEncoder] = {}
    for space_name, space_config in RETRIEVAL_SPACES.items():
        if not space_config['categorical']:
            continue
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32)
        encoder.fit(train[list(space_config['categorical'])])
        encoders[space_name] = encoder
    return CategoricalArtifacts(
        category_mappings=MappingProxyType(mappings),
        category_cardinalities=MappingProxyType(cardinalities),
        embedding_dimensions=MappingProxyType(dimensions),
        retrieval_one_hot_encoders=MappingProxyType(encoders),
    )
