"""Construction of three retrieval feature spaces"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from hdb_compnet.config import RETRIEVAL_SPACES

RETRIEVAL_REFERENCE_COLUMNS = (
    'resale_id',
    'month_index',
    'floor_area_sqm',
    'flat_type',
    'target_scaled',
)
RETRIEVAL_GROUP_COLUMN = 'flat_type'


def build_retrieval_matrices(
    numeric_features: pd.DataFrame,
    categorical_features: Mapping[str, csr_matrix],
) -> dict[str, np.ndarray | csr_matrix]:
    """Build structural/spatial CSR matrices and the dense market matrix"""
    retrieval_matrices: dict[str, np.ndarray | csr_matrix] = {}
    for space_name, space_config in RETRIEVAL_SPACES.items():
        numeric_matrix = numeric_features[list(space_config['numeric'])].to_numpy(
            dtype=np.float32, copy=True
        )
        if space_config['categorical']:
            retrieval_matrices[space_name] = hstack(
                [csr_matrix(numeric_matrix), categorical_features[space_name]],
                format='csr',
                dtype=np.float32,
            )
        else:
            retrieval_matrices[space_name] = np.ascontiguousarray(
                numeric_matrix, dtype=np.float32
            )
    return retrieval_matrices
