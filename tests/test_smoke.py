from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hdb_compnet.config import load_yaml_config
from hdb_compnet.data.cache import FinalCandidateCache
from hdb_compnet.model.hdb_compnet import HDBCompNet


@pytest.mark.parametrize(
    'config_path',
    ['configs/baseline.yaml', 'configs/experiment_2.yaml'],
)
def test_model_forward_is_finite_and_gates_are_masked(config_path: str) -> None:
    config = load_yaml_config(config_path)
    model = HDBCompNet(config).eval()
    batch_size = 2
    masks = torch.zeros(batch_size, config.k_candidates, dtype=torch.bool)
    masks[0, :3] = True
    batch = {
        'numeric': torch.zeros(batch_size, len(config.numeric_columns)),
        'categorical': torch.zeros(
            batch_size, len(config.categorical_columns), dtype=torch.long
        ),
        'target': torch.zeros(batch_size),
        'candidates': {},
    }
    for space_name in config.retrieval_space_names:
        batch['candidates'][space_name] = {
            'numeric': torch.zeros(
                batch_size, config.k_candidates, len(config.numeric_columns)
            ),
            'categorical': torch.zeros(
                batch_size,
                config.k_candidates,
                len(config.categorical_columns),
                dtype=torch.long,
            ),
            'targets': torch.zeros(batch_size, config.k_candidates),
            'distances': torch.zeros(batch_size, config.k_candidates),
            'time_gaps': torch.zeros(batch_size, config.k_candidates),
            'masks': masks,
            'counts': masks.sum(dim=-1).float(),
        }
    with torch.inference_mode():
        output = model(batch)
    assert torch.isfinite(output['prediction']).all()
    assert torch.allclose(output['gate_weights'].sum(dim=-1), torch.ones(batch_size))
    assert torch.equal(
        output['gate_weights'][1, 1:], torch.zeros(len(config.retrieval_space_names))
    )


def test_final_candidate_cache_loads_read_only_memmaps(tmp_path: Path) -> None:
    values = {
        'indices': np.array([[1, -1]], dtype=np.int32),
        'distances': np.array([[0.5, 0]], dtype=np.float32),
        'time_gaps': np.array([[2, 0]], dtype=np.int16),
        'masks': np.array([[True, False]], dtype=np.bool_),
        'counts': np.array([1], dtype=np.uint8),
    }
    for name, array in values.items():
        np.save(tmp_path / f'{name}.npy', array, allow_pickle=False)
    cache = FinalCandidateCache(
        indices_path=tmp_path / 'indices.npy',
        distances_path=tmp_path / 'distances.npy',
        time_gaps_path=tmp_path / 'time_gaps.npy',
        masks_path=tmp_path / 'masks.npy',
        counts_path=tmp_path / 'counts.npy',
    )
    loaded = (
        cache.load_indices(),
        cache.load_distances(),
        cache.load_time_gaps(),
        cache.load_masks(),
        cache.load_counts(),
    )
    assert all(isinstance(array, np.memmap) for array in loaded)
    assert all(not array.flags.writeable for array in loaded)
