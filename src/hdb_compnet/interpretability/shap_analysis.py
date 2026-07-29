"""Fixed-retrieval target numeric GradientSHAP and categorical KernelSHAP."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from hdb_compnet.config import HDBCompNetConfig
from hdb_compnet.data.datasets import ModelSplitData
from hdb_compnet.model.hdb_compnet import HDBCompNet
from hdb_compnet.training.engine import move_batch_to_device


def slice_nested_batch(batch: object, index: int) -> object:
    """Slice one example while preserving the nested batch structure."""
    if isinstance(batch, torch.Tensor):
        return batch if batch.ndim == 0 else batch[index : index + 1]
    if isinstance(batch, dict):
        return {key: slice_nested_batch(value, index) for key, value in batch.items()}
    if isinstance(batch, list):
        return [slice_nested_batch(value, index) for value in batch]
    if isinstance(batch, tuple):
        return tuple(slice_nested_batch(value, index) for value in batch)
    return batch


def expand_single_example_batch(batch: object, batch_size: int) -> object:
    """Expand a fixed single-example retrieval batch for SHAP perturbations."""
    if isinstance(batch, torch.Tensor):
        if batch.ndim == 0:
            return batch
        if batch.shape[0] != 1:
            raise ValueError('The fixed SHAP batch must contain one observation.')
        return batch.expand(batch_size, *batch.shape[1:])
    if isinstance(batch, dict):
        return {
            key: expand_single_example_batch(value, batch_size) for key, value in batch.items()
        }
    if isinstance(batch, list):
        return [expand_single_example_batch(value, batch_size) for value in batch]
    if isinstance(batch, tuple):
        return tuple(expand_single_example_batch(value, batch_size) for value in batch)
    return batch


def iterate_single_example_batches(
    data_loader: DataLoader,
    device: torch.device,
    config: HDBCompNetConfig,
    max_examples: int,
) -> Iterator[dict[str, object]]:
    """Yield fixed candidate batches one observation at a time."""
    processed = 0
    for batch in data_loader:
        batch = move_batch_to_device(
            batch,
            device,
            non_blocking=config.pin_memory and device.type == 'cuda',
        )
        for index in range(batch['numeric'].shape[0]):
            if processed >= max_examples:
                return
            yield slice_nested_batch(batch, index)
            processed += 1


class TargetNumericSHAPWrapper(nn.Module):
    """Vary target numerics while keeping target categories and retrieval fixed."""

    def __init__(self, model: HDBCompNet, fixed_batch: dict[str, object]):
        super().__init__()
        self.model = model
        self.fixed_batch = fixed_batch

    def forward(self, target_numeric: torch.Tensor) -> torch.Tensor:
        batch = expand_single_example_batch(self.fixed_batch, target_numeric.shape[0])
        batch['numeric'] = target_numeric
        return self.model(batch)['prediction']


class TargetCategoricalSHAPWrapper(nn.Module):
    """Vary target categories while keeping numerics and retrieval fixed."""

    def __init__(self, model: HDBCompNet, fixed_batch: dict[str, object]):
        super().__init__()
        self.model = model
        self.fixed_batch = fixed_batch

    def forward(self, target_categorical: torch.Tensor) -> torch.Tensor:
        batch = expand_single_example_batch(self.fixed_batch, target_categorical.shape[0])
        batch['categorical'] = target_categorical.long()
        return self.model(batch)['prediction']


def create_target_shap_baselines(
    train_split_data: ModelSplitData,
    device: torch.device,
    numeric_background_size: int = 128,
    random_state: int = 42,
) -> dict[str, torch.Tensor]:
    """Create train-only numeric background and modal category baseline."""
    rng = np.random.default_rng(random_state)
    size = min(numeric_background_size, len(train_split_data))
    indices = rng.choice(len(train_split_data), size=size, replace=False)
    numeric = torch.from_numpy(
        np.ascontiguousarray(train_split_data.numeric_features[indices], dtype=np.float32)
    ).to(device)
    categorical = np.asarray(
        [
            np.bincount(
                train_split_data.categorical_features[:, index].astype(np.int64)
            ).argmax()
            for index in range(train_split_data.categorical_features.shape[1])
        ],
        dtype=np.int64,
    )
    return {
        'numeric': numeric,
        'categorical': torch.from_numpy(categorical).unsqueeze(0).to(device),
    }


def collect_target_numeric_shap(
    model: HDBCompNet,
    data_loader: DataLoader,
    numeric_background: torch.Tensor,
    device: torch.device,
    config: HDBCompNetConfig,
    max_examples: int = 500,
    n_samples: int = 50,
    stdevs: float = 0.0,
) -> dict[str, object]:
    """Run Captum GradientShap for target numeric features."""
    from captum.attr import GradientShap

    model.eval()
    attribution_chunks: list[torch.Tensor] = []
    value_chunks: list[torch.Tensor] = []
    for example in iterate_single_example_batches(data_loader, device, config, max_examples):
        wrapper = TargetNumericSHAPWrapper(model, example)
        numeric_input = example['numeric'].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            attributions = GradientShap(wrapper).attribute(
                inputs=numeric_input,
                baselines=numeric_background,
                n_samples=n_samples,
                stdevs=stdevs,
            )
        attribution_chunks.append(attributions.detach().float().cpu())
        value_chunks.append(numeric_input.detach().float().cpu())
    shap_values = torch.cat(attribution_chunks).numpy()
    feature_values = torch.cat(value_chunks).numpy()
    importance = (
        pd.DataFrame(
            {
                'feature': config.numeric_columns,
                'feature_type': 'numeric',
                'mean_abs_shap': np.abs(shap_values).mean(axis=0),
                'mean_shap': shap_values.mean(axis=0),
            }
        )
        .sort_values('mean_abs_shap', ascending=False)
        .reset_index(drop=True)
    )
    return {
        'shap_values': shap_values,
        'feature_values': feature_values,
        'feature_names': list(config.numeric_columns),
        'importance': importance,
    }


def collect_target_categorical_shap(
    model: HDBCompNet,
    data_loader: DataLoader,
    categorical_baseline: torch.Tensor,
    device: torch.device,
    config: HDBCompNetConfig,
    max_examples: int = 200,
    n_samples: int = 128,
    perturbations_per_eval: int = 8,
) -> dict[str, object]:
    """Run Captum KernelShap for target category indices."""
    from captum.attr import KernelShap

    model.eval()
    attribution_chunks: list[torch.Tensor] = []
    value_chunks: list[torch.Tensor] = []
    feature_mask = torch.arange(
        len(config.categorical_columns), device=device, dtype=torch.long
    ).unsqueeze(0)
    for example in iterate_single_example_batches(data_loader, device, config, max_examples):
        wrapper = TargetCategoricalSHAPWrapper(model, example)
        categorical_input = example['categorical'].detach().clone().long()
        attributions = KernelShap(wrapper).attribute(
            inputs=categorical_input,
            baselines=categorical_baseline,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=True,
            show_progress=False,
        )
        attribution_chunks.append(attributions.detach().float().cpu())
        value_chunks.append(categorical_input.detach().cpu())
    shap_values = torch.cat(attribution_chunks).numpy()
    feature_values = torch.cat(value_chunks).numpy()
    importance = (
        pd.DataFrame(
            {
                'feature': config.categorical_columns,
                'feature_type': 'categorical',
                'mean_abs_shap': np.abs(shap_values).mean(axis=0),
                'mean_shap': shap_values.mean(axis=0),
            }
        )
        .sort_values('mean_abs_shap', ascending=False)
        .reset_index(drop=True)
    )
    return {
        'shap_values': shap_values,
        'feature_values': feature_values,
        'feature_names': list(config.categorical_columns),
        'importance': importance,
    }


def combine_target_shap_importance(
    numeric_shap_result: dict[str, object],
    categorical_shap_result: dict[str, object],
) -> pd.DataFrame:
    """Combine mean absolute target-feature SHAP importance."""
    importance = pd.concat(
        [
            numeric_shap_result['importance'],
            categorical_shap_result['importance'],
        ],
        ignore_index=True,
    )
    total = importance['mean_abs_shap'].sum()
    importance['importance_share_pct'] = importance['mean_abs_shap'] / total * 100
    return importance.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)


def calculate_target_shap_analysis(
    model: HDBCompNet,
    train_split_data: ModelSplitData,
    explanation_loader: DataLoader,
    device: torch.device,
    config: HDBCompNetConfig,
    numeric_background_size: int = 128,
    numeric_examples: int = 500,
    categorical_examples: int = 200,
    numeric_n_samples: int = 50,
    categorical_n_samples: int = 128,
    categorical_perturbations_per_eval: int = 8,
    random_state: int = 42,
) -> dict[str, object]:
    """Run both fixed-candidate target analyses and combine importance."""
    torch.manual_seed(random_state)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(random_state)
    baselines = create_target_shap_baselines(
        train_split_data, device, numeric_background_size, random_state
    )
    numeric = collect_target_numeric_shap(
        model,
        explanation_loader,
        baselines['numeric'],
        device,
        config,
        numeric_examples,
        numeric_n_samples,
        0.0,
    )
    categorical = collect_target_categorical_shap(
        model,
        explanation_loader,
        baselines['categorical'],
        device,
        config,
        categorical_examples,
        categorical_n_samples,
        categorical_perturbations_per_eval,
    )
    return {
        'numeric': numeric,
        'categorical': categorical,
        'combined_importance': combine_target_shap_importance(numeric, categorical),
        'baselines': baselines,
    }
