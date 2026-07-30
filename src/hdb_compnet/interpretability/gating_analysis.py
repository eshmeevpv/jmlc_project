"""Collect branch, gating, contribution, and reliability diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hdb_compnet.config import HDBCompNetConfig
from hdb_compnet.model.hdb_compnet import HDBCompNet
from hdb_compnet.training.engine import move_batch_to_device

RELIABILITY_NAMES = (
    'candidate_fraction',
    'log_mean_distance',
    'log_mean_time_gap',
    'weighted_price_std',
    'max_attention',
)


def get_gate_columns(config: HDBCompNetConfig) -> dict[str, str]:
    """Map branch names to interpretation-frame columns."""
    return {branch_name: f'gate_{branch_name}' for branch_name in config.branch_names}


def get_retrieval_analysis_values(
    interpretation_df: pd.DataFrame,
    space_name: str,
    config: HDBCompNetConfig,
) -> pd.DataFrame:
    """Normalize reliability columns for plotting or tabular analysis."""
    analysis = pd.DataFrame(index=interpretation_df.index)
    analysis['gate_weight'] = interpretation_df[f'gate_{space_name}']
    if f'{space_name}_candidate_count' in interpretation_df:
        analysis['candidate_count'] = interpretation_df[f'{space_name}_candidate_count']
    else:
        analysis['candidate_count'] = (
            interpretation_df[f'{space_name}_candidate_fraction'] * config.k_candidates
        )
    if f'{space_name}_mean_distance' in interpretation_df:
        analysis['mean_distance'] = interpretation_df[f'{space_name}_mean_distance']
    else:
        analysis['mean_distance'] = np.expm1(
            interpretation_df[f'{space_name}_log_mean_distance']
        )
    if f'{space_name}_mean_time_gap' in interpretation_df:
        analysis['mean_time_gap'] = interpretation_df[f'{space_name}_mean_time_gap']
    else:
        analysis['mean_time_gap'] = np.expm1(
            interpretation_df[f'{space_name}_log_mean_time_gap']
        )
    availability_column = f'{space_name}_availability'
    if availability_column in interpretation_df:
        analysis = analysis.loc[interpretation_df[availability_column].astype(bool)]
    return analysis.replace([np.inf, -np.inf], np.nan).dropna()


def calculate_binned_relationship(
    x: pd.Series,
    y: pd.Series,
    n_bins: int,
) -> pd.DataFrame:
    """Calculate quantile-bin means without implying causality."""
    unique_count = x.nunique()
    if unique_count < 2:
        return pd.DataFrame(columns=['x_mean', 'y_mean'])
    quantile_bins = pd.qcut(x, q=min(n_bins, unique_count), duplicates='drop')
    return (
        pd.DataFrame({'x': x, 'y': y, 'bin': quantile_bins})
        .groupby('bin', observed=True)
        .agg(x_mean=('x', 'mean'), y_mean=('y', 'mean'))
        .reset_index(drop=True)
    )


def collect_hdb_compnet_interpretation_outputs(
    model: HDBCompNet,
    data_loader: DataLoader,
    device: torch.device,
    config: HDBCompNetConfig,
) -> pd.DataFrame:
    """Collect predictions and diagnostics without changing retrieval candidates."""
    model.eval()
    chunks: list[pd.DataFrame] = []
    amp_enabled = config.use_amp and device.type == 'cuda'
    amp_dtype = getattr(torch, config.amp_dtype)
    with torch.inference_mode():
        for batch in data_loader:
            batch = move_batch_to_device(
                batch,
                device,
                non_blocking=config.pin_memory and device.type == 'cuda',
            )
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(batch)
            batch_data: dict[str, object] = {
                'target_scaled': batch['target'].detach().float().cpu().numpy(),
                'prediction_scaled': (output['prediction'].detach().float().cpu().numpy()),
            }
            gate_weights = output['gate_weights'].detach().float().cpu().numpy()
            branch_predictions = (
                output['stacked_branch_predictions'].detach().float().cpu().numpy()
            )
            contributions = (
                output['weighted_branch_contributions'].detach().float().cpu().numpy()
            )
            for index, branch_name in enumerate(config.branch_names):
                batch_data[f'gate_{branch_name}'] = gate_weights[:, index]
                batch_data[f'prediction_{branch_name}'] = branch_predictions[:, index]
                batch_data[f'contribution_{branch_name}'] = contributions[:, index]
            for space_name in config.retrieval_space_names:
                retrieval_output = output['retrieval_outputs'][space_name]
                reliability = retrieval_output['reliability'].detach().float().cpu().numpy()
                availability = (
                    retrieval_output['availability'].detach().cpu().numpy().astype(bool)
                )
                for index, name in enumerate(RELIABILITY_NAMES):
                    batch_data[f'{space_name}_{name}'] = reliability[:, index]
                batch_data[f'{space_name}_availability'] = availability
                batch_data[f'{space_name}_candidate_count'] = np.rint(
                    reliability[:, 0] * config.k_candidates
                ).astype(np.int32)
                batch_data[f'{space_name}_mean_distance'] = np.expm1(reliability[:, 1])
                batch_data[f'{space_name}_mean_time_gap'] = np.expm1(reliability[:, 2])
                batch_data[f'{space_name}_decay_rate'] = float(
                    retrieval_output['decay_rate'].detach().float().cpu().item()
                )
            chunks.append(pd.DataFrame(batch_data))
    return pd.concat(chunks, ignore_index=True)
