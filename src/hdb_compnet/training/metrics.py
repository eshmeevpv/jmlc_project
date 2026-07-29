"""Regression metrics in scaled, log, and price space."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def to_numpy_1d(values: torch.Tensor | np.ndarray) -> np.ndarray:
    """Detach a tensor or normalize an array to float64 vector form."""
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def inverse_transform_target(
    values_scaled: torch.Tensor | np.ndarray,
    target_scaler: object,
    log_transform: Literal['log', 'log1p'] = 'log',
) -> tuple[np.ndarray, np.ndarray]:
    """Undo standardization and the notebook-selected logarithm."""
    values_scaled = to_numpy_1d(values_scaled)
    values_log = target_scaler.inverse_transform(values_scaled.reshape(-1, 1)).reshape(-1)
    if log_transform == 'log':
        values_price = np.exp(values_log)
    elif log_transform == 'log1p':
        values_price = np.expm1(values_log)
    else:
        raise ValueError(f'Unsupported log transform: {log_transform}')
    return values_log, values_price


def calculate_regression_metrics(
    target_scaled: torch.Tensor | np.ndarray,
    prediction_scaled: torch.Tensor | np.ndarray,
    target_scaler: object,
    log_transform: Literal['log', 'log1p'] = 'log',
) -> dict[str, float]:
    """Calculate individual-observation metrics at all three scales."""
    target_scaled = to_numpy_1d(target_scaled)
    prediction_scaled = to_numpy_1d(prediction_scaled)
    target_log, target_price = inverse_transform_target(
        target_scaled, target_scaler, log_transform
    )
    prediction_log, prediction_price = inverse_transform_target(
        prediction_scaled, target_scaler, log_transform
    )
    return {
        'scaled_mae': float(mean_absolute_error(target_scaled, prediction_scaled)),
        'scaled_rmse': float(root_mean_squared_error(target_scaled, prediction_scaled)),
        'scaled_r2': float(r2_score(target_scaled, prediction_scaled)),
        'log_mae': float(mean_absolute_error(target_log, prediction_log)),
        'log_rmse': float(root_mean_squared_error(target_log, prediction_log)),
        'log_r2': float(r2_score(target_log, prediction_log)),
        'price_mae_sgd': float(mean_absolute_error(target_price, prediction_price)),
        'price_rmse_sgd': float(root_mean_squared_error(target_price, prediction_price)),
        'price_r2': float(r2_score(target_price, prediction_price)),
    }
