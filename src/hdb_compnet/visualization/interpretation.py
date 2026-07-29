"""Gate, retrieval, and target-SHAP plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from hdb_compnet.config import HDBCompNetConfig
from hdb_compnet.interpretability.gating_analysis import (
    calculate_binned_relationship,
    get_gate_columns,
    get_retrieval_analysis_values,
)


def _plot_gate_relationship(
    analysis_df: pd.DataFrame,
    x_column: str,
    x_label: str,
    title: str,
    max_points: int,
    trend_bins: int,
    random_state: int,
) -> pd.DataFrame:
    plotting = analysis_df[[x_column, 'gate_weight']].dropna()
    scatter = (
        plotting.sample(n=max_points, random_state=random_state)
        if len(plotting) > max_points
        else plotting
    )
    trend = calculate_binned_relationship(
        plotting[x_column], plotting['gate_weight'], trend_bins
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.scatter(
        scatter[x_column],
        scatter['gate_weight'] * 100,
        s=10,
        alpha=0.15,
        label='Observations',
    )
    if not trend.empty:
        axis.plot(
            trend['x_mean'],
            trend['y_mean'] * 100,
            marker='o',
            linewidth=2,
            label='Binned mean',
        )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel('Gate weight, %')
    axis.set_ylim(0, 100)
    axis.grid(visible=True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plt.show()
    return trend


def plot_gate_weight_analysis(
    interpretation_df: pd.DataFrame,
    config: HDBCompNetConfig,
    bins: int = 40,
) -> pd.Series:
    """Plot mean and distributions of gate weights."""
    gate_columns = get_gate_columns(config)
    weights = interpretation_df[list(gate_columns.values())].rename(
        columns={column: branch.capitalize() for branch, column in gate_columns.items()}
    )
    means = weights.mean() * 100
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(means.index, means.values)
    axis.set_title('Mean HDB-CompNet gate weights')
    axis.set_xlabel('Branch')
    axis.set_ylabel('Mean gate weight, %')
    axis.set_ylim(0, max(100, means.max() * 1.15))
    axis.grid(visible=True, axis='y', alpha=0.3)
    for bar, value in zip(bars, means.values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{value:.2f}%',
            ha='center',
            va='bottom',
        )
    figure.tight_layout()
    plt.show()
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(
        [weights[column].dropna() * 100 for column in weights.columns],
        bins=bins,
        density=True,
        histtype='step',
        linewidth=2,
        label=list(weights.columns),
    )
    axis.set_title('Distribution of HDB-CompNet gate weights')
    axis.set_xlabel('Gate weight, %')
    axis.set_ylabel('Density')
    axis.set_xlim(0, 100)
    axis.grid(visible=True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plt.show()
    return means


def plot_retrieval_branch_analysis(
    interpretation_df: pd.DataFrame,
    config: HDBCompNetConfig,
    max_points: int = 20000,
    trend_bins: int = 20,
    random_state: int = 42,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Plot non-causal binned relationships for retrieval reliability."""
    trends: dict[str, dict[str, pd.DataFrame]] = {}
    for space_name in config.retrieval_space_names:
        analysis = get_retrieval_analysis_values(interpretation_df, space_name, config)
        trends[space_name] = {}
        for column, label in (
            ('candidate_count', 'Number of available candidates'),
            ('mean_distance', 'Mean retrieval distance'),
            ('mean_time_gap', 'Mean time gap, months'),
        ):
            trends[space_name][column] = _plot_gate_relationship(
                analysis,
                column,
                label,
                f'{space_name.capitalize()} gate weight and {label.lower()}',
                max_points,
                trend_bins,
                random_state,
            )
    return trends


def plot_target_shap_importance(
    importance: pd.DataFrame,
    top_n: int = 30,
) -> None:
    """Plot mean absolute target-feature SHAP values."""
    plotting = importance.nlargest(top_n, 'mean_abs_shap').sort_values(
        'mean_abs_shap', ascending=True
    )
    figure, axis = plt.subplots(figsize=(10, max(6, len(plotting) * 0.35)))
    bars = axis.barh(plotting['feature'], plotting['mean_abs_shap'])
    axis.set_title('Target feature importance in HDB-CompNet')
    axis.set_xlabel('Mean absolute SHAP value')
    axis.set_ylabel('Feature')
    axis.grid(visible=True, axis='x', alpha=0.3)
    for bar, value in zip(bars, plotting['importance_share_pct'], strict=True):
        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f' {value:.2f}%',
            va='center',
        )
    figure.tight_layout()
    plt.show()
