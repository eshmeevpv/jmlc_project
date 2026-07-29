"""Notebook training-history and final comparison plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from hdb_compnet.training.history import HDBCompNetTrainingHistory


def get_training_history_data(
    training_history: HDBCompNetTrainingHistory | Mapping[str, object],
) -> Mapping[str, object]:
    """Return the underlying history mapping."""
    if isinstance(training_history, HDBCompNetTrainingHistory):
        return training_history.history
    return training_history


def plot_history_series(
    epochs: Sequence[int],
    series: Mapping[str, Sequence[float]],
    title: str,
    y_label: str,
) -> None:
    """Plot one or more epoch series."""
    figure, axis = plt.subplots(figsize=(10, 5))
    for label, values in series.items():
        axis.plot(epochs, values, marker='o', label=label)
    axis.set_title(title)
    axis.set_xlabel('Epoch')
    axis.set_ylabel(y_label)
    axis.grid(visible=True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plt.show()


def plot_model_losses(
    training_history: HDBCompNetTrainingHistory | Mapping[str, object],
) -> None:
    """Plot total and main train/validation losses."""
    history = get_training_history_data(training_history)
    plot_history_series(
        history['epoch'],
        {
            'Train': history['train']['loss'],
            'Validation': history['validation']['loss'],
        },
        'Total loss',
        'Loss',
    )
    plot_history_series(
        history['epoch'],
        {
            'Train': history['train']['main_loss'],
            'Validation': history['validation']['main_loss'],
        },
        'Main Huber loss',
        'Huber loss',
    )


def plot_branch_losses(
    training_history: HDBCompNetTrainingHistory | Mapping[str, object],
    split_name: str,
) -> None:
    """Plot the four auxiliary branch losses for one split."""
    if split_name not in {'train', 'validation'}:
        raise ValueError('split_name must be train or validation.')
    history = get_training_history_data(training_history)
    plot_history_series(
        history['epoch'],
        {
            branch.capitalize(): history[split_name]['branch_losses'][branch]
            for branch in ('baseline', 'structural', 'spatial', 'market')
        },
        f'{split_name.capitalize()} branch losses',
        'Huber loss',
    )


def plot_branch_availability(
    training_history: HDBCompNetTrainingHistory | Mapping[str, object],
    split_name: str,
) -> None:
    """Plot retrieval branch availability fractions."""
    if split_name not in {'train', 'validation'}:
        raise ValueError('split_name must be train or validation.')
    history = get_training_history_data(training_history)
    plot_history_series(
        history['epoch'],
        {
            branch.capitalize(): (
                np.asarray(history[split_name]['branch_availability'][branch]) * 100
            )
            for branch in ('baseline', 'structural', 'spatial', 'market')
        },
        f'{split_name.capitalize()} branch availability',
        'Availability, %',
    )


def plot_validation_metrics(
    training_history: HDBCompNetTrainingHistory | Mapping[str, object],
) -> None:
    """Plot validation price MAE, RMSE, and R²."""
    history = get_training_history_data(training_history)
    metrics = history['validation']['metrics']
    plot_history_series(
        history['epoch'],
        {'MAE, SGD': metrics['price_mae_sgd'], 'RMSE, SGD': metrics['price_rmse_sgd']},
        'Validation price errors',
        'SGD',
    )
    plot_history_series(
        history['epoch'],
        {'R²': metrics['price_r2']},
        'Validation price R²',
        'R²',
    )


def plot_learning_rate(
    training_history: HDBCompNetTrainingHistory | Mapping[str, object],
) -> None:
    """Plot learning-rate schedule."""
    history = get_training_history_data(training_history)
    plot_history_series(
        history['epoch'],
        {'Learning rate': history['learning_rate']},
        'Learning rate',
        'Learning rate',
    )


def plot_training_progress(
    training_history: HDBCompNetTrainingHistory | Mapping[str, object],
    *,
    clear_previous: bool = False,
    show_availability: bool = False,
    show_learning_rate: bool = True,
) -> None:
    """Render the notebook's standard collection of training plots."""
    if clear_previous:
        try:
            from IPython.display import clear_output

            clear_output(wait=True)
        except ImportError:
            pass
    plot_model_losses(training_history)
    plot_branch_losses(training_history, 'train')
    plot_branch_losses(training_history, 'validation')
    if show_availability:
        plot_branch_availability(training_history, 'validation')
    plot_validation_metrics(training_history)
    if show_learning_rate:
        plot_learning_rate(training_history)


def plot_validation_test_metric_comparison(
    validation_metrics: Mapping[str, float],
    test_metrics: Mapping[str, float],
) -> None:
    """Compare final validation and test price metrics."""
    labels = ('MAE, SGD', 'RMSE, SGD', 'R²')
    validation = (
        validation_metrics['price_mae_sgd'],
        validation_metrics['price_rmse_sgd'],
        validation_metrics['price_r2'],
    )
    test = (
        test_metrics['price_mae_sgd'],
        test_metrics['price_rmse_sgd'],
        test_metrics['price_r2'],
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for index, axis in enumerate(axes):
        axis.bar(['Validation', 'Test'], [validation[index], test[index]])
        axis.set_title(labels[index])
        axis.grid(visible=True, axis='y', alpha=0.3)
    figure.tight_layout()
    plt.show()


def plot_final_validation_test_comparison(
    training_history: HDBCompNetTrainingHistory | Mapping[str, object],
    test_statistics: Mapping[str, object],
) -> None:
    """Compare the last recorded validation metrics with test metrics."""
    history = get_training_history_data(training_history)
    validation_metrics = {
        metric_name: values[-1]
        for metric_name, values in history['validation']['metrics'].items()
    }
    plot_validation_test_metric_comparison(validation_metrics, test_statistics['metrics'])
