"""AMP-aware train/evaluate loops and notebook experiment orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from hdb_compnet.config import HDBCompNetConfig
from hdb_compnet.model.hdb_compnet import HDBCompNet
from hdb_compnet.training.checkpointing import HDBCompNetCheckpointManager
from hdb_compnet.training.history import (
    EpochStatisticsAggregator,
    HDBCompNetTrainingHistory,
)
from hdb_compnet.training.losses import calculate_hdb_compnet_loss
from hdb_compnet.training.metrics import calculate_regression_metrics


def move_batch_to_device(
    batch: object,
    device: torch.device,
    non_blocking: bool = True,
) -> object:
    """Recursively move tensors while preserving the nested batch structure."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {
            key: move_batch_to_device(value, device, non_blocking)
            for key, value in batch.items()
        }
    if isinstance(batch, list):
        return [move_batch_to_device(value, device, non_blocking) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device, non_blocking) for value in batch)
    return batch


def train_one_epoch(
    model: HDBCompNet,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_scaler: torch.amp.GradScaler,
    device: torch.device,
    config: HDBCompNetConfig,
) -> dict[str, object]:
    """Train for one epoch with AMP and gradient clipping."""
    model.train()
    aggregator = EpochStatisticsAggregator(config)
    amp_enabled = config.use_amp and device.type == 'cuda'
    amp_dtype = getattr(torch, config.amp_dtype)
    for batch in data_loader:
        batch = move_batch_to_device(
            batch,
            device,
            non_blocking=config.pin_memory and device.type == 'cuda',
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            model_output = model(batch)
            loss_output = calculate_hdb_compnet_loss(model_output, batch['target'], config)
            loss = loss_output['loss']
        grad_scaler.scale(loss).backward()
        if config.gradient_clip_norm is not None:
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.gradient_clip_norm
            )
        grad_scaler.step(optimizer)
        grad_scaler.update()
        aggregator.update(loss_output, model_output, batch['target'])
    statistics = aggregator.compute()
    statistics['learning_rate'] = optimizer.param_groups[0]['lr']
    return statistics


def evaluate_one_epoch(
    model: HDBCompNet,
    data_loader: DataLoader,
    device: torch.device,
    config: HDBCompNetConfig,
    target_scaler: object,
    log_transform: Literal['log', 'log1p'] = 'log',
) -> dict[str, object]:
    """Evaluate without gradients and calculate all notebook metrics."""
    model.eval()
    aggregator = EpochStatisticsAggregator(config, collect_predictions=True)
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
                model_output = model(batch)
                loss_output = calculate_hdb_compnet_loss(model_output, batch['target'], config)
            aggregator.update(loss_output, model_output, batch['target'])
    statistics = aggregator.compute()
    statistics['metrics'] = calculate_regression_metrics(
        statistics['targets'],
        statistics['predictions'],
        target_scaler,
        log_transform,
    )
    return statistics


def fit_hdb_compnet(
    model: HDBCompNet,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    target_scaler: object,
    config: HDBCompNetConfig,
    checkpoint_dir: str | Path,
    log_transform: Literal['log', 'log1p'] = 'log',
    resume_from: str | Path | None = None,
    plot_after_each_epoch: bool = True,
) -> dict[str, object]:
    """Fit, select by validation main loss, and restore the best checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_learning_rate,
    )
    amp_enabled = config.use_amp and device.type == 'cuda'
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    training_history = HDBCompNetTrainingHistory(config)
    checkpoint_manager = HDBCompNetCheckpointManager(
        checkpoint_dir,
        patience=config.early_stopping_patience,
        min_delta=0.0,
    )
    start_epoch = 1
    if resume_from is not None:
        start_epoch, _ = checkpoint_manager.load_checkpoint(
            resume_from,
            model,
            device,
            optimizer,
            scheduler,
            grad_scaler,
            training_history,
        )
    for epoch in range(start_epoch, config.max_epochs + 1):
        train_statistics = train_one_epoch(
            model, train_loader, optimizer, grad_scaler, device, config
        )
        validation_statistics = evaluate_one_epoch(
            model,
            validation_loader,
            device,
            config,
            target_scaler,
            log_transform,
        )
        training_history.update(epoch, train_statistics, validation_statistics)
        scheduler.step(validation_statistics['main_loss'])
        status = checkpoint_manager.step(
            epoch,
            model,
            optimizer,
            scheduler,
            grad_scaler,
            training_history,
            validation_statistics,
            config,
        )
        if plot_after_each_epoch:
            from hdb_compnet.visualization.training import plot_training_progress

            plot_training_progress(
                training_history,
                clear_previous=True,
                show_availability=False,
                show_learning_rate=True,
            )
        print(
            f'Epoch {epoch}/{config.max_epochs} | '
            f'Train main loss: {train_statistics["main_loss"]:.6f} | '
            f'Validation main loss: {validation_statistics["main_loss"]:.6f} | '
            f'Validation MAE: '
            f'{validation_statistics["metrics"]["price_mae_sgd"]:,.0f} SGD | '
            f'Validation RMSE: '
            f'{validation_statistics["metrics"]["price_rmse_sgd"]:,.0f} SGD | '
            f'Validation R2: '
            f'{validation_statistics["metrics"]["price_r2"]:.4f} | '
            f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
        )
        print(
            f'Best epoch: {status["best_epoch"]} | '
            f'Epochs without improvement: '
            f'{status["epochs_without_improvement"]}'
        )
        if status['should_stop']:
            print('Early stopping triggered.')
            break
    _, best_checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_manager.best_checkpoint_path, model, device
    )
    model.eval()
    return {
        'model': model,
        'device': device,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'grad_scaler': grad_scaler,
        'training_history': training_history,
        'checkpoint_manager': checkpoint_manager,
        'best_epoch': int(best_checkpoint['epoch']),
        'best_validation_statistics': best_checkpoint['validation_statistics'],
    }


def evaluate_hdb_compnet_on_test(
    fit_output: dict[str, object],
    test_loader: DataLoader,
    target_scaler: object,
    config: HDBCompNetConfig,
    log_transform: Literal['log', 'log1p'] = 'log',
    plot_comparison: bool = True,
) -> dict[str, object]:
    """Evaluate the already selected best model once on test."""
    statistics = evaluate_one_epoch(
        fit_output['model'],
        test_loader,
        fit_output['device'],
        config,
        target_scaler,
        log_transform,
    )
    if plot_comparison:
        from hdb_compnet.visualization.training import (
            plot_final_validation_test_comparison,
        )

        plot_final_validation_test_comparison(fit_output['training_history'], statistics)
    return statistics
