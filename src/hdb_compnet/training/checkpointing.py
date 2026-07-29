"""Atomic state-dict checkpointing and early stopping."""

from __future__ import annotations

from pathlib import Path

import torch

from hdb_compnet.config import HDBCompNetConfig
from hdb_compnet.model.hdb_compnet import HDBCompNet
from hdb_compnet.training.history import HDBCompNetTrainingHistory
from hdb_compnet.utils.serialization import (
    compact_epoch_statistics,
    convert_to_serializable,
)


class HDBCompNetCheckpointManager:
    """Save latest/best checkpoints selected by validation main loss."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        patience: int,
        min_delta: float = 0.0,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / 'best_checkpoint.pt'
        self.latest_checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        self.patience = patience
        self.min_delta = min_delta
        self.best_validation_main_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience

    def state_dict(self) -> dict[str, object]:
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'best_validation_main_loss': self.best_validation_main_loss,
            'best_epoch': self.best_epoch,
            'epochs_without_improvement': self.epochs_without_improvement,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.patience = int(state_dict['patience'])
        self.min_delta = float(state_dict['min_delta'])
        self.best_validation_main_loss = float(state_dict['best_validation_main_loss'])
        self.best_epoch = int(state_dict['best_epoch'])
        self.epochs_without_improvement = int(state_dict['epochs_without_improvement'])

    def _build_checkpoint(
        self,
        epoch: int,
        model: HDBCompNet,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        grad_scaler: torch.amp.GradScaler,
        training_history: HDBCompNetTrainingHistory,
        validation_statistics: dict[str, object],
        config: HDBCompNetConfig,
    ) -> dict[str, object]:
        return {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'grad_scaler_state_dict': grad_scaler.state_dict(),
            'training_history': training_history.state_dict(),
            'checkpoint_manager_state': self.state_dict(),
            'validation_statistics': compact_epoch_statistics(validation_statistics),
            'config': convert_to_serializable(config.to_dict()),
        }

    @staticmethod
    def _save_checkpoint(checkpoint: dict[str, object], checkpoint_path: Path) -> None:
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)

    def step(
        self,
        epoch: int,
        model: HDBCompNet,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        grad_scaler: torch.amp.GradScaler,
        training_history: HDBCompNetTrainingHistory,
        validation_statistics: dict[str, object],
        config: HDBCompNetConfig,
    ) -> dict[str, object]:
        validation_main_loss = float(validation_statistics['main_loss'])
        improved = validation_main_loss < self.best_validation_main_loss - self.min_delta
        if improved:
            self.best_validation_main_loss = validation_main_loss
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        checkpoint = self._build_checkpoint(
            epoch,
            model,
            optimizer,
            scheduler,
            grad_scaler,
            training_history,
            validation_statistics,
            config,
        )
        self._save_checkpoint(checkpoint, self.latest_checkpoint_path)
        if improved:
            self._save_checkpoint(checkpoint, self.best_checkpoint_path)
        return {
            'improved': improved,
            'should_stop': self.should_stop,
            'best_epoch': self.best_epoch,
            'best_validation_main_loss': self.best_validation_main_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
        }

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: HDBCompNet,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: object | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        training_history: HDBCompNetTrainingHistory | None = None,
    ) -> tuple[int, dict[str, object]]:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if grad_scaler is not None:
            grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
        if training_history is not None:
            training_history.load_state_dict(checkpoint['training_history'])
        self.load_state_dict(checkpoint['checkpoint_manager_state'])
        return int(checkpoint['epoch']) + 1, checkpoint
