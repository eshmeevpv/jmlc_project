"""Epoch aggregation and serializable training history."""

from __future__ import annotations

from copy import deepcopy

import torch

from hdb_compnet.config import HDBCompNetConfig


class EpochStatisticsAggregator:
    """Aggregate sample-weighted losses and optional prediction tensors."""

    def __init__(
        self,
        config: HDBCompNetConfig,
        collect_predictions: bool = False,
    ):
        self.config = config
        self.collect_predictions = collect_predictions
        self.reset()

    def reset(self) -> None:
        self.sample_count = 0
        self.loss_sums = {
            'loss': 0.0,
            'main_loss': 0.0,
            'auxiliary_loss': 0.0,
            'weighted_auxiliary_loss': 0.0,
        }
        self.branch_loss_sums = {branch_name: 0.0 for branch_name in self.config.branch_names}
        self.branch_sample_counts = {branch_name: 0 for branch_name in self.config.branch_names}
        self.prediction_chunks: list[torch.Tensor] = []
        self.target_chunks: list[torch.Tensor] = []

    def update(
        self,
        loss_output: dict[str, object],
        model_output: dict[str, object],
        target: torch.Tensor,
    ) -> None:
        batch_size = target.shape[0]
        self.sample_count += batch_size
        for loss_name in self.loss_sums:
            self.loss_sums[loss_name] += (
                loss_output[loss_name].detach().float().item() * batch_size
            )
        baseline_loss = loss_output['branch_losses']['baseline']
        self.branch_loss_sums['baseline'] += baseline_loss.detach().float().item() * batch_size
        self.branch_sample_counts['baseline'] += batch_size
        retrieval_outputs = model_output['retrieval_outputs']
        for space_name in self.config.retrieval_space_names:
            availability = retrieval_outputs[space_name]['availability']
            available_count = int(availability.sum().item())
            if available_count == 0:
                continue
            branch_loss = loss_output['branch_losses'][space_name]
            self.branch_loss_sums[space_name] += (
                branch_loss.detach().float().item() * available_count
            )
            self.branch_sample_counts[space_name] += available_count
        if self.collect_predictions:
            self.prediction_chunks.append(model_output['prediction'].detach().float().cpu())
            self.target_chunks.append(target.detach().float().cpu())

    def compute(self) -> dict[str, object]:
        if self.sample_count == 0:
            raise RuntimeError('No batches were added to the aggregator.')
        statistics: dict[str, object] = {
            name: value / self.sample_count for name, value in self.loss_sums.items()
        }
        statistics['branch_losses'] = {
            branch_name: (
                self.branch_loss_sums[branch_name] / self.branch_sample_counts[branch_name]
                if self.branch_sample_counts[branch_name] > 0
                else float('nan')
            )
            for branch_name in self.config.branch_names
        }
        statistics['branch_availability'] = {
            branch_name: self.branch_sample_counts[branch_name] / self.sample_count
            for branch_name in self.config.branch_names
        }
        statistics['sample_count'] = self.sample_count
        if self.collect_predictions:
            statistics['predictions'] = torch.cat(self.prediction_chunks, dim=0)
            statistics['targets'] = torch.cat(self.target_chunks, dim=0)
        return statistics


class HDBCompNetTrainingHistory:
    """Serializable per-epoch loss, availability, and validation metrics."""

    loss_names = (
        'loss',
        'main_loss',
        'auxiliary_loss',
        'weighted_auxiliary_loss',
    )
    metric_names = (
        'scaled_mae',
        'scaled_rmse',
        'scaled_r2',
        'log_mae',
        'log_rmse',
        'log_r2',
        'price_mae_sgd',
        'price_rmse_sgd',
        'price_r2',
    )

    def __init__(self, config: HDBCompNetConfig):
        self.branch_names = tuple(config.branch_names)
        self.history = {
            'epoch': [],
            'learning_rate': [],
            'train': self._create_split_history(include_metrics=False),
            'validation': self._create_split_history(include_metrics=True),
        }

    def _create_split_history(self, include_metrics: bool) -> dict[str, object]:
        split: dict[str, object] = {loss_name: [] for loss_name in self.loss_names}
        split['branch_losses'] = {branch_name: [] for branch_name in self.branch_names}
        split['branch_availability'] = {branch_name: [] for branch_name in self.branch_names}
        if include_metrics:
            split['metrics'] = {metric_name: [] for metric_name in self.metric_names}
        return split

    def _append_split_statistics(self, split_name: str, statistics: dict[str, object]) -> None:
        split = self.history[split_name]
        for loss_name in self.loss_names:
            split[loss_name].append(float(statistics[loss_name]))
        for branch_name in self.branch_names:
            split['branch_losses'][branch_name].append(
                float(statistics['branch_losses'][branch_name])
            )
            split['branch_availability'][branch_name].append(
                float(statistics['branch_availability'][branch_name])
            )
        if 'metrics' in split:
            for metric_name in self.metric_names:
                split['metrics'][metric_name].append(float(statistics['metrics'][metric_name]))

    def update(
        self,
        epoch: int,
        train_statistics: dict[str, object],
        validation_statistics: dict[str, object],
    ) -> None:
        self.history['epoch'].append(int(epoch))
        self.history['learning_rate'].append(float(train_statistics['learning_rate']))
        self._append_split_statistics('train', train_statistics)
        self._append_split_statistics('validation', validation_statistics)

    def state_dict(self) -> dict[str, object]:
        return deepcopy(self.history)

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.history = deepcopy(state_dict)

    def __getitem__(self, key: str) -> object:
        return self.history[key]

    def __len__(self) -> int:
        return len(self.history['epoch'])
