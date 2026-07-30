"""Huber objective with masked auxiliary branch losses."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from hdb_compnet.config import HDBCompNetConfig


def calculate_hdb_compnet_loss(
    model_output: dict[str, object],
    target: torch.Tensor,
    config: HDBCompNetConfig,
) -> dict[str, object]:
    """Calculate main and availability-masked auxiliary Huber losses."""
    target = target.float()
    prediction = model_output['prediction']
    branch_predictions = model_output['branch_predictions']
    retrieval_outputs = model_output['retrieval_outputs']
    main_loss = functional.huber_loss(
        prediction.float(),
        target,
        reduction='mean',
        delta=config.huber_delta,
    )
    baseline_loss = functional.huber_loss(
        branch_predictions['baseline'].float(),
        target,
        reduction='mean',
        delta=config.huber_delta,
    )
    branch_losses = {'baseline': baseline_loss}
    auxiliary_loss_sum = baseline_loss
    active_branch_count = torch.ones((), device=target.device, dtype=target.dtype)
    for space_name in config.retrieval_space_names:
        retrieval_output = retrieval_outputs[space_name]
        elementwise_loss = functional.huber_loss(
            retrieval_output['prediction'].float(),
            target,
            reduction='none',
            delta=config.huber_delta,
        )
        availability = retrieval_output['availability'].to(dtype=elementwise_loss.dtype)
        available_count = availability.sum()
        branch_loss = (elementwise_loss * availability).sum() / available_count.clamp_min(1)
        branch_is_active = (available_count > 0).to(dtype=elementwise_loss.dtype)
        branch_losses[space_name] = branch_loss
        auxiliary_loss_sum = auxiliary_loss_sum + branch_loss * branch_is_active
        active_branch_count = active_branch_count + branch_is_active
    auxiliary_loss = auxiliary_loss_sum / active_branch_count
    weighted_auxiliary_loss = config.auxiliary_loss_weight * auxiliary_loss
    return {
        'loss': main_loss + weighted_auxiliary_loss,
        'main_loss': main_loss,
        'auxiliary_loss': auxiliary_loss,
        'weighted_auxiliary_loss': weighted_auxiliary_loss,
        'branch_losses': branch_losses,
        'active_branch_count': active_branch_count,
    }
