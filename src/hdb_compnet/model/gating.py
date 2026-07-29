"""Availability-aware gating of the four branch predictions."""

from __future__ import annotations

import torch
from torch import nn

from hdb_compnet.config import HDBCompNetConfig
from hdb_compnet.model.blocks import MLPBlock, safe_masked_softmax


class GatingNetwork(nn.Module):
    """Produce masked softmax weights from representation and reliability."""

    def __init__(self, config: HDBCompNetConfig):
        super().__init__()
        self.branch_names = config.branch_names
        self.retrieval_space_names = config.retrieval_space_names
        self.gate_network = nn.Sequential(
            MLPBlock(config.gate_input_dim, config.gate_hidden_dim, config.dropout),
            MLPBlock(
                config.gate_hidden_dim,
                config.gate_hidden_dim // 2,
                config.dropout,
            ),
            nn.Linear(config.gate_hidden_dim // 2, len(config.branch_names)),
        )

    def forward(
        self,
        global_representation: torch.Tensor,
        branch_predictions: dict[str, torch.Tensor],
        retrieval_outputs: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        stacked_predictions = torch.stack(
            [branch_predictions[branch_name] for branch_name in self.branch_names],
            dim=-1,
        )
        reliability_features = torch.cat(
            [
                retrieval_outputs[space_name]['reliability']
                for space_name in self.retrieval_space_names
            ],
            dim=-1,
        )
        gate_features = torch.cat(
            [global_representation, stacked_predictions, reliability_features],
            dim=-1,
        )
        gate_logits = self.gate_network(gate_features)
        branch_availability = torch.stack(
            [
                torch.ones_like(branch_predictions['baseline'], dtype=torch.bool),
                *[
                    retrieval_outputs[space_name]['availability']
                    for space_name in self.retrieval_space_names
                ],
            ],
            dim=-1,
        )
        gate_weights = safe_masked_softmax(gate_logits, branch_availability, dim=-1)
        return {
            'weights': gate_weights,
            'logits': gate_logits,
            'availability': branch_availability,
            'features': gate_features,
        }


def combine_branch_predictions(
    branch_predictions: dict[str, torch.Tensor],
    gate_weights: torch.Tensor,
    branch_names: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return final prediction, stacked branches, and weighted contributions."""
    stacked = torch.stack(
        [branch_predictions[branch_name] for branch_name in branch_names], dim=-1
    )
    weighted = gate_weights * stacked
    return weighted.sum(dim=-1), stacked, weighted
