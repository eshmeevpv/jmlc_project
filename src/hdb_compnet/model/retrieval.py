"""HDB-CompNet retrieval branch implementation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn

from hdb_compnet.config import HDBCompNetConfig
from hdb_compnet.model.blocks import MLPBlock, safe_masked_softmax
from hdb_compnet.model.encoders import RetrievalRepresentationEncoders


class PairFeatureBuilder(nn.Module):
    """Build five representation interactions plus distance."""

    def forward(
        self,
        target_representation: torch.Tensor,
        candidate_representations: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        expanded_target = target_representation.unsqueeze(1).expand_as(
            candidate_representations
        )
        difference = expanded_target - candidate_representations
        return torch.cat(
            [
                expanded_target,
                candidate_representations,
                difference,
                torch.abs(difference),
                expanded_target * candidate_representations,
                torch.log1p(distances.clamp_min(0)).unsqueeze(-1),
            ],
            dim=-1,
        )


class PairwiseScoreNetwork(nn.Module):
    """Assign one score to every target-candidate pair."""

    def __init__(self, config: HDBCompNetConfig):
        super().__init__()
        self.pair_feature_builder = PairFeatureBuilder()
        self.score_network = nn.Sequential(
            MLPBlock(config.pair_feature_dim, config.score_hidden_dim, config.dropout),
            MLPBlock(
                config.score_hidden_dim,
                config.retrieval_embedding_dim,
                config.dropout,
            ),
            nn.Linear(config.retrieval_embedding_dim, 1),
        )

    def forward(
        self,
        target_representation: torch.Tensor,
        candidate_representations: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_features = self.pair_feature_builder(
            target_representation, candidate_representations, distances
        )
        return pair_features, self.score_network(pair_features).squeeze(-1)


class TemporalRecencyPenalty(nn.Module):
    """Apply a learned non-negative logarithmic penalty for month gaps."""

    def __init__(self, initial_decay_rate: float = 0.05):
        super().__init__()
        self.raw_decay_rate = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_decay_rate)), dtype=torch.float32)
        )

    def forward(
        self, base_scores: torch.Tensor, time_gaps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decay_rate = functional.softplus(self.raw_decay_rate)
        return (
            base_scores - decay_rate * torch.log1p(time_gaps.clamp_min(0)),
            decay_rate,
        )


class RetrievalPredictionAggregator(nn.Module):
    """Predict as an attention-weighted sum of original candidate targets."""

    def forward(
        self,
        candidate_targets: torch.Tensor,
        attention_weights: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_targets = torch.where(
            masks, candidate_targets, torch.zeros_like(candidate_targets)
        )
        contributions = attention_weights * masked_targets
        return contributions.sum(dim=-1), contributions


class RetrievalReliabilityEstimator(nn.Module):
    """Compute the five deterministic reliability features used by gating."""

    def forward(
        self,
        candidate_targets: torch.Tensor,
        distances: torch.Tensor,
        time_gaps: torch.Tensor,
        masks: torch.Tensor,
        attention_weights: torch.Tensor,
        branch_prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_dtype = branch_prediction.dtype
        masks = masks.to(dtype=torch.bool)
        mask_values = masks.to(dtype=torch.float32)
        valid_counts = mask_values.sum(dim=-1)
        availability = valid_counts > 0
        safe_counts = valid_counts.clamp_min(1)
        candidate_fraction = valid_counts / masks.shape[-1]
        mean_distance = (distances.float() * mask_values).sum(dim=-1) / safe_counts
        mean_time_gap = (time_gaps.float() * mask_values).sum(dim=-1) / safe_counts
        deviations = candidate_targets.float() - branch_prediction.float().unsqueeze(-1)
        weighted_variance = (attention_weights.float() * deviations.square()).sum(dim=-1)
        weighted_price_std = torch.sqrt(
            weighted_variance.clamp_min(0) + torch.finfo(torch.float32).eps
        )
        maximum_attention = attention_weights.float().max(dim=-1).values
        reliability = torch.stack(
            [
                candidate_fraction,
                torch.log1p(mean_distance.clamp_min(0)),
                torch.log1p(mean_time_gap.clamp_min(0)),
                weighted_price_std,
                maximum_attention,
            ],
            dim=-1,
        )
        reliability = reliability * availability.unsqueeze(-1)
        return reliability.to(output_dtype), availability


class RetrievalBranch(nn.Module):
    """One complete structural, spatial, or market retrieval branch."""

    def __init__(
        self,
        config: HDBCompNetConfig,
        space_name: str,
        initial_decay_rate: float = 0.05,
    ):
        super().__init__()
        self.space_name = space_name
        self.representation_encoders = RetrievalRepresentationEncoders(config, space_name)
        self.pairwise_score_network = PairwiseScoreNetwork(config)
        self.temporal_recency_penalty = TemporalRecencyPenalty(initial_decay_rate)
        self.prediction_aggregator = RetrievalPredictionAggregator()
        self.reliability_estimator = RetrievalReliabilityEstimator()

    def forward(self, branch_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        target_representation, candidate_representations = self.representation_encoders(
            branch_inputs['target_features'],
            branch_inputs['candidate_features'],
        )
        _, base_scores = self.pairwise_score_network(
            target_representation,
            candidate_representations,
            branch_inputs['distances'],
        )
        adjusted_scores, decay_rate = self.temporal_recency_penalty(
            base_scores, branch_inputs['time_gaps']
        )
        attention_weights = safe_masked_softmax(adjusted_scores, branch_inputs['masks'], dim=-1)
        prediction, candidate_contributions = self.prediction_aggregator(
            branch_inputs['candidate_targets'],
            attention_weights,
            branch_inputs['masks'],
        )
        reliability, availability = self.reliability_estimator(
            branch_inputs['candidate_targets'],
            branch_inputs['distances'],
            branch_inputs['time_gaps'],
            branch_inputs['masks'],
            attention_weights,
            prediction,
        )
        return {
            'prediction': prediction,
            'attention_weights': attention_weights,
            'reliability': reliability,
            'availability': availability,
            'base_scores': base_scores,
            'adjusted_scores': adjusted_scores,
            'candidate_contributions': candidate_contributions,
            'decay_rate': decay_rate,
        }
