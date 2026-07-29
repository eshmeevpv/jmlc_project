"""Reusable PyTorch blocks."""

from __future__ import annotations

import torch
from torch import nn

from hdb_compnet.config import HDBCompNetConfig


class CategoricalEmbeddingBlock(nn.Module):
    """Shared embedding tables used by the global and retrieval encoders."""

    def __init__(self, config: HDBCompNetConfig):
        super().__init__()
        self.categorical_columns = config.categorical_columns
        self.embeddings = nn.ModuleDict(
            {
                column: nn.Embedding(
                    num_embeddings=config.category_cardinalities[column],
                    embedding_dim=config.embedding_dimensions[column],
                    padding_idx=0,
                )
                for column in self.categorical_columns
            }
        )

    def forward(
        self,
        categorical_indices: torch.Tensor,
        column_indices: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if column_indices is None:
            column_indices = tuple(range(len(self.categorical_columns)))
        return torch.cat(
            [
                self.embeddings[self.categorical_columns[index]](
                    categorical_indices[..., index]
                )
                for index in column_indices
            ],
            dim=-1,
        )


class MLPBlock(nn.Module):
    """Linear, LayerNorm, GELU, and dropout block."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def safe_masked_softmax(
    scores: torch.Tensor,
    masks: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Softmax with exact zeros for padding and fully masked rows."""
    masks = masks.to(dtype=torch.bool)
    original_dtype = scores.dtype
    working_scores = scores.float()
    masked_scores = working_scores.masked_fill(~masks, torch.finfo(working_scores.dtype).min)
    has_candidates = masks.any(dim=dim, keepdim=True)
    max_scores = masked_scores.max(dim=dim, keepdim=True).values
    max_scores = torch.where(has_candidates, max_scores, torch.zeros_like(max_scores))
    exponentials = torch.exp(masked_scores - max_scores) * masks.to(working_scores.dtype)
    denominator = exponentials.sum(dim=dim, keepdim=True)
    attention_weights = exponentials / denominator.clamp_min(
        torch.finfo(working_scores.dtype).tiny
    )
    attention_weights = torch.where(
        has_candidates, attention_weights, torch.zeros_like(attention_weights)
    )
    return attention_weights.to(original_dtype)
