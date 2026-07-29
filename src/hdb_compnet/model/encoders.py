"""Global and retrieval input encoders."""

from __future__ import annotations

import torch
from torch import nn

from hdb_compnet.config import HDBCompNetConfig
from hdb_compnet.model.blocks import CategoricalEmbeddingBlock, MLPBlock


class GlobalPropertyEncoder(nn.Module):
    """Encode all target-property numeric features and category embeddings."""

    def __init__(self, config: HDBCompNetConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            MLPBlock(config.global_input_dim, config.global_hidden_dim, config.dropout),
            MLPBlock(
                config.global_hidden_dim,
                config.global_embedding_dim,
                config.dropout,
            ),
        )

    def forward(
        self,
        numeric_features: torch.Tensor,
        categorical_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(torch.cat([numeric_features, categorical_embeddings], dim=-1))


class BaselineRegressionBranch(nn.Module):
    """Baseline scaled-log-price regressor."""

    def __init__(self, config: HDBCompNetConfig):
        super().__init__()
        self.regression_head = nn.Sequential(
            MLPBlock(
                config.global_embedding_dim,
                config.global_embedding_dim // 2,
                config.dropout,
            ),
            nn.Linear(config.global_embedding_dim // 2, 1),
        )

    def forward(self, global_representation: torch.Tensor) -> torch.Tensor:
        return self.regression_head(global_representation).squeeze(-1)


class RetrievalInputAdapter(nn.Module):
    """Select one retrieval space from full target and candidate features."""

    def __init__(self, config: HDBCompNetConfig, space_name: str):
        super().__init__()
        self.categorical_indices = config.retrieval_categorical_indices[space_name]
        self.register_buffer(
            'numeric_indices',
            torch.tensor(config.retrieval_numeric_indices[space_name], dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        target_numeric: torch.Tensor,
        target_categorical: torch.Tensor,
        candidate_numeric: torch.Tensor,
        candidate_categorical: torch.Tensor,
        categorical_embedding_block: CategoricalEmbeddingBlock,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_target_numeric = torch.index_select(
            target_numeric, dim=-1, index=self.numeric_indices
        )
        selected_candidate_numeric = torch.index_select(
            candidate_numeric, dim=-1, index=self.numeric_indices
        )
        target_embeddings = categorical_embedding_block(
            target_categorical, self.categorical_indices
        )
        candidate_embeddings = categorical_embedding_block(
            candidate_categorical, self.categorical_indices
        )
        return (
            torch.cat([selected_target_numeric, target_embeddings], dim=-1),
            torch.cat([selected_candidate_numeric, candidate_embeddings], dim=-1),
        )


class RetrievalInputPreparer(nn.Module):
    """Prepare all retrieval branch inputs with shared category embeddings."""

    def __init__(self, config: HDBCompNetConfig):
        super().__init__()
        self.adapters = nn.ModuleDict(
            {
                space_name: RetrievalInputAdapter(config, space_name)
                for space_name in config.retrieval_space_names
            }
        )

    def forward(
        self,
        batch: dict[str, object],
        categorical_embedding_block: CategoricalEmbeddingBlock,
    ) -> dict[str, dict[str, torch.Tensor]]:
        prepared: dict[str, dict[str, torch.Tensor]] = {}
        candidates = batch['candidates']
        assert isinstance(candidates, dict)
        for space_name, adapter in self.adapters.items():
            candidate_data = candidates[space_name]
            target_features, candidate_features = adapter(
                target_numeric=batch['numeric'],
                target_categorical=batch['categorical'],
                candidate_numeric=candidate_data['numeric'],
                candidate_categorical=candidate_data['categorical'],
                categorical_embedding_block=categorical_embedding_block,
            )
            prepared[space_name] = {
                'target_features': target_features,
                'candidate_features': candidate_features,
                'candidate_targets': candidate_data['targets'],
                'distances': candidate_data['distances'],
                'time_gaps': candidate_data['time_gaps'],
                'masks': candidate_data['masks'],
                'counts': candidate_data['counts'],
            }
        return prepared


class RetrievalRepresentationEncoders(nn.Module):
    """Independent target and shared-per-candidate encoders for one space."""

    def __init__(self, config: HDBCompNetConfig, space_name: str):
        super().__init__()
        input_dim = config.retrieval_input_dim(space_name)
        self.target_encoder = nn.Sequential(
            MLPBlock(input_dim, config.retrieval_hidden_dim, config.dropout),
            MLPBlock(
                config.retrieval_hidden_dim,
                config.retrieval_embedding_dim,
                config.dropout,
            ),
        )
        self.candidate_encoder = nn.Sequential(
            MLPBlock(input_dim, config.retrieval_hidden_dim, config.dropout),
            MLPBlock(
                config.retrieval_hidden_dim,
                config.retrieval_embedding_dim,
                config.dropout,
            ),
        )

    def forward(
        self,
        target_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.target_encoder(target_features),
            self.candidate_encoder(candidate_features),
        )
