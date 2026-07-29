"""Full four-branch HDB-CompNet."""

from __future__ import annotations

from torch import nn

from hdb_compnet.config import HDBCompNetConfig
from hdb_compnet.model.blocks import CategoricalEmbeddingBlock
from hdb_compnet.model.encoders import (
    BaselineRegressionBranch,
    GlobalPropertyEncoder,
    RetrievalInputPreparer,
)
from hdb_compnet.model.gating import GatingNetwork, combine_branch_predictions
from hdb_compnet.model.retrieval import RetrievalBranch


class HDBCompNet(nn.Module):
    """Notebook-equivalent baseline plus three retrieval branches and gating."""

    def __init__(self, config: HDBCompNetConfig):
        super().__init__()
        self.config = config
        self.categorical_embedding_block = CategoricalEmbeddingBlock(config)
        self.global_encoder = GlobalPropertyEncoder(config)
        self.baseline_branch = BaselineRegressionBranch(config)
        self.retrieval_input_preparer = RetrievalInputPreparer(config)
        self.retrieval_branches = nn.ModuleDict(
            {
                space_name: RetrievalBranch(config, space_name)
                for space_name in config.retrieval_space_names
            }
        )
        self.gating_network = GatingNetwork(config)

    def forward(self, batch: dict[str, object]) -> dict[str, object]:
        categorical_embeddings = self.categorical_embedding_block(batch['categorical'])
        global_representation = self.global_encoder(batch['numeric'], categorical_embeddings)
        baseline_prediction = self.baseline_branch(global_representation)
        retrieval_inputs = self.retrieval_input_preparer(
            batch, self.categorical_embedding_block
        )
        retrieval_outputs = {
            space_name: self.retrieval_branches[space_name](retrieval_inputs[space_name])
            for space_name in self.config.retrieval_space_names
        }
        branch_predictions = {
            'baseline': baseline_prediction,
            'structural': retrieval_outputs['structural']['prediction'],
            'spatial': retrieval_outputs['spatial']['prediction'],
            'market': retrieval_outputs['market']['prediction'],
        }
        gating_output = self.gating_network(
            global_representation, branch_predictions, retrieval_outputs
        )
        final_prediction, stacked, weighted = combine_branch_predictions(
            branch_predictions,
            gating_output['weights'],
            self.config.branch_names,
        )
        return {
            'prediction': final_prediction,
            'branch_predictions': branch_predictions,
            'stacked_branch_predictions': stacked,
            'weighted_branch_contributions': weighted,
            'gate_weights': gating_output['weights'],
            'gate_logits': gating_output['logits'],
            'branch_availability': gating_output['availability'],
            'gate_features': gating_output['features'],
            'global_representation': global_representation,
            'retrieval_outputs': retrieval_outputs,
        }
