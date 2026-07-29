"""Model output collection and fixed-candidate SHAP analysis."""

from hdb_compnet.interpretability.gating_analysis import (
    collect_hdb_compnet_interpretation_outputs,
)
from hdb_compnet.interpretability.shap_analysis import (
    calculate_target_shap_analysis,
)

__all__ = [
    'calculate_target_shap_analysis',
    'collect_hdb_compnet_interpretation_outputs',
]
