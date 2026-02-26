"""Model implementations for RTB prediction (Bid→Win→Click)."""

from .base import (
    # Base layers
    MLP,
    EmbeddingLayer,
    FeatureInteraction,
    DenseLayer,
    MultiHeadAttention,
    # Configurations
    MLPConfig,
    # Loss utilities
    binary_cross_entropy,
    counterfactual_risk,
    compute_effective_sample_size,
    compute_weight_statistics,
)
from .esmm_wc import (
    ESMMWCConfig,
    ESMMWCOutput,
    ESMMWCLossComponents,
    ESMMWC,
    create_esmm_wc_loss_fn,
    create_esmm_wc_train_step,
    create_esmm_wc_eval_step,
)
from .escm2_wc import (
    ESCM2WCConfig,
    ESCM2WCOutput,
    ESCM2WCLossComponents,
    ESCM2WC,
    create_escm2wc_loss_fn,
    create_escm2wc_train_step,
    create_escm2wc_eval_step,
)

__all__ = [
    # Base layers
    "MLP",
    "EmbeddingLayer",
    "FeatureInteraction",
    "DenseLayer",
    "MultiHeadAttention",
    # Configurations
    "MLPConfig",
    # Loss utilities
    "binary_cross_entropy",
    "counterfactual_risk",
    "compute_effective_sample_size",
    "compute_weight_statistics",
    # ESMM-WC (Bid→Win→Click, 2-tower)
    "ESMMWCConfig",
    "ESMMWCOutput",
    "ESMMWCLossComponents",
    "ESMMWC",
    "create_esmm_wc_loss_fn",
    "create_esmm_wc_train_step",
    "create_esmm_wc_eval_step",
    # ESCM2-WC (Bid→Win→Click, 3-tower with DR)
    "ESCM2WCConfig",
    "ESCM2WCOutput",
    "ESCM2WCLossComponents",
    "ESCM2WC",
    "create_escm2wc_loss_fn",
    "create_escm2wc_train_step",
    "create_escm2wc_eval_step",
]
