"""
ESCM²-WC: Entire Space Counterfactual Multi-Task Model for Win→Click (3-tower)

Primary model for Bid→Win→Click debiasing. Extends ESMM-WC with DR/IPW
debiasing of CTR using win propensity from the Win Tower.

Architecture:
- Win Tower: P(Win|X, bid) — propensity model, trained on all 65M bids
- CTR Tower: P(Click|X, Win) — debiased via DR/IPW using Win Tower propensity
- Imputation Tower: predicts CTR error delta_hat for DR (no sigmoid)
- Joint: P(Click_bid) = P(Win) × P(Click|Win) — ESMM constraint

Structural mapping from original ESCM² (click→conversion):
  CTR Tower (propensity)     → Win Tower (propensity)
  CVR Tower (debiased)       → CTR Tower (debiased)
  Imputation Tower (DR)      → Imputation Tower (DR)
  CTCVR = pCTR × pCVR       → P(Click_bid) = P(Win) × P(Click|Win)
  Selection: click            → Selection: win
  Target: conversion          → Target: click

Win Tower dual purpose:
  (a) CTR debiasing propensity (this model)
  (b) Bid shading win rate model (downstream bidding)

Reference: Wang et al., "ESCM²" (SIGIR 2022), adapted for Bid→Win→Click.
"""

from typing import NamedTuple, Sequence, Optional, Dict, List
import jax
import jax.numpy as jnp
from flax import nnx

from .base import (
    MLP,
    EmbeddingLayer,
    FeatureInteraction,
    binary_cross_entropy,
    counterfactual_risk,
)


# =============================================================================
# Configurations
# =============================================================================


class ESCM2WCConfig(NamedTuple):
    """ESCM²-WC Configuration"""
    feature_dims: Dict[str, int]  # Feature name -> vocab size
    embed_dim: int = 32
    hidden_dims: Sequence[int] = (128, 64)
    win_hidden_dims: Sequence[int] = (64, 32)
    dropout: float = 0.3
    # Debiasing config
    loss_type: str = "dr"  # 'ipw' or 'dr'
    dr_loss_type: str = "mse"  # 'mse' or 'bce' (for DR variant)
    win_eps: float = 0.05  # Win propensity clipping
    max_weight: float = 10.0
    normalize_weights: bool = True  # Self-normalized IPW
    # Counterfactual regularization
    cfr_lambda: float = 0.1
    cfr_type: str = "consistency"  # 'consistency', 'shrink', or 'none'
    # Loss weights
    win_weight: float = 1.0
    ctr_weight: float = 1.0
    joint_weight: float = 1.0
    impute_loss_weight: float = 0.5
    # Imputation loss config
    impute_loss_type: str = "mse"  # 'mse' or 'huber'
    impute_huber_delta: float = 0.1  # Huber loss delta (only used when impute_loss_type='huber')
    # Architecture options
    use_fm_interaction: bool = True
    use_layer_norm: bool = False
    use_numeric_bypass: bool = False
    stop_grad_win_embedding: bool = False  # Stop gradient from win tower to shared embedding
    multi_hot_features: Optional[Sequence[str]] = None
    impute_hidden_dims: Optional[Sequence[int]] = None  # Imputation tower dims (None → use hidden_dims)
    # Per-tower dropout (None → use global dropout)
    win_dropout: Optional[float] = None
    ctr_dropout: Optional[float] = None
    impute_dropout: Optional[float] = None


class ESCM2WCOutput(NamedTuple):
    """ESCM²-WC Output"""
    p_win: jax.Array        # P(Win|X, bid)
    p_ctr: jax.Array        # P(Click|X, Win)
    p_click_bid: jax.Array  # P(Win) × P(Click|Win)
    y_impute: jax.Array     # Imputed CTR error delta_hat (no sigmoid)


class ESCM2WCLossComponents(NamedTuple):
    """ESCM²-WC Loss Components"""
    total: float
    win: float      # Win BCE (all bids)
    ctr: float      # CTR DR/IPW loss
    joint: float    # Joint BCE (all bids, ESMM constraint)
    cfr: float      # Counterfactual risk
    impute: float   # Imputation tower loss (won only)


# =============================================================================
# Model
# =============================================================================


class ESCM2WC(nnx.Module):
    """Entire Space Counterfactual Multi-Task Model for Win→Click

    3-tower architecture with DR/IPW debiasing of CTR using Win Tower
    propensity. The Win Tower serves dual purpose: (a) provides propensity
    scores for DR/IPW debiasing of CTR, (b) models win rate for bid shading.
    """

    def __init__(self, config: ESCM2WCConfig, *, rngs: nnx.Rngs):
        self.config = config

        # Shared embedding
        multi_hot = list(config.multi_hot_features) if config.multi_hot_features else None
        self.embedding = EmbeddingLayer(
            feature_dims=config.feature_dims,
            embed_dim=config.embed_dim,
            multi_hot_features=multi_hot,
            use_numeric_bypass=config.use_numeric_bypass,
            rngs=rngs,
        )

        # Count feature types for input_dim calculation
        n_cat = sum(1 for v in config.feature_dims.values() if v > 1)
        n_num = sum(1 for v in config.feature_dims.values() if v <= 1)
        self._n_cat = n_cat

        # Embedding output dim depends on bypass mode
        if config.use_numeric_bypass:
            embed_output_dim = n_cat * config.embed_dim + n_num
        else:
            embed_output_dim = len(config.feature_dims) * config.embed_dim

        # Optional FM interaction (bypass: categorical embeddings only)
        self.use_fm_interaction = config.use_fm_interaction
        if config.use_fm_interaction:
            self.interaction = FeatureInteraction(
                n_features=n_cat if config.use_numeric_bypass else len(config.feature_dims),
                embed_dim=config.embed_dim,
                rngs=rngs,
            )
            input_dim = embed_output_dim + config.embed_dim
        else:
            self.interaction = None
            input_dim = embed_output_dim

        # Win Tower: P(Win|X, bid)
        self.win_tower = MLP(
            hidden_dims=config.win_hidden_dims,
            output_dim=1,
            dropout=config.win_dropout if config.win_dropout is not None else config.dropout,
            input_dim=input_dim,
            use_layer_norm=config.use_layer_norm,
            rngs=rngs,
        )

        # CTR Tower: P(Click|X, Win)
        self.ctr_tower = MLP(
            hidden_dims=config.hidden_dims,
            output_dim=1,
            dropout=config.ctr_dropout if config.ctr_dropout is not None else config.dropout,
            input_dim=input_dim,
            use_layer_norm=config.use_layer_norm,
            rngs=rngs,
        )

        # Imputation Tower: predicts CTR error delta_hat (linear, no sigmoid)
        impute_dims = config.impute_hidden_dims if config.impute_hidden_dims is not None else config.hidden_dims
        self.imputation_tower = MLP(
            hidden_dims=impute_dims,
            output_dim=1,
            dropout=config.impute_dropout if config.impute_dropout is not None else config.dropout,
            input_dim=input_dim,
            use_layer_norm=config.use_layer_norm,
            rngs=rngs,
        )

    def __call__(
        self,
        x: Dict[str, jax.Array],
        training: bool = True,
    ) -> ESCM2WCOutput:
        # Embed features
        embed = self.embedding(x)

        # Optional FM interaction
        if self.use_fm_interaction and self.interaction is not None:
            if self.config.use_numeric_bypass:
                # Split: cat embeddings | raw numerics
                cat_dim = self._n_cat * self.config.embed_dim
                cat_embed = embed[..., :cat_dim]
                num_raw = embed[..., cat_dim:]
                cat_reshaped = cat_embed.reshape(
                    *cat_embed.shape[:-1], self._n_cat, self.config.embed_dim
                )
                interaction_out = self.interaction(cat_reshaped)
                embed = jnp.concatenate([cat_embed, interaction_out, num_raw], axis=-1)
            else:
                n_features = len(self.config.feature_dims)
                embed_reshaped = embed.reshape(
                    *embed.shape[:-1], n_features, self.config.embed_dim
                )
                interaction_out = self.interaction(embed_reshaped)
                embed = jnp.concatenate([embed, interaction_out], axis=-1)

        # Win prediction (propensity)
        # Optional: stop gradient from win tower to shared embedding
        # Prevents win tower's strong supervision from dominating embedding gradients
        win_embed = jax.lax.stop_gradient(embed) if self.config.stop_grad_win_embedding else embed
        win_logit = self.win_tower(win_embed, training=training)
        p_win = jax.nn.sigmoid(win_logit).squeeze(-1)

        # CTR prediction
        ctr_logit = self.ctr_tower(embed, training=training)
        p_ctr = jax.nn.sigmoid(ctr_logit).squeeze(-1)

        # ESMM constraint
        p_click_bid = p_win * p_ctr

        # Imputation: predicts CTR error delta_hat (no sigmoid)
        impute_logit = self.imputation_tower(embed, training=training)
        y_impute = impute_logit.squeeze(-1)

        return ESCM2WCOutput(
            p_win=p_win,
            p_ctr=p_ctr,
            p_click_bid=p_click_bid,
            y_impute=y_impute,
        )


# =============================================================================
# Loss and Training Functions
# =============================================================================


def create_escm2wc_loss_fn(
    config: ESCM2WCConfig,
    return_components: bool = False,
    jit_safe: bool = False,
):
    """Create loss function for ESCM²-WC.

    Losses:
    1. Win BCE: BCE(p_win, win) on ALL bids
    2. CTR DR/IPW: debiased CTR loss using win propensity
       - DR: delta_hat + (win/P(Win)) × ((click - p_ctr) - delta_hat)
       - IPW: (win/P(Win)) × BCE(p_ctr, click)
    3. Joint BCE: BCE(p_click_bid, click) on ALL bids — ESMM constraint
    4. Imputation: supervised on WON samples (delta_target = click - p_ctr)
    5. CFR: counterfactual risk on unselected (win=0) samples

    Args:
        config: ESCM²-WC configuration.
        return_components: If True, return (total_loss, ESCM2WCLossComponents).
        jit_safe: If True, skip float() cast in components (avoids
            ConcretizationTypeError inside JIT).
    """

    def loss_fn(model: ESCM2WC, batch: dict, training: bool = True):
        output = model(batch["x"], training=training)

        # --- Win loss (all bids) ---
        win_loss = jnp.mean(
            binary_cross_entropy(output.p_win, batch["win"])
        )

        # --- Propensity weights for CTR debiasing ---
        propensity = jnp.clip(
            jax.lax.stop_gradient(output.p_win), config.win_eps, 1.0
        )
        raw_weights = batch["win"] / propensity
        weights = jnp.clip(raw_weights, 0.0, config.max_weight)

        # Self-normalized weights
        if config.normalize_weights:
            weight_sum = jnp.clip(weights.sum(), 1e-8, None)
            weights = weights / weight_sum * weights.shape[0]

        # --- CTR loss with debiasing ---
        impute_loss = 0.0

        if config.loss_type == "ipw":
            ctr_bce = binary_cross_entropy(output.p_ctr, batch["click"])
            ctr_loss_base = jnp.mean(weights * ctr_bce)
            ctr_loss = ctr_loss_base

        elif config.loss_type == "dr":
            delta_hat_sg = jax.lax.stop_gradient(output.y_impute)

            if config.dr_loss_type == "mse":
                # Paper formulation: MSE of DR error
                ctr_error = batch["click"] - output.p_ctr
                dr_error = delta_hat_sg + weights * (ctr_error - delta_hat_sg)
                ctr_loss_base = jnp.mean(jnp.square(dr_error))
            else:
                # BCE variant with pseudo-labels
                imputed_ctr = jax.lax.stop_gradient(output.p_ctr) + delta_hat_sg
                dr_estimate = imputed_ctr + weights * (batch["click"] - imputed_ctr)
                dr_target = jax.lax.stop_gradient(
                    jnp.clip(dr_estimate, 1e-7, 1.0 - 1e-7)
                )
                ctr_loss_base = jnp.mean(
                    binary_cross_entropy(output.p_ctr, dr_target)
                )

            # Imputation tower supervision (won samples only)
            p_ctr_sg = jax.lax.stop_gradient(output.p_ctr)
            delta_target = batch["click"] - p_ctr_sg
            impute_residual = output.y_impute - delta_target
            if config.impute_loss_type == "huber":
                # Huber loss: robust to outlier deltas
                delta = config.impute_huber_delta
                abs_r = jnp.abs(impute_residual)
                impute_element = jnp.where(
                    abs_r <= delta,
                    0.5 * jnp.square(impute_residual),
                    delta * (abs_r - 0.5 * delta),
                )
            else:
                # MSE (default)
                impute_element = jnp.square(impute_residual)
            n_won = jnp.clip(batch["win"].sum(), 1.0, None)
            impute_loss = jnp.sum(batch["win"] * impute_element) / n_won
            ctr_loss = ctr_loss_base + config.impute_loss_weight * impute_loss

        else:
            raise ValueError(f"Unknown loss_type: {config.loss_type}")

        # --- Joint loss (all bids, ESMM constraint) ---
        joint_loss = jnp.mean(
            binary_cross_entropy(output.p_click_bid, batch["click"])
        )

        # --- Counterfactual risk (unselected = win=0 samples) ---
        cf_risk = counterfactual_risk(
            output.y_impute, batch["win"], config.cfr_type
        )

        # --- Total loss ---
        total_loss = (
            config.win_weight * win_loss
            + config.ctr_weight * ctr_loss
            + config.joint_weight * joint_loss
            + config.cfr_lambda * cf_risk
        )

        if return_components:
            _f = (lambda x: x) if jit_safe else float
            components = ESCM2WCLossComponents(
                total=_f(total_loss),
                win=_f(win_loss),
                ctr=_f(ctr_loss_base) if config.loss_type == "dr" else _f(ctr_loss),
                joint=_f(joint_loss),
                cfr=_f(cf_risk) if config.cfr_type != "none" else 0.0,
                impute=_f(impute_loss),
            )
            return total_loss, components

        return total_loss

    return loss_fn


def create_escm2wc_train_step(config: ESCM2WCConfig):
    """Create JIT-compiled training step for ESCM²-WC.

    Returns (loss, ESCM2WCLossComponents) per step. Components use JAX arrays
    (jit_safe=True) — caller converts to Python float outside JIT.
    """
    loss_fn = create_escm2wc_loss_fn(config, return_components=True, jit_safe=True)

    @nnx.jit
    def train_step(
        model: ESCM2WC,
        optimizer: nnx.Optimizer,
        batch: dict,
    ):
        (loss, components), grads = nnx.value_and_grad(
            lambda m: loss_fn(m, batch), has_aux=True
        )(model)
        optimizer.update(model, grads)
        return loss, components

    return train_step


def create_escm2wc_eval_step():
    """Create JIT-compiled evaluation step for ESCM²-WC."""

    @nnx.jit
    def eval_step(model: ESCM2WC, batch: dict) -> ESCM2WCOutput:
        return model(batch["x"], training=False)

    return eval_step
