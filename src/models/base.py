"""Shared layers and utilities for RTB prediction models (Bid→Win→Click).

Imported by esmm_wc.py (2-tower) and escm2_wc.py (3-tower with DR/IPW).

Contains:
- Section 1: Configurations (MLPConfig)
- Section 2: Base Layers (MLP, EmbeddingLayer, FeatureInteraction, DenseLayer, MultiHeadAttention)
- Section 3: Loss Utilities (binary_cross_entropy, counterfactual_risk, ESS diagnostics)

Reference: Wang et al., "ESCM²: Entire Space Counterfactual Multi-Task Model
for Post-Click Conversion Rate Estimation" (SIGIR 2022)
"""

from typing import NamedTuple, Sequence, Optional, Dict, List
import jax
import jax.numpy as jnp
from flax import nnx


# =============================================================================
# Section 1: Configurations (NamedTuple)
# =============================================================================


class MLPConfig(NamedTuple):
    """MLP Configuration"""
    hidden_dims: Sequence[int] = (128, 64)
    output_dim: int = 1
    dropout: float = 0.3
    activation: str = "relu"
    use_layer_norm: bool = False


# =============================================================================
# Section 2: Base Layers
# =============================================================================


class MLP(nnx.Module):
    """Multi-Layer Perceptron

    Simple feedforward network with dropout and configurable activation.
    Optionally applies LayerNorm after each hidden layer.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int],
        output_dim: int = 1,
        dropout: float = 0.3,
        activation: str = "relu",
        input_dim: Optional[int] = None,
        use_layer_norm: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            hidden_dims: Sequence of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu', 'silu')
            input_dim: Input dimension (required for first layer)
            use_layer_norm: If True, apply LayerNorm after each hidden layer
            rngs: Random number generators
        """
        layers = []
        dropouts = []
        layer_norms = []

        # Hidden layers
        for i, dim in enumerate(hidden_dims):
            if i == 0:
                if input_dim is not None:
                    in_features = input_dim
                else:
                    in_features = hidden_dims[0]
            else:
                in_features = hidden_dims[i - 1]

            layers.append(nnx.Linear(
                in_features=in_features,
                out_features=dim,
                rngs=rngs,
            ))
            dropouts.append(nnx.Dropout(rate=dropout, rngs=rngs))
            if use_layer_norm:
                layer_norms.append(nnx.LayerNorm(dim, rngs=rngs))

        # Use nnx.List for module containers (Flax 0.12.0+)
        self.layers = nnx.List(layers)
        self.dropouts = nnx.List(dropouts)
        self.layer_norms = nnx.List(layer_norms) if use_layer_norm else None
        self.use_layer_norm = use_layer_norm

        # Output layer
        self.output_layer = nnx.Linear(
            in_features=hidden_dims[-1] if hidden_dims else 1,
            out_features=output_dim,
            rngs=rngs,
        )

        self.activation = activation

    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Forward pass

        Args:
            x: Input tensor [batch, features]
            training: Whether in training mode (for dropout)

        Returns:
            Output tensor [batch, output_dim]
        """
        for i, (layer, dropout) in enumerate(zip(self.layers, self.dropouts)):
            x = layer(x)
            if self.use_layer_norm and self.layer_norms is not None:
                x = self.layer_norms[i](x)
            x = self._apply_activation(x)
            if training:
                x = dropout(x)

        return self.output_layer(x)

    def _apply_activation(self, x: jax.Array) -> jax.Array:
        if self.activation == "relu":
            return jax.nn.relu(x)
        elif self.activation == "gelu":
            return jax.nn.gelu(x)
        elif self.activation == "silu":
            return jax.nn.silu(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


class EmbeddingLayer(nnx.Module):
    """Embedding Layer for Sparse and Dense Features

    Handles:
    - Sparse (categorical) features: embedding lookup
    - Dense (numerical) features: linear projection
    - Multi-hot features (e.g., user tags): sum of embeddings
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        embed_dim: int,
        multi_hot_features: Optional[List[str]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            feature_dims: Dict mapping feature name to vocabulary size
                         Use vocab_size=-1 or 1 for dense/numerical features
                         Use vocab_size>1 for sparse/categorical features
            embed_dim: Embedding dimension for each feature
            multi_hot_features: List of features that are multi-hot encoded
            rngs: Random number generators
        """
        self.sparse_features = []
        self.dense_features = []
        self.multi_hot_features = multi_hot_features or []

        embeddings = {}
        dense_projections = {}

        for name, vocab_size in feature_dims.items():
            if vocab_size > 1:
                # Sparse feature: use embedding lookup
                self.sparse_features.append(name)
                embeddings[name] = nnx.Embed(
                    num_embeddings=vocab_size,
                    features=embed_dim,
                    rngs=rngs,
                )
            else:
                # Dense feature (vocab_size <= 1 or -1): use linear projection
                self.dense_features.append(name)
                dense_projections[name] = nnx.Linear(
                    in_features=1,
                    out_features=embed_dim,
                    rngs=rngs,
                )

        # Use nnx.Dict for module containers (Flax 0.12.0+)
        self.embeddings = nnx.Dict(embeddings)
        self.dense_projections = nnx.Dict(dense_projections)

        self.embed_dim = embed_dim
        self.n_features = len(feature_dims)

    def __call__(self, x: Dict[str, jax.Array]) -> jax.Array:
        """Forward pass

        Args:
            x: Dict mapping feature name to values
               - Sparse features: integer indices [batch,]
               - Dense features: float values [batch,]
               - Multi-hot features: [batch, max_tags] with padding

        Returns:
            Concatenated embeddings [batch, n_features * embed_dim]
        """
        outputs = []

        # Process sparse features (embedding lookup)
        for name in self.sparse_features:
            if name not in x:
                continue

            if name in self.multi_hot_features:
                # Multi-hot: sum of embeddings
                # x[name] shape: [batch, max_tags]
                max_idx = self.embeddings[name].num_embeddings - 1
                safe_idx = jnp.clip(x[name], 0, max_idx)
                emb = self.embeddings[name](safe_idx)  # [batch, max_tags, embed_dim]
                # Sum over tags dimension
                outputs.append(emb.sum(axis=-2))  # [batch, embed_dim]
            else:
                # Single-hot: simple lookup
                max_idx = self.embeddings[name].num_embeddings - 1
                safe_idx = jnp.clip(x[name], 0, max_idx)
                outputs.append(self.embeddings[name](safe_idx))

        # Process dense features (linear projection)
        for name in self.dense_features:
            if name in x:
                # Expand dims for linear layer: [batch,] -> [batch, 1]
                dense_input = x[name][..., jnp.newaxis].astype(jnp.float32)
                outputs.append(self.dense_projections[name](dense_input))

        # Concatenate all outputs
        return jnp.concatenate(outputs, axis=-1)


class DenseLayer(nnx.Module):
    """Dense Feature Processing Layer

    Processes continuous/dense features with normalization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            input_dim: Number of input features
            output_dim: Output dimension
            rngs: Random number generators
        """
        self.linear = nnx.Linear(input_dim, output_dim, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(output_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass"""
        x = self.linear(x)
        x = self.layer_norm(x)
        return jax.nn.relu(x)


class FeatureInteraction(nnx.Module):
    """Feature Interaction Layer (FM-style)

    Computes pairwise feature interactions using the FM trick:
    (Σx)² - Σx² gives sum of all pairwise dot products
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            n_features: Number of features
            embed_dim: Embedding dimension
            rngs: Random number generators
        """
        self.n_features = n_features
        self.embed_dim = embed_dim

    def __call__(self, embeddings: jax.Array) -> jax.Array:
        """Forward pass

        Args:
            embeddings: [batch, n_features, embed_dim]

        Returns:
            Interaction output [batch, embed_dim]
        """
        # FM-style interaction: 0.5 * ((Σx)² - Σx²)
        sum_of_squares = jnp.sum(embeddings ** 2, axis=-2)
        square_of_sum = jnp.sum(embeddings, axis=-2) ** 2

        return 0.5 * (square_of_sum - sum_of_squares)


class MultiHeadAttention(nnx.Module):
    """Multi-Head Self-Attention for feature interaction.

    Can be used as an alternative to FM for capturing complex interactions.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            rngs: Random number generators
        """
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.q_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.k_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.out_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        training: bool = True,
    ) -> jax.Array:
        """Forward pass

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            training: Whether in training mode

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: [batch, heads, seq, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        if training:
            attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        # Reshape back
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch_size, seq_len, self.embed_dim)

        return self.out_proj(output)


# =============================================================================
# Section 3: Loss Utilities
# =============================================================================


def binary_cross_entropy(
    y_pred: jax.Array,
    y_true: jax.Array,
    eps: float = 1e-7,
) -> jax.Array:
    """Binary Cross Entropy for probability inputs (NOT logits)

    Use this when y_pred is already a probability (after sigmoid).
    For logits, use optax.sigmoid_binary_cross_entropy instead.

    Args:
        y_pred: Predicted probabilities (0-1)
        y_true: Ground truth labels (0 or 1)
        eps: Small value for numerical stability

    Returns:
        BCE loss per sample
    """
    y_pred = jnp.clip(y_pred, eps, 1.0 - eps)
    return -(y_true * jnp.log(y_pred) + (1.0 - y_true) * jnp.log(1.0 - y_pred))


def counterfactual_risk(
    y_impute: jax.Array,
    selection: jax.Array,
    cfr_type: str = "consistency",
) -> jax.Array:
    """Counterfactual Risk Minimization

    Regularization term for ESCM² that constrains imputation on unselected samples.

    Args:
        y_impute: Imputed error δ̂
        selection: Selection indicator (1 if selected)
        cfr_type: Type of CFR ('consistency', 'shrink', or 'none')

    Returns:
        CFR loss
    """
    if cfr_type == "none":
        return jnp.array(0.0)

    # On unselected samples, δ̂ should be ~0 (no observed error)
    unselected = 1 - selection
    return jnp.mean(unselected * y_impute ** 2)


def compute_effective_sample_size(weights: jax.Array) -> float:
    """Compute Effective Sample Size (ESS) for IPW weights.

    ESS = (Σw)² / Σw²

    Lower ESS indicates higher weight variance and less reliable estimates.

    Args:
        weights: IPW weights

    Returns:
        Effective sample size
    """
    sum_weights = jnp.sum(weights)
    sum_sq_weights = jnp.sum(weights ** 2)
    ess = sum_weights ** 2 / jnp.clip(sum_sq_weights, 1e-8, None)
    return float(ess)


def compute_weight_statistics(weights: jax.Array) -> dict:
    """Compute statistics for IPW weights.

    Args:
        weights: IPW weights

    Returns:
        Dictionary with weight statistics
    """
    n = weights.shape[0]
    ess = compute_effective_sample_size(weights)

    return {
        "n_samples": int(n),
        "ess": float(ess),
        "ess_ratio": float(ess / n) if n > 0 else 0.0,
        "weight_mean": float(jnp.mean(weights)),
        "weight_std": float(jnp.std(weights)),
        "weight_max": float(jnp.max(weights)),
        "weight_min": float(jnp.min(weights)),
    }
