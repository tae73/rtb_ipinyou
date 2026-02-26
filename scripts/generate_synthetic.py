#!/usr/bin/env python
"""Generate synthetic RTB data for model architecture sanity testing.

Produces Parquet + metadata JSON in the same format as the real pipeline,
with learnable win/click signals targeting AUC ≥ 0.8.

Usage:
    python scripts/generate_synthetic.py generate \
        --output-dir data/ipinyou/prediction/synthetic \
        --train-size 500000 \
        --val-size 100000 \
        --test-size 100000 \
        --seed 42
"""

from pathlib import Path
from typing import Dict, NamedTuple, Tuple
import json
import sys

import typer
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

app = typer.Typer(
    name="generate_synthetic",
    help="Generate synthetic RTB data for model sanity testing",
    add_completion=False,
)


# =============================================================================
# Feature specification (matching real data exactly)
# =============================================================================

# Categorical features: name → vocab_size (max index = vocab_size - 1)
CATEGORICAL_SPEC: Dict[str, int] = {
    "region": 35,
    "city": 360,
    "adexchange": 5,
    "slotwidth": 20,
    "slotheight": 15,
    "slotvisibility": 5,
    "slotformat": 5,
    "advertiser": 9,
    "hour": 24,
    "minute": 60,
    "weekday": 7,
    "is_weekend": 2,
    "is_peak_hour": 2,
    "slot_size_group": 6,
    "region_group": 4,
    "domain_hash": 10000,
    "creative_hash": 5000,
}

# Numerical features generated independently (others are derived)
INDEPENDENT_NUMERICAL: Dict[str, Dict[str, float]] = {
    "slotprice": {"mean": 47.0, "std": 41.0, "clip_min": 0.0},
    "bidprice": {"mean": 272.0, "std": 30.0, "clip_min": 50.0},
    "region_freq": {"mean": 8_000_000.0, "std": 7_000_000.0, "clip_min": 1000.0},
    "domain_freq": {"mean": 2_500_000.0, "std": 3_300_000.0, "clip_min": 100.0},
    "creative_freq": {"mean": 6_500_000.0, "std": 5_500_000.0, "clip_min": 100.0},
}

# Ordered feature lists (matching real metadata)
CATEGORICAL_FEATURES = [
    "region", "city", "adexchange", "slotwidth", "slotheight",
    "slotvisibility", "slotformat", "advertiser", "hour", "minute",
    "weekday", "is_weekend", "is_peak_hour", "slot_size_group",
    "region_group", "domain_hash", "creative_hash",
]

NUMERICAL_FEATURES = [
    "slotprice", "bidprice", "hour_sin", "hour_cos",
    "slot_area", "slot_area_log", "slot_aspect_ratio",
    "region_freq", "bid_floor_ratio",
    "domain_freq", "domain_freq_log",
    "creative_freq", "creative_freq_log",
]

# Typical pixel dimensions for slot area derivation
_SLOT_WIDTHS = np.array([
    300, 728, 970, 160, 320, 468, 250, 200, 336, 120,
    300, 300, 728, 970, 160, 320, 468, 250, 200, 336,
])
_SLOT_HEIGHTS = np.array([
    250, 90, 90, 600, 50, 60, 250, 200, 280, 600,
    600, 50, 90, 250, 600, 100, 60, 250, 200, 280,
])


# =============================================================================
# Signal configuration
# =============================================================================

class SignalConfig(NamedTuple):
    """Parameters controlling label generation signals."""
    # Win model: logit(win) = intercept + β·features + noise
    win_intercept: float = -1.1       # logit(0.25) ≈ -1.1
    win_bidprice_coef: float = 1.5    # Higher bid → more wins
    win_slotprice_coef: float = -1.0  # Higher floor → fewer wins
    win_noise_std: float = 1.0
    win_exchange_effects: Tuple[float, ...] = (-0.3, 0.0, 0.2, -0.1, 0.4)

    # Click model: logit(click|win=1) = intercept + γ·features + noise
    # ORTHOGONAL to win signal: uses features NOT in win model
    # (win uses bidprice, slotprice, adexchange)
    click_intercept: float = -1.5     # ~18% click rate among winners
    click_noise_std: float = 0.3      # Low noise for clear signal
    # Continuous numerical features (orthogonal to win signal)
    click_area_coef: float = 1.2      # slot_area: larger → more visible → more clicks
    click_region_freq_coef: float = 0.8  # region_freq: popular regions → more clicks
    click_hour_sin_coef: float = 0.6  # Cyclical hour pattern
    click_hour_cos_coef: float = 0.4
    # Categorical features (orthogonal to win signal)
    click_format_effects: Tuple[float, ...] = (-0.6, 0.3, 1.0, -0.4, 0.0)
    click_adv_effects: Tuple[float, ...] = (
        -0.8, 0.5, 1.2, -0.4, 0.2, 0.9, -0.5, 0.4, -0.2,
    )


# =============================================================================
# Generation logic
# =============================================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def generate_split(
    n: int,
    rng: np.random.Generator,
    signal_cfg: SignalConfig = SignalConfig(),
) -> pd.DataFrame:
    """Generate one data split with learnable win/click signals.

    Args:
        n: Number of samples
        rng: NumPy random generator
        signal_cfg: Label generation parameters

    Returns:
        DataFrame with 30 features + win/click labels
    """
    data: Dict[str, np.ndarray] = {}

    # --- Categorical features (uniform random within vocab) ---
    for feat, vocab in CATEGORICAL_SPEC.items():
        data[feat] = rng.integers(0, vocab, size=n, dtype=np.int64)

    # Derive is_weekend / is_peak_hour from hour / weekday
    data["is_weekend"] = (data["weekday"] >= 5).astype(np.int64)
    hour = data["hour"]
    data["is_peak_hour"] = (
        ((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 20))
    ).astype(np.int64)

    # --- Independent numerical features ---
    for feat, spec in INDEPENDENT_NUMERICAL.items():
        values = rng.normal(spec["mean"], spec["std"], size=n)
        data[feat] = np.clip(values, spec["clip_min"], None)

    # --- Derived numerical features ---
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Slot area from lookup tables (real pixel dimensions)
    w_idx = data["slotwidth"] % len(_SLOT_WIDTHS)
    h_idx = data["slotheight"] % len(_SLOT_HEIGHTS)
    data["slot_area"] = (_SLOT_WIDTHS[w_idx] * _SLOT_HEIGHTS[h_idx]).astype(np.float64)
    data["slot_area_log"] = np.log1p(data["slot_area"])
    data["slot_aspect_ratio"] = _SLOT_WIDTHS[w_idx] / np.maximum(_SLOT_HEIGHTS[h_idx], 1)

    data["bid_floor_ratio"] = np.clip(
        data["bidprice"] / np.maximum(data["slotprice"], 1.0), 0, 100,
    )
    data["domain_freq_log"] = np.log1p(data["domain_freq"])
    data["creative_freq_log"] = np.log1p(data["creative_freq"])

    # --- Win labels ---
    bp_z = (data["bidprice"] - data["bidprice"].mean()) / max(data["bidprice"].std(), 1e-8)
    sp_z = (data["slotprice"] - data["slotprice"].mean()) / max(data["slotprice"].std(), 1e-8)

    exchange_effects = np.array(signal_cfg.win_exchange_effects)
    ex_effect = exchange_effects[data["adexchange"] % len(exchange_effects)]

    win_logit = (
        signal_cfg.win_intercept
        + signal_cfg.win_bidprice_coef * bp_z
        + signal_cfg.win_slotprice_coef * sp_z
        + ex_effect
        + rng.normal(0, signal_cfg.win_noise_std, size=n)
    )
    data["win"] = rng.binomial(1, _sigmoid(win_logit)).astype(np.int64)

    # --- Click labels (only possible when win=1) ---
    # Continuous numerical signals ORTHOGONAL to win features
    area_z = (data["slot_area"] - data["slot_area"].mean()) / max(data["slot_area"].std(), 1e-8)
    rfreq_z = (data["region_freq"] - data["region_freq"].mean()) / max(data["region_freq"].std(), 1e-8)
    numerical_signal = (
        signal_cfg.click_area_coef * area_z
        + signal_cfg.click_region_freq_coef * rfreq_z
        + signal_cfg.click_hour_sin_coef * data["hour_sin"]
        + signal_cfg.click_hour_cos_coef * data["hour_cos"]
    )

    # Categorical signals (via embeddings)
    format_effects = np.array(signal_cfg.click_format_effects)
    fmt_effect = format_effects[data["slotformat"] % len(format_effects)]

    adv_effects = np.array(signal_cfg.click_adv_effects)
    adv_effect = adv_effects[data["advertiser"] % len(adv_effects)]

    click_logit = (
        signal_cfg.click_intercept
        + numerical_signal
        + fmt_effect
        + adv_effect
        + rng.normal(0, signal_cfg.click_noise_std, size=n)
    )
    data["click"] = np.where(
        data["win"] == 1,
        rng.binomial(1, _sigmoid(click_logit)),
        0,
    ).astype(np.int64)

    # Build DataFrame with correct column order
    columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + ["win", "click"]
    return pd.DataFrame({col: data[col] for col in columns})


def _compute_normalization_stats(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Compute Z-score stats from training split (no data leakage)."""
    mean_dict, std_dict = {}, {}
    for col in NUMERICAL_FEATURES:
        values = df[col].dropna()
        mean_dict[col] = float(values.mean())
        std_dict[col] = max(float(values.std(ddof=0)), 1e-8)
    return {"mean": mean_dict, "std": std_dict}


# =============================================================================
# CLI
# =============================================================================

@app.command()
def generate(
    output_dir: Path = typer.Option(
        "data/ipinyou/prediction/synthetic",
        "--output-dir",
        "-o",
        help="Output directory for synthetic data",
    ),
    train_size: int = typer.Option(
        500_000, "--train-size", help="Number of training samples",
    ),
    val_size: int = typer.Option(
        100_000, "--val-size", help="Number of validation samples",
    ),
    test_size: int = typer.Option(
        100_000, "--test-size", help="Number of test samples",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Generate synthetic RTB data with learnable win/click signals."""
    rng = np.random.default_rng(seed)
    signal_cfg = SignalConfig()

    typer.echo(f"Generating synthetic data (seed={seed})...")
    typer.echo(f"  Train: {train_size:,}")
    typer.echo(f"  Val:   {val_size:,}")
    typer.echo(f"  Test:  {test_size:,}")

    # Generate splits
    train_df = generate_split(train_size, rng, signal_cfg)
    val_df = generate_split(val_size, rng, signal_cfg)
    test_df = generate_split(test_size, rng, signal_cfg)

    # Report statistics
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        win_rate = df["win"].mean()
        click_rate_won = (
            df.loc[df["win"] == 1, "click"].mean() if df["win"].sum() > 0 else 0.0
        )
        typer.echo(
            f"  {name}: win_rate={win_rate:.3f}, "
            f"click_rate(won)={click_rate_won:.3f}"
        )

    # Compute normalization stats from training set only
    norm_stats = _compute_normalization_stats(train_df)

    # Build metadata (same structure as real feature_metadata.json)
    all_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + ["win", "click"]
    metadata = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "columns": all_columns,
        "feature_info": {
            "categorical": CATEGORICAL_FEATURES,
            "numerical": NUMERICAL_FEATURES,
            "total_count": len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES),
            "feature_groups": {
                "time": [
                    "hour", "minute", "weekday", "is_weekend",
                    "is_peak_hour", "hour_sin", "hour_cos",
                ],
                "slot": [
                    "slot_area", "slot_area_log", "slot_aspect_ratio",
                    "slot_size_group", "slotvisibility", "slotformat",
                ],
                "region": ["region", "city", "region_freq", "region_group"],
                "competition": ["bidprice", "slotprice", "bid_floor_ratio"],
                "id": ["adexchange", "advertiser"],
                "high_cardinality": [
                    "domain_hash", "creative_hash", "domain_freq",
                    "domain_freq_log", "creative_freq", "creative_freq_log",
                ],
            },
        },
        "normalization_stats": norm_stats,
    }

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    with open(output_dir / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    typer.echo(f"\nSaved to: {output_dir}")
    typer.echo(f"  train.parquet: {train_size:,} rows × {len(all_columns)} cols")
    typer.echo(f"  val.parquet:   {val_size:,} rows × {len(all_columns)} cols")
    typer.echo(f"  test.parquet:  {test_size:,} rows × {len(all_columns)} cols")
    typer.echo("  feature_metadata.json")


if __name__ == "__main__":
    app()
