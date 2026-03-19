#!/usr/bin/env python
"""Train prediction models for RTB (Bid→Win→Click).

Supports baseline LightGBM, ESMM-WC (2-tower), and ESCM2-WC (3-tower with DR/IPW).

Usage:
    # Train baseline LightGBM for CTR prediction (biased, winners only)
    python scripts/train.py baseline \
        --data-dir data/ipinyou/prediction/features \
        --model-dir results/models \
        --task ctr

    # Train baseline for Win prediction (all bids)
    python scripts/train.py baseline \
        --data-dir data/ipinyou/prediction/features \
        --model-dir results/models \
        --task win

    # Train ESMM-WC (2-tower, ESMM constraint only)
    python scripts/train.py esmmwc \
        --data-dir data/ipinyou/prediction/features \
        --model-dir results/models

    # Train ESCM2-WC with DR debiasing (primary model)
    python scripts/train.py escm2wc \
        --data-dir data/ipinyou/prediction/features \
        --model-dir results/models \
        --debiasing dr

    # Evaluate trained models
    python scripts/train.py evaluate \
        --model-dir results/models \
        --data-dir data/ipinyou/prediction/features
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
import sys
import json
import time

import typer
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.engineering import load_feature_splits
from src.metrics.evaluation import EvalMetrics, compute_ece, compute_ieb, compute_metrics


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pure numpy ROC AUC (no sklearn dependency).

    Equivalent to sklearn.metrics.roc_auc_score for binary classification.
    Uses the ranking-based formula: AUC = (sum_ranks_positive - n1*(n1+1)/2) / (n1*n0).
    """
    y_true = np.asarray(y_true, dtype=np.intp)
    y_score = np.asarray(y_score, dtype=np.float64)
    n1 = y_true.sum()
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("Only one class present in y_true.")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_true) + 1, dtype=np.float64)
    # Handle ties: average rank for tied scores
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j > i + 1:
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg_rank
        i = j
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


app = typer.Typer(
    name="train",
    help="Train prediction models for RTB (Bid→Win→Click)",
    add_completion=False,
)


# =============================================================================
# Result Types
# =============================================================================

class ModelResult(NamedTuple):
    """Training result."""
    model_name: str
    task: str
    train_metrics: EvalMetrics
    val_metrics: EvalMetrics
    test_metrics: Optional[EvalMetrics]
    training_time: float


# =============================================================================
# CLI Display Functions
# =============================================================================

def print_metrics(name: str, metrics: EvalMetrics) -> None:
    """Print metrics in a formatted way."""
    typer.echo(f"\n{name} Metrics:")
    typer.echo(f"  AUC:       {metrics.auc:.4f}")
    typer.echo(f"  Log Loss:  {metrics.log_loss:.4f}")
    typer.echo(f"  Accuracy:  {metrics.accuracy:.4f}")
    typer.echo(f"  Precision: {metrics.precision:.4f}")
    typer.echo(f"  Recall:    {metrics.recall:.4f}")
    typer.echo(f"  F1:        {metrics.f1:.4f}")
    typer.echo(f"  ECE:       {metrics.ece:.4f}")


# =============================================================================
# Training Loop Helpers (ESMM-WC / ESCM²-WC)
# =============================================================================


def _accumulate_components(running: Optional[Dict[str, float]], components) -> Dict[str, float]:
    """Accumulate loss components across batches for epoch averaging."""
    fields = components._fields
    if running is None:
        return {f: float(getattr(components, f)) for f in fields}
    return {f: running[f] + float(getattr(components, f)) for f in fields}


def _average_components(accumulated: Dict[str, float], n_batches: int) -> Dict[str, float]:
    """Average accumulated loss components over batches."""
    return {k: v / max(n_batches, 1) for k, v in accumulated.items()}


def _format_epoch_summary(
    epoch: int,
    epochs: int,
    train_comps: Dict[str, float],
    val_comps: Dict[str, float],
    epoch_time: float,
    lr: float,
    model_type: str,
    patience: int,
    max_patience: int,
    best_val_loss: float,
    best_epoch: int,
    val_metrics: Optional[Dict[str, float]] = None,
) -> str:
    """Format epoch summary with loss components table."""
    lines = [f"  Epoch {epoch}/{epochs} ({epoch_time:.1f}s, lr={lr:.2e})"]

    # Loss component table
    # Determine which components to show (skip cfr/impute for ESMM-WC)
    skip_keys = set()
    if model_type == "esmmwc":
        skip_keys = {"cfr", "impute"}
    show_keys = [k for k in train_comps if k not in skip_keys]

    header = f"    {'Component':<14} {'Train':>10} {'Val':>10}"
    sep = f"    {'─' * 14} {'─' * 10} {'─' * 10}"
    lines.append(header)
    lines.append(sep)
    for k in show_keys:
        tv = train_comps.get(k, 0.0)
        vv = val_comps.get(k, 0.0)
        lines.append(f"    {k:<14} {tv:>10.4f} {vv:>10.4f}")

    # Val metrics (AUC / IEB / ECE) if available
    if val_metrics:
        parts = []
        for key in ("win_auc", "ctr_auc", "wctr_auc"):
            if key in val_metrics:
                label = key.replace("_", " ").upper().replace("AUC", "AUC:")
                parts.append(f"{label} {val_metrics[key]:.4f}")
        if parts:
            lines.append("    " + "  ".join(parts))

        parts2 = []
        for key in ("ctr_ieb", "wctr_ieb", "ctr_ece"):
            if key in val_metrics:
                label = key.replace("_", " ").upper().replace("IEB", "IEB:").replace("ECE", "ECE:")
                parts2.append(f"{label} {val_metrics[key]:.4f}")
        if parts2:
            lines.append("    " + "  ".join(parts2))

    # Patience / best info
    lines.append(
        f"    patience: {patience}/{max_patience}  "
        f"best: {best_val_loss:.4f} (ep{best_epoch})"
    )

    return "\n".join(lines)


def _compute_val_metrics(
    model,
    eval_step,
    val_source,
    global_batch_size: int,
    data_sharding,
    create_eval_loader_fn,
    batch_to_jax_fn,
) -> Dict[str, float]:
    """Compute validation AUC / IEB / ECE metrics.

    Runs eval_step (training=False) over entire val set, then computes:
    - Win AUC, CTR AUC (won only), WCTR AUC
    - CTR IEB, WCTR IEB, CTR ECE
    """
    all_p_win, all_p_ctr, all_p_click_bid = [], [], []
    loader = create_eval_loader_fn(val_source, global_batch_size)
    for raw_batch in loader:
        batch = batch_to_jax_fn(raw_batch, data_sharding)
        output = eval_step(model, batch)
        all_p_win.append(np.array(output.p_win))
        all_p_ctr.append(np.array(output.p_ctr))
        all_p_click_bid.append(np.array(output.p_click_bid))

    p_win = np.concatenate(all_p_win)
    p_ctr = np.concatenate(all_p_ctr)
    p_click_bid = np.concatenate(all_p_click_bid)

    # Labels from source
    win_labels = val_source.win[:len(p_win)]
    click_labels = val_source.click[:len(p_win)]

    metrics: Dict[str, float] = {}

    # Win AUC (all bids)
    try:
        metrics["win_auc"] = float(_roc_auc_score(win_labels, p_win))
    except ValueError:
        metrics["win_auc"] = 0.5

    # CTR AUC (won only)
    won_mask = win_labels == 1
    if won_mask.sum() > 0:
        try:
            metrics["ctr_auc"] = float(_roc_auc_score(click_labels[won_mask], p_ctr[won_mask]))
        except ValueError:
            metrics["ctr_auc"] = 0.5
        metrics["ctr_ece"] = float(compute_ece(click_labels[won_mask], p_ctr[won_mask]))
        metrics["ctr_ieb"] = float(compute_ieb(click_labels[won_mask], p_ctr[won_mask]))

    # WCTR AUC (all bids)
    try:
        metrics["wctr_auc"] = float(_roc_auc_score(click_labels, p_click_bid))
    except ValueError:
        metrics["wctr_auc"] = 0.5
    metrics["wctr_ieb"] = float(compute_ieb(click_labels, p_click_bid))

    return metrics


# =============================================================================
# Baseline Model (LightGBM)
# =============================================================================

@app.command()
def baseline(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing feature files",
    ),
    model_dir: Path = typer.Option(
        ...,
        "--model-dir",
        "-m",
        help="Directory to save trained models",
    ),
    task: str = typer.Option(
        "ctr",
        "--task",
        "-t",
        help="Task: 'ctr' (won only), 'win' (all bids), or 'ctr_all' (all bids)",
    ),
    n_estimators: int = typer.Option(
        300,
        "--n-estimators",
        help="Number of boosting rounds",
    ),
    learning_rate: float = typer.Option(
        0.1,
        "--learning-rate",
        "-lr",
        help="Learning rate",
    ),
    max_depth: int = typer.Option(
        6,
        "--max-depth",
        help="Maximum tree depth",
    ),
    num_leaves: int = typer.Option(
        31,
        "--num-leaves",
        help="Number of leaves per tree",
    ),
    include_lr: bool = typer.Option(
        True,
        "--include-lr/--no-include-lr",
        help="Also train LR baseline (LogisticRegression, all features)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress output",
    ),
    config_dir: Optional[str] = typer.Option(
        None,
        "--config-dir",
        help="Hydra config directory (e.g., configs)",
    ),
    overrides: Optional[str] = typer.Option(
        None,
        "--overrides",
        "-O",
        help="Hydra overrides, comma-separated (e.g., 'model.n_estimators=200,model.max_depth=8')",
    ),
) -> None:
    """Train baseline LightGBM model.

    Tasks:
      ctr     — P(Click|Win=1, X) on won impressions only (biased baseline)
      win     — P(Win|X, bid) on ALL bids (Win Tower baseline)
      ctr_all — P(Click_bid|X) on ALL bids (population-level CTR baseline)
    """
    # Load Hydra config if --config-dir or --overrides provided
    cfg = None
    if config_dir is not None or overrides is not None:
        from src.config_utils import load_config, parse_overrides
        override_list = parse_overrides(overrides)
        cfg = load_config(config_dir=config_dir, overrides=override_list)
        # Apply config values (CLI args take precedence via typer defaults)
        if n_estimators == 300 and hasattr(cfg, "model"):
            n_estimators = cfg.model.get("n_estimators", n_estimators)
        if learning_rate == 0.1 and hasattr(cfg, "model"):
            learning_rate = cfg.model.get("learning_rate", learning_rate)
        if max_depth == 6 and hasattr(cfg, "model"):
            max_depth = cfg.model.get("max_depth", max_depth)
        if num_leaves == 31 and hasattr(cfg, "model"):
            num_leaves = cfg.model.get("num_leaves", num_leaves)

    import lightgbm as lgb

    typer.echo(f"Loading data from: {data_dir}")

    # Pre-read metadata to determine needed columns (saves ~50% memory)
    import json as _json
    with open(Path(data_dir) / "feature_metadata.json") as _f:
        _meta = _json.load(_f)
    if "feature_info" not in _meta:
        typer.echo("Error: No feature_info in metadata", err=True)
        raise typer.Exit(1)
    feature_cols = _meta["feature_info"]["categorical"] + _meta["feature_info"]["numerical"]
    needed_cols = sorted(set(feature_cols + ["win", "click"]))

    train_df, val_df, test_df, metadata = load_feature_splits(data_dir, columns=needed_cols)

    # Filter to columns that actually exist
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    typer.echo(f"Using {len(feature_cols)} features")

    # Define target and data masks based on task
    if task == "ctr":
        # CTR on won impressions only (biased baseline)
        train_mask = train_df["win"] == 1
        val_mask = val_df["win"] == 1
        test_mask = test_df["win"] == 1
        target_col = "click"
    elif task == "win":
        # Win prediction: ALL bids, target = win
        train_mask = pd.Series(True, index=train_df.index)
        val_mask = pd.Series(True, index=val_df.index)
        test_mask = pd.Series(True, index=test_df.index)
        target_col = "win"
    elif task == "ctr_all":
        # Population-level CTR: ALL bids (win=0 + win=1), target = click
        # Predicts P(Click_bid|X) — ESMM-WC joint prediction comparison
        train_mask = pd.Series(True, index=train_df.index)
        val_mask = pd.Series(True, index=val_df.index)
        test_mask = pd.Series(True, index=test_df.index)
        target_col = "click"
    else:
        typer.echo(f"Error: Unknown task: {task}. Use 'ctr', 'win', or 'ctr_all'", err=True)
        raise typer.Exit(1)

    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, target_col]
    X_val = val_df.loc[val_mask, feature_cols]
    y_val = val_df.loc[val_mask, target_col]
    X_test = test_df.loc[test_mask, feature_cols]
    y_test = test_df.loc[test_mask, target_col]

    typer.echo(f"\nTraining {task.upper()} baseline:")
    typer.echo(f"  Train: {len(X_train):,} samples")
    typer.echo(f"  Val:   {len(X_val):,} samples")
    typer.echo(f"  Test:  {len(X_test):,} samples")

    # Handle categorical features
    categorical_features = [c for c in metadata["feature_info"]["categorical"] if c in feature_cols]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Training parameters
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "feature_fraction": 0.8,
        "verbose": -1 if quiet else 1,
        "seed": 42,
        "num_threads": -1,
    }

    # Note: scale_pos_weight=sqrt(neg/pos) was tested but caused adverse effects
    # on CTR tasks (1:1000+ imbalance): gradient instability → 1-tree early stop.
    # Win task (23:77 ratio) was unaffected. Removed to avoid miscalibration.

    # Train
    typer.echo("\nTraining...")
    start_time = time.time()

    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=not quiet),
            lgb.log_evaluation(period=10 if not quiet else 0),
        ],
    )

    training_time = time.time() - start_time
    typer.echo(f"Training time: {training_time:.1f}s")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Compute metrics
    y_train_vals = y_train.values if hasattr(y_train, "values") else y_train
    y_val_vals = y_val.values if hasattr(y_val, "values") else y_val
    y_test_vals = y_test.values if hasattr(y_test, "values") else y_test
    train_metrics = compute_metrics(y_train_vals, y_train_pred)
    val_metrics = compute_metrics(y_val_vals, y_val_pred)
    test_metrics = compute_metrics(y_test_vals, y_test_pred)

    print_metrics("Train", train_metrics)
    print_metrics("Val", val_metrics)
    print_metrics("Test", test_metrics)

    # Save model
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_suffix = f"lgb_{task}"
    model_path = model_dir / f"{model_suffix}.txt"
    model.save_model(str(model_path))

    # Save metadata
    result = {
        "model_name": model_suffix,
        "task": task,
        "n_features": len(feature_cols),
        "n_estimators": model.num_trees(),
        "training_time": training_time,
        "train_metrics": train_metrics._asdict(),
        "val_metrics": val_metrics._asdict(),
        "test_metrics": test_metrics._asdict(),
        "params": params,
    }

    with open(model_dir / f"{model_suffix}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    typer.echo(f"\nModel saved to: {model_path}")

    # Feature importance
    typer.echo("\nTop 10 important features:")
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(),
    }).sort_values("importance", ascending=False)

    for _, row in importance.head(10).iterrows():
        typer.echo(f"  {row['feature']}: {row['importance']}")

    # ---- LR baseline ----
    if include_lr:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import joblib

        lr_name = f"lr_{task}"
        typer.echo(f"\n--- Training LR baseline ({lr_name}, LogisticRegression, all features) ---")

        lr_start = time.time()

        scaler = StandardScaler()
        X_train_lr = scaler.fit_transform(X_train.values.astype(np.float32))
        X_val_lr = scaler.transform(X_val.values.astype(np.float32))
        X_test_lr = scaler.transform(X_test.values.astype(np.float32))

        lr_model = LogisticRegression(
            solver="saga", max_iter=100, tol=1e-4, C=1.0,
            random_state=42, n_jobs=-1,
        )
        lr_model.fit(X_train_lr, y_train)

        lr_training_time = time.time() - lr_start
        typer.echo(f"LR training time: {lr_training_time:.1f}s")

        # Predictions
        lr_train_pred = lr_model.predict_proba(X_train_lr)[:, 1]
        lr_val_pred = lr_model.predict_proba(X_val_lr)[:, 1]
        lr_test_pred = lr_model.predict_proba(X_test_lr)[:, 1]

        lr_train_metrics = compute_metrics(y_train_vals, lr_train_pred)
        lr_val_metrics = compute_metrics(y_val_vals, lr_val_pred)
        lr_test_metrics = compute_metrics(y_test_vals, lr_test_pred)

        print_metrics("LR Train", lr_train_metrics)
        print_metrics("LR Val", lr_val_metrics)
        print_metrics("LR Test", lr_test_metrics)

        # Save model + scaler + feature names
        lr_artifact = {
            "model": lr_model,
            "scaler": scaler,
            "feature_names": feature_cols,
        }
        lr_model_path = model_dir / f"{lr_name}.joblib"
        joblib.dump(lr_artifact, str(lr_model_path))

        lr_result = {
            "model_name": lr_name,
            "task": task,
            "n_features": len(feature_cols),
            "training_time": lr_training_time,
            "train_metrics": lr_train_metrics._asdict(),
            "val_metrics": lr_val_metrics._asdict(),
            "test_metrics": lr_test_metrics._asdict(),
        }
        with open(model_dir / f"{lr_name}_result.json", "w") as f:
            json.dump(lr_result, f, indent=2)

        typer.echo(f"\nLR model saved to: {lr_model_path}")


# =============================================================================
# Evaluation
# =============================================================================

@app.command()
def evaluate(
    model_dir: Path = typer.Option(
        ...,
        "--model-dir",
        "-m",
        help="Directory containing trained models",
    ),
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing feature files",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for evaluation results",
    ),
) -> None:
    """Evaluate all trained models.

    Loads all model results and creates a comparison table.
    """
    model_dir = Path(model_dir)

    # Find all result files
    result_files = list(model_dir.glob("*_result.json"))

    if not result_files:
        typer.echo("No model results found", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(result_files)} model results")

    # Load results
    results = []
    for path in result_files:
        with open(path, "r") as f:
            result = json.load(f)
            results.append(result)

    # Create comparison table
    typer.echo("\n" + "=" * 80)
    typer.echo("Model Comparison (Test Set)")
    typer.echo("=" * 80)

    header = f"{'Model':<25} {'AUC':>10} {'LogLoss':>10} {'ECE':>10} {'Time(s)':>10}"
    typer.echo(header)
    typer.echo("-" * 80)

    for r in results:
        name = r.get("model_name", "unknown")

        # Get test metrics (handle different metric keys)
        test_metrics = r.get("test_metrics") or r.get("test_ctr_metrics", {})

        auc = test_metrics.get("auc", 0)
        logloss = test_metrics.get("log_loss", 0)
        ece = test_metrics.get("ece", 0)
        time_s = r.get("training_time", 0)

        row = f"{name:<25} {auc:>10.4f} {logloss:>10.4f} {ece:>10.4f} {time_s:>10.1f}"
        typer.echo(row)

    # Save to file if requested
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w") as f:
            json.dump(results, f, indent=2)

        typer.echo(f"\nResults saved to: {output}")


@app.command()
def calibration(
    model_dir: Path = typer.Option(
        ...,
        "--model-dir",
        "-m",
        help="Directory containing trained models",
    ),
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing feature files",
    ),
    task: str = typer.Option(
        "ctr",
        "--task",
        "-t",
        help="Task: 'ctr'",
    ),
) -> None:
    """Analyze model calibration.

    Shows calibration curve data for a trained baseline model.
    """
    import lightgbm as lgb

    typer.echo(f"Loading model from: {model_dir}")

    model_path = Path(model_dir) / f"lgb_{task}.txt"
    if not model_path.exists():
        typer.echo(f"Model not found: {model_path}", err=True)
        raise typer.Exit(1)

    model = lgb.Booster(model_file=str(model_path))

    typer.echo(f"Loading data from: {data_dir}")

    # Pre-read metadata to determine needed columns (saves ~50% memory)
    import json as _json
    with open(Path(data_dir) / "feature_metadata.json") as _f:
        _meta = _json.load(_f)
    feature_cols = _meta["feature_info"]["categorical"] + _meta["feature_info"]["numerical"]
    needed_cols = sorted(set(feature_cols + ["win", "click"]))

    train_df, val_df, test_df, metadata = load_feature_splits(data_dir, columns=needed_cols)

    # Filter to columns that actually exist
    feature_cols = [c for c in feature_cols if c in test_df.columns]

    # Filter by task
    if task == "ctr":
        test_mask = test_df["win"] == 1
        target_col = "click"
    elif task == "win":
        test_mask = pd.Series(True, index=test_df.index)
        target_col = "win"
    elif task == "ctr_all":
        test_mask = pd.Series(True, index=test_df.index)
        target_col = "click"
    else:
        typer.echo(f"Error: Unknown task: {task}", err=True)
        raise typer.Exit(1)

    X_test = test_df.loc[test_mask, feature_cols]
    y_test = test_df.loc[test_mask, target_col].values

    # Predictions
    y_pred = model.predict(X_test)

    # Calibration curve
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    typer.echo("\nCalibration Curve:")
    typer.echo(f"{'Bin':<15} {'Mean Pred':>12} {'Mean Actual':>12} {'Count':>10}")
    typer.echo("-" * 55)

    for i in range(n_bins):
        in_bin = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            mean_pred = y_pred[in_bin].mean()
            mean_actual = y_test[in_bin].mean()
            count = in_bin.sum()

            bin_label = f"[{bin_boundaries[i]:.1f}, {bin_boundaries[i+1]:.1f}]"
            typer.echo(f"{bin_label:<15} {mean_pred:>12.4f} {mean_actual:>12.4f} {count:>10,}")

    # Overall calibration
    overall_pred = y_pred.mean()
    overall_actual = y_test.mean()
    typer.echo("-" * 55)
    typer.echo(f"{'Overall':<15} {overall_pred:>12.4f} {overall_actual:>12.4f} {len(y_test):>10,}")


# =============================================================================
# ESMM-WC (Bid→Win→Click, 2-tower)
# =============================================================================

@app.command()
def esmmwc(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing feature files",
    ),
    model_dir: Path = typer.Option(
        ...,
        "--model-dir",
        "-m",
        help="Directory to save trained models",
    ),
    epochs: int = typer.Option(
        50,
        "--epochs",
        "-e",
        help="Number of training epochs",
    ),
    batch_size: int = typer.Option(
        4096,
        "--batch-size",
        "-b",
        help="Batch size",
    ),
    learning_rate: float = typer.Option(
        0.001,
        "--learning-rate",
        "-lr",
        help="Learning rate",
    ),
    embedding_dim: int = typer.Option(
        16,
        "--embedding-dim",
        help="Embedding dimension",
    ),
    hidden_dims: str = typer.Option(
        "128,64",
        "--hidden-dims",
        help="CTR tower hidden dims (comma-separated)",
    ),
    win_hidden_dims: str = typer.Option(
        "64,32",
        "--win-hidden-dims",
        help="Win tower hidden dims (comma-separated)",
    ),
    dropout: float = typer.Option(
        0.3,
        "--dropout",
        help="Dropout rate",
    ),
    win_weight: float = typer.Option(
        1.0,
        "--win-weight",
        help="Win loss weight",
    ),
    ctr_weight: float = typer.Option(
        0.0,
        "--ctr-weight",
        help="CTR loss weight (ESMM has no direct CVR supervision; 0.0 per Ma et al. 2018)",
    ),
    joint_weight: float = typer.Option(
        1.0,
        "--joint-weight",
        help="Joint (ESMM) loss weight (1.0 per Ma et al. 2018)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress output",
    ),
    config_dir: Optional[str] = typer.Option(
        None,
        "--config-dir",
        help="Hydra config directory (e.g., configs)",
    ),
    overrides: Optional[str] = typer.Option(
        None,
        "--overrides",
        "-O",
        help="Hydra overrides, comma-separated (e.g., 'model=esmmwc,training.batch_size=2048')",
    ),
    distributed: bool = typer.Option(
        False,
        "--distributed/--no-distributed",
        help="Enable distributed SPMD training",
    ),
    num_devices: Optional[int] = typer.Option(
        None,
        "--num-devices",
        help="Number of devices (None = auto-detect)",
    ),
    resume_from: Optional[str] = typer.Option(
        None,
        "--resume-from",
        help="Checkpoint path to resume from",
    ),
    scheduler: str = typer.Option(
        "constant",
        "--scheduler",
        help="LR scheduler: 'constant', 'cosine', 'linear'",
    ),
    warmup_steps: int = typer.Option(
        0,
        "--warmup-steps",
        help="Number of LR warmup steps",
    ),
    gradient_clip: float = typer.Option(
        0.0,
        "--gradient-clip",
        help="Gradient clipping norm (0 = disabled)",
    ),
    weight_decay: float = typer.Option(
        1e-5,
        "--weight-decay",
        help="AdamW weight decay coefficient (0 = vanilla Adam)",
    ),
    max_samples: Optional[int] = typer.Option(
        None,
        "--max-samples",
        help="Limit training samples (for smoke testing)",
    ),
    eval_every: int = typer.Option(
        5,
        "--eval-every",
        help="Compute val AUC/ECE/IEB metrics every N epochs",
    ),
    use_wandb: bool = typer.Option(
        False,
        "--use-wandb/--no-use-wandb",
        help="Enable W&B logging",
    ),
    wandb_project: str = typer.Option(
        "rtb-ipinyou",
        "--wandb-project",
        help="W&B project name",
    ),
    wandb_run_name: Optional[str] = typer.Option(
        None,
        "--wandb-run-name",
        help="W&B run name (auto-generated if None)",
    ),
    es_metric: str = typer.Option(
        "joint",
        "--es-metric",
        help="Early stopping metric: 'total', 'joint', or 'ctr_auc' (requires eval_every=1)",
    ),
    patience: int = typer.Option(
        10,
        "--patience",
        help="Early stopping patience (epochs without improvement)",
    ),
    use_layer_norm: bool = typer.Option(
        False,
        "--use-layer-norm/--no-use-layer-norm",
        help="Use LayerNorm in MLP towers",
    ),
    use_numeric_bypass: bool = typer.Option(
        False,
        "--use-numeric-bypass/--no-use-numeric-bypass",
        help="Pass raw normalized numerical features to MLP (skip embedding projection)",
    ),
    use_scalar_input: bool = typer.Option(
        False,
        "--use-scalar-input/--no-use-scalar-input",
        help="Treat ALL features (including categorical) as scalar floats (like LR)",
    ),
    exclude_features: Optional[str] = typer.Option(
        None,
        "--exclude-features",
        help="Comma-separated feature names to exclude from training",
    ),
) -> None:
    """Train ESMM-WC model (Bid->Win->Click, 2-tower).

    ESMM constraint only, no DR/IPW. Serves as ablation baseline for ESCM2-WC.
    Uses ALL bids (not just winners) for training.
    """
    # Load Hydra config if requested
    cfg = None
    if config_dir is not None or overrides is not None:
        from src.config_utils import load_config, parse_overrides
        override_list = parse_overrides(overrides)
        cfg = load_config(config_dir=config_dir, overrides=override_list)

    _train_wc_model(
        data_dir=data_dir,
        model_dir=model_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        win_hidden_dims=win_hidden_dims,
        model_type="esmmwc",
        debiasing="none",
        quiet=quiet,
        cfg=cfg,
        distributed=distributed,
        num_devices=num_devices,
        resume_from=resume_from,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        gradient_clip=gradient_clip,
        weight_decay=weight_decay,
        max_samples=max_samples,
        eval_every_n_epochs=eval_every,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        dropout=dropout,
        win_weight=win_weight,
        ctr_weight=ctr_weight,
        joint_weight=joint_weight,
        es_metric=es_metric,
        patience=patience,
        use_layer_norm=use_layer_norm,
        use_numeric_bypass=use_numeric_bypass,
        use_scalar_input=use_scalar_input,
        exclude_features=exclude_features,
    )


# =============================================================================
# ESCM2-WC (Bid→Win→Click, 3-tower with DR/IPW)
# =============================================================================

@app.command()
def escm2wc(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing feature files",
    ),
    model_dir: Path = typer.Option(
        ...,
        "--model-dir",
        "-m",
        help="Directory to save trained models",
    ),
    epochs: int = typer.Option(
        50,
        "--epochs",
        "-e",
        help="Number of training epochs",
    ),
    batch_size: int = typer.Option(
        4096,
        "--batch-size",
        "-b",
        help="Batch size",
    ),
    learning_rate: float = typer.Option(
        0.001,
        "--learning-rate",
        "-lr",
        help="Learning rate",
    ),
    debiasing: str = typer.Option(
        "dr",
        "--debiasing",
        help="Debiasing method: 'ipw' or 'dr'",
    ),
    embedding_dim: int = typer.Option(
        16,
        "--embedding-dim",
        help="Embedding dimension",
    ),
    hidden_dims: str = typer.Option(
        "128,64",
        "--hidden-dims",
        help="CTR tower hidden dims (comma-separated)",
    ),
    win_hidden_dims: str = typer.Option(
        "64,32",
        "--win-hidden-dims",
        help="Win tower hidden dims (comma-separated)",
    ),
    dropout: float = typer.Option(
        0.3,
        "--dropout",
        help="Dropout rate",
    ),
    cfr_lambda: float = typer.Option(
        0.1,
        "--cfr-lambda",
        help="CFR regularization weight",
    ),
    win_eps: float = typer.Option(
        0.05,
        "--win-eps",
        help="Win propensity clipping floor",
    ),
    max_weight: float = typer.Option(
        10.0,
        "--max-weight",
        help="IPW/DR weight clipping ceiling",
    ),
    win_weight: float = typer.Option(
        1.0,
        "--win-weight",
        help="Win loss weight",
    ),
    ctr_weight: float = typer.Option(
        0.1,
        "--ctr-weight",
        help="CTR loss weight λ_c (ESCM² range [0, 0.1]; Wang et al. 2022)",
    ),
    joint_weight: float = typer.Option(
        1.0,
        "--joint-weight",
        help="Joint loss weight (1.0 per Wang et al. 2022)",
    ),
    impute_loss_weight: float = typer.Option(
        0.5,
        "--impute-loss-weight",
        help="Imputation loss weight (DR)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress output",
    ),
    config_dir: Optional[str] = typer.Option(
        None,
        "--config-dir",
        help="Hydra config directory (e.g., configs)",
    ),
    overrides: Optional[str] = typer.Option(
        None,
        "--overrides",
        "-O",
        help="Hydra overrides, comma-separated (e.g., 'model=escm2wc_ipw,training.batch_size=2048')",
    ),
    distributed: bool = typer.Option(
        False,
        "--distributed/--no-distributed",
        help="Enable distributed SPMD training",
    ),
    num_devices: Optional[int] = typer.Option(
        None,
        "--num-devices",
        help="Number of devices (None = auto-detect)",
    ),
    resume_from: Optional[str] = typer.Option(
        None,
        "--resume-from",
        help="Checkpoint path to resume from",
    ),
    scheduler: str = typer.Option(
        "constant",
        "--scheduler",
        help="LR scheduler: 'constant', 'cosine', 'linear'",
    ),
    warmup_steps: int = typer.Option(
        0,
        "--warmup-steps",
        help="Number of LR warmup steps",
    ),
    gradient_clip: float = typer.Option(
        0.0,
        "--gradient-clip",
        help="Gradient clipping norm (0 = disabled)",
    ),
    weight_decay: float = typer.Option(
        1e-5,
        "--weight-decay",
        help="AdamW weight decay coefficient (0 = vanilla Adam)",
    ),
    max_samples: Optional[int] = typer.Option(
        None,
        "--max-samples",
        help="Limit training samples (for smoke testing)",
    ),
    eval_every: int = typer.Option(
        5,
        "--eval-every",
        help="Compute val AUC/ECE/IEB metrics every N epochs",
    ),
    use_wandb: bool = typer.Option(
        False,
        "--use-wandb/--no-use-wandb",
        help="Enable W&B logging",
    ),
    wandb_project: str = typer.Option(
        "rtb-ipinyou",
        "--wandb-project",
        help="W&B project name",
    ),
    wandb_run_name: Optional[str] = typer.Option(
        None,
        "--wandb-run-name",
        help="W&B run name (auto-generated if None)",
    ),
    es_metric: str = typer.Option(
        "joint",
        "--es-metric",
        help="Early stopping metric: 'total', 'joint', or 'ctr_auc' (requires eval_every=1)",
    ),
    patience: int = typer.Option(
        10,
        "--patience",
        help="Early stopping patience (epochs without improvement)",
    ),
    use_layer_norm: bool = typer.Option(
        False,
        "--use-layer-norm/--no-use-layer-norm",
        help="Use LayerNorm in MLP towers",
    ),
    use_numeric_bypass: bool = typer.Option(
        False,
        "--use-numeric-bypass/--no-use-numeric-bypass",
        help="Pass raw normalized numerical features to MLP (skip embedding projection)",
    ),
    use_scalar_input: bool = typer.Option(
        False,
        "--use-scalar-input/--no-use-scalar-input",
        help="Treat ALL features (including categorical) as scalar floats (like LR)",
    ),
    exclude_features: Optional[str] = typer.Option(
        None,
        "--exclude-features",
        help="Comma-separated feature names to exclude from training",
    ),
    dr_loss_type: str = typer.Option(
        "mse",
        "--dr-loss-type",
        help="DR loss variant: 'mse' (paper default) or 'bce' (pseudo-label)",
    ),
    stop_grad_win_embedding: bool = typer.Option(
        False,
        "--stop-grad-win-embedding/--no-stop-grad-win-embedding",
        help="Stop gradient from win tower to shared embedding (gradient isolation)",
    ),
    impute_hidden_dims: Optional[str] = typer.Option(
        None,
        "--impute-hidden-dims",
        help="Imputation tower hidden dims (comma-separated, e.g. '32,16'); defaults to --hidden-dims",
    ),
    impute_loss_type: str = typer.Option(
        "mse",
        "--impute-loss-type",
        help="Imputation loss: 'mse' (default) or 'huber' (robust to outlier deltas)",
    ),
    impute_huber_delta: float = typer.Option(
        0.1,
        "--impute-huber-delta",
        help="Huber loss delta (only used when --impute-loss-type=huber)",
    ),
    win_dropout: Optional[float] = typer.Option(
        None,
        "--win-dropout",
        help="Win tower dropout (None → use global --dropout)",
    ),
    ctr_dropout: Optional[float] = typer.Option(
        None,
        "--ctr-dropout",
        help="CTR tower dropout (None → use global --dropout)",
    ),
    impute_dropout: Optional[float] = typer.Option(
        None,
        "--impute-dropout",
        help="Imputation tower dropout (None → use global --dropout)",
    ),
    top_k_avg: int = typer.Option(
        1,
        "--top-k-avg",
        help="Average top-K checkpoints by val loss (1 = no averaging, best only)",
    ),
    use_external_propensity: bool = typer.Option(
        False,
        "--use-external-propensity/--no-use-external-propensity",
        help="Use external LGB win PS for DR importance weights (instead of internal win tower)",
    ),
    external_ps_model_dir: Optional[str] = typer.Option(
        None,
        "--external-ps-model-dir",
        help="Directory containing lgb_win.txt for external win PS (defaults to --model-dir)",
    ),
) -> None:
    """Train ESCM2-WC model (Bid->Win->Click, 3-tower with DR/IPW).

    Primary debiasing model. Uses Win Tower propensity for DR/IPW debiasing
    of CTR. Uses ALL bids (not just winners) for training.
    """
    # Load Hydra config if requested
    cfg = None
    if config_dir is not None or overrides is not None:
        from src.config_utils import load_config, parse_overrides
        override_list = parse_overrides(overrides)
        cfg = load_config(config_dir=config_dir, overrides=override_list)

    _train_wc_model(
        data_dir=data_dir,
        model_dir=model_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        win_hidden_dims=win_hidden_dims,
        model_type="escm2wc",
        debiasing=debiasing,
        quiet=quiet,
        cfg=cfg,
        distributed=distributed,
        num_devices=num_devices,
        resume_from=resume_from,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        gradient_clip=gradient_clip,
        weight_decay=weight_decay,
        max_samples=max_samples,
        eval_every_n_epochs=eval_every,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        dropout=dropout,
        cfr_lambda=cfr_lambda,
        win_eps=win_eps,
        max_weight=max_weight,
        win_weight=win_weight,
        ctr_weight=ctr_weight,
        joint_weight=joint_weight,
        impute_loss_weight=impute_loss_weight,
        es_metric=es_metric,
        patience=patience,
        use_layer_norm=use_layer_norm,
        use_numeric_bypass=use_numeric_bypass,
        use_scalar_input=use_scalar_input,
        exclude_features=exclude_features,
        dr_loss_type=dr_loss_type,
        stop_grad_win_embedding=stop_grad_win_embedding,
        impute_hidden_dims=impute_hidden_dims,
        impute_loss_type=impute_loss_type,
        impute_huber_delta=impute_huber_delta,
        win_dropout=win_dropout,
        ctr_dropout=ctr_dropout,
        impute_dropout=impute_dropout,
        top_k_avg=top_k_avg,
        use_external_propensity=use_external_propensity,
        external_ps_model_dir=external_ps_model_dir,
    )


def _train_wc_model(
    data_dir: Path,
    model_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    embedding_dim: int,
    hidden_dims: str,
    win_hidden_dims: str,
    model_type: str,
    debiasing: str,
    quiet: bool,
    cfg: Optional[object] = None,
    distributed: bool = False,
    num_devices: Optional[int] = None,
    resume_from: Optional[str] = None,
    scheduler: str = "constant",
    warmup_steps: int = 0,
    gradient_clip: float = 0.0,
    weight_decay: float = 1e-5,
    max_samples: Optional[int] = None,
    use_wandb: bool = False,
    wandb_project: str = "rtb-ipinyou",
    wandb_run_name: Optional[str] = None,
    # Debiasing hyperparameters (ESCM2-WC specific, ignored for esmmwc)
    dropout: float = 0.3,
    cfr_lambda: float = 0.1,
    win_eps: float = 0.05,
    max_weight: float = 10.0,
    win_weight: float = 1.0,
    ctr_weight: float = 1.0,
    joint_weight: float = 1.0,
    impute_loss_weight: float = 0.5,
    # Evaluation frequency
    eval_every_n_epochs: int = 1,
    # W&B sweep support
    wandb_run: Optional[object] = None,
    # Early stopping
    es_metric: str = "joint",
    patience: int = 10,
    # Architecture
    use_layer_norm: bool = False,
    use_numeric_bypass: bool = False,
    use_scalar_input: bool = False,
    exclude_features: Optional[str] = None,
    # ESCM2-WC specific architecture options
    dr_loss_type: str = "mse",
    stop_grad_win_embedding: bool = False,
    impute_hidden_dims: Optional[str] = None,
    impute_loss_type: str = "mse",
    impute_huber_delta: float = 0.1,
    # Per-tower dropout (ESCM2-WC only)
    win_dropout: Optional[float] = None,
    ctr_dropout: Optional[float] = None,
    impute_dropout: Optional[float] = None,
    # Checkpoint averaging
    top_k_avg: int = 1,
    # External propensity (ESCM2-WC only)
    use_external_propensity: bool = False,
    external_ps_model_dir: Optional[str] = None,
) -> None:
    """Shared training logic for ESMM-WC and ESCM2-WC.

    Both models use the full bid space (no win==1 filtering).
    Supports single-device and distributed SPMD training via grain DataLoader.
    """
    import contextlib
    import gc

    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx
        import optax
    except ImportError:
        typer.echo(
            "Error: JAX/Flax not installed. Install with: pip install jax flax optax",
            err=True,
        )
        raise typer.Exit(1)

    from src.distributed.data_loader import (
        materialize_to_source,
        create_train_loader,
        create_eval_loader,
        batch_to_jax,
    )
    from src.distributed.mesh import MeshConfig, create_mesh, get_data_sharding
    from src.distributed.train_state import create_optimizer
    from src.distributed.checkpoint import (
        CheckpointMetadata,
        save_checkpoint,
        restore_checkpoint,
    )

    typer.echo(f"Loading data from: {data_dir}")

    # Pre-read metadata to determine needed columns (saves ~50% memory)
    import json as _json
    with open(Path(data_dir) / "feature_metadata.json") as _f:
        _meta = _json.load(_f)
    feature_cols = _meta["feature_info"]["categorical"] + _meta["feature_info"]["numerical"]
    needed_cols = sorted(set(feature_cols + ["win", "click"]))

    train_df, val_df, test_df, metadata = load_feature_splits(data_dir, columns=needed_cols)

    # Subsample for smoke testing
    if max_samples is not None:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max(max_samples // 5, 10000))
        test_df = test_df.head(max(max_samples // 5, 10000))
        typer.echo(f"  Subsampled: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    # Parse hidden dims
    hidden_dims_list = tuple(int(d) for d in hidden_dims.split(","))
    win_hidden_dims_list = tuple(int(d) for d in win_hidden_dims.split(","))
    impute_hidden_dims_list = (
        tuple(int(d) for d in impute_hidden_dims.split(","))
        if impute_hidden_dims is not None else None
    )

    model_label = "ESMM-WC" if model_type == "esmmwc" else f"ESCM2-WC({debiasing})"
    typer.echo(f"\n{model_label} Configuration:")
    typer.echo(f"  Model type: {model_type}")
    if model_type == "escm2wc":
        typer.echo(f"  Debiasing: {debiasing}")
    typer.echo(f"  Embedding dim: {embedding_dim}")
    typer.echo(f"  CTR hidden dims: {hidden_dims_list}")
    typer.echo(f"  Win hidden dims: {win_hidden_dims_list}")
    typer.echo(f"  Epochs: {epochs}")
    typer.echo(f"  Batch size: {batch_size}")
    typer.echo(f"  Learning rate: {learning_rate}")
    typer.echo(f"  Early stopping: metric={es_metric}, patience={patience}")
    if use_layer_norm:
        typer.echo(f"  LayerNorm: enabled")
    if use_numeric_bypass:
        typer.echo(f"  Numeric bypass: enabled")
    if use_scalar_input:
        typer.echo(f"  Scalar input: enabled (all features as dense scalars)")
    if exclude_features:
        typer.echo(f"  Exclude features: {exclude_features}")
    if model_type == "escm2wc":
        if impute_hidden_dims_list is not None:
            typer.echo(f"  Impute hidden dims: {impute_hidden_dims_list}")
        if stop_grad_win_embedding:
            typer.echo(f"  Stop grad win embedding: enabled")
        if dr_loss_type != "mse":
            typer.echo(f"  DR loss type: {dr_loss_type}")
        if impute_loss_type != "mse":
            typer.echo(f"  Impute loss type: {impute_loss_type} (delta={impute_huber_delta})")
        if win_dropout is not None or ctr_dropout is not None or impute_dropout is not None:
            typer.echo(f"  Per-tower dropout: win={win_dropout}, ctr={ctr_dropout}, impute={impute_dropout}")
    if use_external_propensity and model_type == "escm2wc":
        typer.echo(f"  External propensity: enabled (dir={external_ps_model_dir or model_dir})")
    if top_k_avg > 1:
        typer.echo(f"  Checkpoint averaging: top-{top_k_avg}")
    if distributed:
        typer.echo(f"  Distributed: True (devices={num_devices or 'auto'})")
        typer.echo(f"  Scheduler: {scheduler}, warmup: {warmup_steps}, clip: {gradient_clip}")

    # Get feature info and normalization stats
    feature_info = metadata.get("feature_info", {})
    cat_features = feature_info.get("categorical", [])
    num_features = feature_info.get("numerical", [])

    # Z-score normalization stats (computed from training set in build_features.py)
    norm_stats = metadata.get("normalization_stats", None)
    norm_mean = norm_stats["mean"] if norm_stats else {}
    norm_std = norm_stats["std"] if norm_stats else {}

    # Filter to existing columns
    cat_features = [c for c in cat_features if c in train_df.columns]
    num_features = [c for c in num_features if c in train_df.columns]

    # Ensure bidprice is included (required for Win Tower)
    if "bidprice" not in num_features and "bidprice" in train_df.columns:
        num_features.append("bidprice")

    # Exclude specified features
    if exclude_features:
        exclude_set = set(f.strip() for f in exclude_features.split(","))
        n_cat_before, n_num_before = len(cat_features), len(num_features)
        cat_features = [c for c in cat_features if c not in exclude_set]
        num_features = [c for c in num_features if c not in exclude_set]
        excluded_cat = n_cat_before - len(cat_features)
        excluded_num = n_num_before - len(num_features)
        typer.echo(f"\n  Excluded features: {exclude_set}")
        typer.echo(f"  Removed {excluded_cat} cat + {excluded_num} num features")

    # Scalar input mode: treat ALL features as dense scalars (like LR)
    if use_scalar_input:
        typer.echo(f"\n  Scalar input mode: converting {len(cat_features)} categorical → dense scalars")
        # Compute z-score stats for categorical features (treating them as ordinal scalars)
        for col in cat_features:
            train_df[col] = train_df[col].astype("float32")
            val_df[col] = val_df[col].astype("float32")
            test_df[col] = test_df[col].astype("float32")
            # Add normalization stats for categorical-as-scalar
            col_mean = float(train_df[col].mean())
            col_std = float(train_df[col].std())
            if col_std < 1e-8:
                col_std = 1.0
            norm_mean[col] = col_mean
            norm_std[col] = col_std
        # Move all categorical to numerical (all treated as dense)
        num_features = cat_features + num_features
        cat_features = []
        # Force numeric bypass so raw scalars pass through (no Linear(1, embed_dim))
        use_numeric_bypass = True
        typer.echo(f"  Total scalar features: {len(num_features)}")

    typer.echo(f"\nFeatures:")
    typer.echo(f"  Categorical: {len(cat_features)}")
    typer.echo(f"  Numerical: {len(num_features)}")
    typer.echo(f"  bidprice included: {'bidprice' in num_features}")

    # NOTE: Use ALL bids, no win==1 filtering
    typer.echo(f"\nData (ALL bids, no win filter):")
    typer.echo(f"  Train: {len(train_df):,} samples (win={train_df['win'].sum():,.0f})")
    typer.echo(f"  Val:   {len(val_df):,} samples (win={val_df['win'].sum():,.0f})")
    typer.echo(f"  Test:  {len(test_df):,} samples (win={test_df['win'].sum():,.0f})")

    # Auto-encode string categorical columns to integer codes
    for col in cat_features:
        if train_df[col].dtype == "object" or train_df[col].dtype.name == "category" or pd.api.types.is_string_dtype(train_df[col]):
            all_cats = sorted(set(train_df[col].dropna().unique()))
            cat_map = {v: i + 1 for i, v in enumerate(all_cats)}  # 0 reserved for unknown
            train_df[col] = train_df[col].map(cat_map).fillna(0).astype("int32")
            val_df[col] = val_df[col].map(cat_map).fillna(0).astype("int32")
            test_df[col] = test_df[col].map(cat_map).fillna(0).astype("int32")
            typer.echo(f"  Label-encoded '{col}': {len(cat_map)} categories")

    # Compute feature_dims: categorical → vocab size, numerical → -1 (dense)
    feature_dims = {}
    for col in cat_features:
        feature_dims[col] = int(train_df[col].max()) + 2  # +2 for 0-indexing + unknown
    for col in num_features:
        feature_dims[col] = -1  # Dense feature sentinel

    # External win propensity (ESCM2-WC DR only)
    ext_ps_train, ext_ps_val, ext_ps_test = None, None, None
    if use_external_propensity and model_type == "escm2wc":
        from src.debiasing.win_propensity import load_win_propensity_models
        ps_dir = Path(external_ps_model_dir) if external_ps_model_dir else model_dir
        typer.echo(f"\nLoading external win PS from: {ps_dir}")
        ps_train = load_win_propensity_models(ps_dir, train_df)
        ps_val = load_win_propensity_models(ps_dir, val_df)
        ps_test = load_win_propensity_models(ps_dir, test_df)
        ext_ps_train = ps_train.result_win.propensity_clipped
        ext_ps_val = ps_val.result_win.propensity_clipped
        ext_ps_test = ps_test.result_win.propensity_clipped
        typer.echo(f"  Train PS: AUC={ps_train.result_win.auc:.4f}, mean={ext_ps_train.mean():.4f}")
        typer.echo(f"  Val PS:   AUC={ps_val.result_win.auc:.4f}, mean={ext_ps_val.mean():.4f}")
        typer.echo(f"  Test PS:  AUC={ps_test.result_win.auc:.4f}, mean={ext_ps_test.mean():.4f}")

    # Pre-materialize DataFrames → grain RTBDataSource (numpy arrays)
    typer.echo("\nMaterializing data to numpy...")
    train_source = materialize_to_source(train_df, cat_features, num_features, norm_mean, norm_std, ext_propensity=ext_ps_train)
    val_source = materialize_to_source(val_df, cat_features, num_features, norm_mean, norm_std, ext_propensity=ext_ps_val)
    test_source = materialize_to_source(test_df, cat_features, num_features, norm_mean, norm_std, ext_propensity=ext_ps_test)
    del train_df, val_df  # Free DataFrame memory (test_df kept for evaluation labels)
    gc.collect()
    typer.echo(f"  Train source: {len(train_source):,} samples")
    typer.echo(f"  Val source: {len(val_source):,} samples")
    typer.echo(f"  Test source: {len(test_source):,} samples")

    # Build model config
    if model_type == "esmmwc":
        from src.models.esmm_wc import (
            ESMMWCConfig,
            ESMMWC,
            create_esmm_wc_loss_fn,
            create_esmm_wc_train_step,
            create_esmm_wc_eval_step,
        )

        if cfg is not None:
            from src.config_utils import build_esmmwc_config
            config = build_esmmwc_config(cfg, feature_dims)
        else:
            config = ESMMWCConfig(
                feature_dims=feature_dims,
                embed_dim=embedding_dim,
                hidden_dims=hidden_dims_list,
                win_hidden_dims=win_hidden_dims_list,
                dropout=dropout,
                win_weight=win_weight,
                ctr_weight=ctr_weight,
                joint_weight=joint_weight,
                use_layer_norm=use_layer_norm,
                use_numeric_bypass=use_numeric_bypass,
                use_fm_interaction=not use_scalar_input,
            )
        typer.echo("\nInitializing ESMM-WC model...")
        rngs = nnx.Rngs(0)
        model = ESMMWC(config, rngs=rngs)
        loss_fn_with_components = create_esmm_wc_loss_fn(config, return_components=True, jit_safe=True)
        train_step = create_esmm_wc_train_step(config)
        eval_step = create_esmm_wc_eval_step()

    else:  # escm2wc
        from src.models.escm2_wc import (
            ESCM2WCConfig,
            ESCM2WC,
            create_escm2wc_loss_fn,
            create_escm2wc_train_step,
            create_escm2wc_eval_step,
        )

        if cfg is not None:
            from src.config_utils import build_escm2wc_config
            config = build_escm2wc_config(cfg, feature_dims)
        else:
            config = ESCM2WCConfig(
                feature_dims=feature_dims,
                embed_dim=embedding_dim,
                hidden_dims=hidden_dims_list,
                win_hidden_dims=win_hidden_dims_list,
                loss_type=debiasing,
                dr_loss_type=dr_loss_type,
                dropout=dropout,
                cfr_lambda=cfr_lambda,
                win_eps=win_eps,
                max_weight=max_weight,
                win_weight=win_weight,
                ctr_weight=ctr_weight,
                joint_weight=joint_weight,
                impute_loss_weight=impute_loss_weight,
                impute_loss_type=impute_loss_type,
                impute_huber_delta=impute_huber_delta,
                use_layer_norm=use_layer_norm,
                use_numeric_bypass=use_numeric_bypass,
                use_fm_interaction=not use_scalar_input,
                stop_grad_win_embedding=stop_grad_win_embedding,
                impute_hidden_dims=impute_hidden_dims_list,
                win_dropout=win_dropout,
                ctr_dropout=ctr_dropout,
                impute_dropout=impute_dropout,
                use_external_propensity=use_external_propensity,
            )
        typer.echo(f"\nInitializing ESCM2-WC({debiasing}) model...")
        rngs = nnx.Rngs(0)
        model = ESCM2WC(config, rngs=rngs)
        loss_fn_with_components = create_escm2wc_loss_fn(config, return_components=True, jit_safe=True)
        train_step = create_escm2wc_train_step(config)
        eval_step = create_escm2wc_eval_step()

    # Distributed setup
    mesh = None
    data_sharding = None
    mesh_context = contextlib.nullcontext()

    if distributed:
        mesh = create_mesh(MeshConfig(num_devices=num_devices))
        data_sharding = get_data_sharding(mesh)
        mesh_context = mesh
        global_batch_size = batch_size * mesh.size
        typer.echo(f"\nSPMD Mesh: {mesh.size} devices, global batch size: {global_batch_size:,}")
    else:
        global_batch_size = batch_size

    # Optimizer
    start_epoch = 0
    n_samples = len(train_source)
    total_steps = epochs * (n_samples // global_batch_size)
    _use_advanced_optim = (
        scheduler != "constant"
        or warmup_steps > 0
        or gradient_clip > 0
        or weight_decay > 0
    )
    if _use_advanced_optim:
        tx = create_optimizer(
            base_lr=learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            num_devices=mesh.size if mesh else 1,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            scheduler=scheduler,
        )
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    else:
        optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    # Resume from checkpoint
    if resume_from:
        typer.echo(f"\nResuming from checkpoint: {resume_from}")
        ckpt_meta = restore_checkpoint(model, optimizer, resume_from)
        start_epoch = ckpt_meta.epoch + 1
        best_val_loss = ckpt_meta.best_val_loss
        typer.echo(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    else:
        best_val_loss = float("inf")

    best_state = None  # Will store nnx.State of best model
    # Top-K state tracking for checkpoint averaging (Phase 16)
    top_k_states = []  # List of (val_loss, epoch, nnx.State)

    # Checkpoint config
    checkpoint_enabled = distributed  # Only auto-checkpoint in distributed mode
    checkpoint_dir = "results/checkpoints"
    checkpoint_every_n_epochs = 5

    # Model name (used for W&B run name and result JSON)
    model_name = model_type if model_type == "esmmwc" else f"escm2wc_{debiasing}"

    # W&B init (conditional) — sweep agent may provide pre-initialized run
    _wandb_run_provided = wandb_run is not None
    if wandb_run is None and use_wandb:
        try:
            import wandb
            wandb_config = {
                "model_type": model_type,
                "debiasing": debiasing,
                "embedding_dim": embedding_dim,
                "hidden_dims": list(hidden_dims_list),
                "win_hidden_dims": list(win_hidden_dims_list),
                "dropout": dropout,
                "batch_size": global_batch_size,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "scheduler": scheduler,
                "warmup_steps": warmup_steps,
                "gradient_clip": gradient_clip,
                "weight_decay": weight_decay,
                "distributed": distributed,
                "num_devices": mesh.size if mesh else 1,
                "train_samples": len(train_source),
                "val_samples": len(val_source),
                "win_weight": win_weight,
                "ctr_weight": ctr_weight,
                "joint_weight": joint_weight,
                "es_metric": es_metric,
                "patience": max_patience,
                "use_layer_norm": use_layer_norm,
                "use_numeric_bypass": use_numeric_bypass,
                "use_scalar_input": use_scalar_input,
                "exclude_features": exclude_features,
            }
            if model_type == "escm2wc":
                wandb_config.update({
                    "cfr_lambda": cfr_lambda,
                    "win_eps": win_eps,
                    "max_weight": max_weight,
                    "impute_loss_weight": impute_loss_weight,
                    "dr_loss_type": dr_loss_type,
                    "stop_grad_win_embedding": stop_grad_win_embedding,
                    "impute_hidden_dims": list(impute_hidden_dims_list) if impute_hidden_dims_list else None,
                    "impute_loss_type": impute_loss_type,
                    "impute_huber_delta": impute_huber_delta if impute_loss_type == "huber" else None,
                    "win_dropout": win_dropout,
                    "ctr_dropout": ctr_dropout,
                    "impute_dropout": impute_dropout,
                    "top_k_avg": top_k_avg,
                    "use_external_propensity": use_external_propensity,
                })
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"{model_name}_ep{epochs}",
                config=wandb_config,
                tags=[model_type, debiasing] if model_type == "escm2wc" else [model_type],
            )
            typer.echo(f"  W&B run: {wandb_run.url}")
        except ImportError:
            typer.echo("Warning: wandb not installed, skipping W&B logging", err=True)
            use_wandb = False
    elif wandb_run is not None:
        # Sweep agent provided run — update config with data/training details
        wandb_run.config.update({
            "model_type": model_type,
            "debiasing": debiasing,
            "train_samples": len(train_source),
            "val_samples": len(val_source),
            "distributed": distributed,
            "num_devices": mesh.size if mesh else 1,
        }, allow_val_change=True)

    # LR schedule function (for display only)
    lr_schedule_fn = None
    if scheduler != "constant" or warmup_steps > 0:
        from src.distributed.train_state import create_lr_schedule
        lr_schedule_fn = create_lr_schedule(
            base_lr=learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            num_devices=mesh.size if mesh else 1,
            scheduler=scheduler,
        )

    # Training loop (grain DataLoader based)
    typer.echo("\nTraining...")
    start_time = time.time()

    es_patience_counter = 0
    max_patience = patience
    best_epoch = start_epoch
    global_step = start_epoch * (len(train_source) // global_batch_size)
    training_history = []
    n_batches_est = len(train_source) // global_batch_size

    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    with mesh_context:
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            # Create epoch-specific DataLoader (deterministic shuffle)
            train_loader = create_train_loader(
                train_source,
                global_batch_size,
                seed=42,
                epoch=epoch,
                num_devices=mesh.size if mesh else 1,
            )

            # --- Training (batch progress bar + component accumulation) ---
            train_comp_accum = None
            n_batches = 0

            if _has_tqdm and not quiet:
                batch_iter = tqdm(
                    train_loader, total=n_batches_est,
                    desc=f"Epoch {epoch + 1}/{epochs}", leave=False,
                )
            else:
                batch_iter = train_loader

            for raw_batch in batch_iter:
                batch = batch_to_jax(raw_batch, data_sharding)
                loss, components = train_step(model, optimizer, batch)
                train_comp_accum = _accumulate_components(train_comp_accum, components)
                n_batches += 1
                global_step += 1
                if _has_tqdm and not quiet and n_batches % 100 == 0:
                    batch_iter.set_postfix(
                        loss=f"{train_comp_accum['total'] / n_batches:.4f}"
                    )

            avg_train = _average_components(train_comp_accum, n_batches)

            # --- Validation loss (component accumulation) ---
            val_loader = create_eval_loader(val_source, global_batch_size)
            val_comp_accum = None
            val_batches = 0
            for raw_batch in val_loader:
                vbatch = batch_to_jax(raw_batch, data_sharding)
                vl, v_comp = loss_fn_with_components(model, vbatch, training=False)
                val_comp_accum = _accumulate_components(val_comp_accum, v_comp)
                val_batches += 1
            avg_val = _average_components(val_comp_accum, val_batches)

            # --- LR tracking ---
            current_lr = (
                float(lr_schedule_fn(global_step))
                if lr_schedule_fn is not None
                else learning_rate
            )

            epoch_time = time.time() - epoch_start

            # --- Full val metrics (every N epochs + first/last/near-stop) ---
            # For ctr_auc early stopping, we need metrics every epoch
            _force_eval = es_metric == "ctr_auc"
            is_eval_epoch = (
                _force_eval
                or (epoch + 1) % eval_every_n_epochs == 0
                or epoch == start_epoch
                or es_patience_counter >= max_patience
                or epoch == epochs - 1
            )
            val_metrics = (
                _compute_val_metrics(
                    model, eval_step, val_source,
                    global_batch_size, data_sharding,
                    create_eval_loader, batch_to_jax,
                )
                if is_eval_epoch
                else {}
            )

            # --- Early stopping ---
            if es_metric == "joint":
                val_loss = avg_val["joint"]
            elif es_metric == "ctr_auc":
                val_loss = -val_metrics.get("ctr_auc", 0.0)  # higher = better → negate
            else:
                val_loss = avg_val["total"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                es_patience_counter = 0
                _, best_state = nnx.split(model)
            else:
                es_patience_counter += 1

            # Track top-K states for checkpoint averaging
            if top_k_avg > 1:
                _, current_state = nnx.split(model)
                top_k_states.append((val_loss, epoch + 1, current_state))
                top_k_states.sort(key=lambda x: x[0])
                if len(top_k_states) > top_k_avg:
                    top_k_states = top_k_states[:top_k_avg]

            # --- Training history record ---
            training_history.append({
                "epoch": epoch + 1,
                **{f"train_{k}": v for k, v in avg_train.items()},
                **{f"val_{k}": v for k, v in avg_val.items()},
                "epoch_time_s": round(epoch_time, 2),
                "learning_rate": current_lr,
                "global_step": global_step,
                **val_metrics,
            })

            # --- W&B epoch logging ---
            if wandb_run is not None:
                import wandb
                wandb.log(training_history[-1], step=global_step)

            # --- Console output (every epoch) ---
            if not quiet:
                typer.echo(_format_epoch_summary(
                    epoch=epoch + 1,
                    epochs=epochs,
                    train_comps=avg_train,
                    val_comps=avg_val,
                    epoch_time=epoch_time,
                    lr=current_lr,
                    model_type=model_type,
                    patience=es_patience_counter,
                    max_patience=max_patience,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    val_metrics=val_metrics if val_metrics else None,
                ))

            # Checkpoint
            if checkpoint_enabled and (epoch + 1) % checkpoint_every_n_epochs == 0:
                ckpt_meta = CheckpointMetadata(
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    model_type=model_type,
                    config={"embedding_dim": embedding_dim, "debiasing": debiasing},
                )
                ckpt_path = save_checkpoint(
                    model, optimizer, ckpt_meta, checkpoint_dir, step=global_step,
                )
                if not quiet:
                    typer.echo(f"  Checkpoint saved: {ckpt_path}")

            if es_patience_counter >= max_patience:
                typer.echo(f"  Early stopping at epoch {epoch + 1} (metric={es_metric})")
                break

    training_time = time.time() - start_time
    typer.echo(f"\nTraining time: {training_time:.1f}s")

    # Restore best model (or averaged top-K) before evaluation
    if top_k_avg > 1 and len(top_k_states) >= 2:
        avg_epochs = [s[1] for s in top_k_states]
        typer.echo(f"  Averaging top-{len(top_k_states)} checkpoints from epochs {avg_epochs}")
        # Average only floating-point parameter leaves; keep non-numeric (RNG keys) from first state
        states = [s[2] for s in top_k_states]

        def _safe_avg(*leaves):
            if hasattr(leaves[0], 'dtype') and jnp.issubdtype(leaves[0].dtype, jnp.floating):
                return sum(leaves) / len(leaves)
            return leaves[0]

        avg_state = jax.tree.map(_safe_avg, *states)
        nnx.update(model, avg_state)
    elif best_state is not None:
        typer.echo(f"  Restoring best model from epoch {best_epoch}")
        nnx.update(model, best_state)

    # Evaluation using grain DataLoader
    typer.echo("\nEvaluating...")

    def predict_batched_grain(source, data_sharding_eval=None) -> dict:
        """Run prediction in batches using grain DataLoader."""
        all_p_win, all_p_ctr, all_p_click_bid = [], [], []
        loader = create_eval_loader(source, global_batch_size)
        for raw_batch in loader:
            batch = batch_to_jax(raw_batch, data_sharding_eval)
            output = eval_step(model, batch)
            all_p_win.append(np.array(output.p_win))
            all_p_ctr.append(np.array(output.p_ctr))
            all_p_click_bid.append(np.array(output.p_click_bid))

        return {
            "p_win": np.concatenate(all_p_win),
            "p_ctr": np.concatenate(all_p_ctr),
            "p_click_bid": np.concatenate(all_p_click_bid),
        }

    test_pred = predict_batched_grain(test_source, data_sharding)
    val_pred = predict_batched_grain(val_source, data_sharding)

    # Save per-sample test predictions (.npz) for diagnostics plots
    pred_path = model_dir / f"{model_name}_test_predictions.npz"
    np.savez_compressed(
        pred_path,
        p_win=test_pred["p_win"].astype(np.float32),
        p_ctr=test_pred["p_ctr"].astype(np.float32),
        p_click_bid=test_pred["p_click_bid"].astype(np.float32),
        y_win=test_df["win"].values.astype(np.int8),
        y_click=test_df["click"].values.astype(np.int8),
    )
    typer.echo(f"Test predictions saved: {pred_path}")

    # Win AUC (all bids)
    win_auc_test = compute_metrics(
        test_df["win"].values, test_pred["p_win"]
    )
    typer.echo(f"\nWin AUC (test, all bids): {win_auc_test.auc:.4f}")

    # CTR AUC (biased — on won impressions)
    won_mask_test = test_df["win"].values == 1
    if won_mask_test.sum() > 0:
        ctr_biased_test = compute_metrics(
            test_df["click"].values[won_mask_test],
            test_pred["p_ctr"][won_mask_test],
        )
        typer.echo(f"CTR AUC (test, won only, biased): {ctr_biased_test.auc:.4f}")
        typer.echo(f"CTR ECE (test, won only): {ctr_biased_test.ece:.4f}")

    # WCTR AUC (all bids) — P(Win) × P(Click|Win)
    wctr_auc_test = compute_metrics(
        test_df["click"].values, test_pred["p_click_bid"]
    )
    typer.echo(f"WCTR AUC (test, all bids): {wctr_auc_test.auc:.4f}")

    # IEB (Inherent Estimation Bias)
    if won_mask_test.sum() > 0:
        ctr_ieb = compute_ieb(
            test_df["click"].values[won_mask_test],
            test_pred["p_ctr"][won_mask_test],
        )
    else:
        ctr_ieb = 0.0

    wctr_ieb = compute_ieb(
        test_df["click"].values,
        test_pred["p_click_bid"],
    )

    typer.echo(f"CTR IEB (test): {ctr_ieb:.4f}")
    typer.echo(f"WCTR IEB (test): {wctr_ieb:.4f}")

    # Save results
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model_name": model_name,
        "model_type": model_type,
        "debiasing": debiasing,
        "config": {
            "embedding_dim": embedding_dim,
            "hidden_dims": list(hidden_dims_list),
            "win_hidden_dims": list(win_hidden_dims_list),
            "n_cat_features": len(cat_features),
            "n_num_features": len(num_features),
            "dropout": dropout,
            "batch_size": global_batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler": scheduler,
            "warmup_steps": warmup_steps,
            "gradient_clip": gradient_clip,
            "win_weight": win_weight,
            "ctr_weight": ctr_weight,
            "joint_weight": joint_weight,
            "es_metric": es_metric,
            "patience": max_patience,
            "use_layer_norm": use_layer_norm,
            "use_numeric_bypass": use_numeric_bypass,
            "use_scalar_input": use_scalar_input,
            "exclude_features": exclude_features,
            **({"cfr_lambda": cfr_lambda,
                "win_eps": win_eps,
                "max_weight": max_weight,
                "impute_loss_weight": impute_loss_weight,
                "dr_loss_type": dr_loss_type,
                "stop_grad_win_embedding": stop_grad_win_embedding,
                "impute_hidden_dims": list(impute_hidden_dims_list) if impute_hidden_dims_list else None,
                "impute_loss_type": impute_loss_type,
                "impute_huber_delta": impute_huber_delta if impute_loss_type == "huber" else None,
                "win_dropout": win_dropout,
                "ctr_dropout": ctr_dropout,
                "impute_dropout": impute_dropout,
                "top_k_avg": top_k_avg,
                "use_external_propensity": use_external_propensity,
                } if model_type == "escm2wc" else {}),
        },
        "training_time": training_time,
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "test_win_auc": win_auc_test.auc,
        "test_ctr_biased_auc": ctr_biased_test.auc if won_mask_test.sum() > 0 else None,
        "test_ctr_biased_ece": ctr_biased_test.ece if won_mask_test.sum() > 0 else None,
        "test_wctr_auc": wctr_auc_test.auc,
        "test_wctr_ece": wctr_auc_test.ece,
        "test_ctr_ieb": ctr_ieb,
        "test_wctr_ieb": wctr_ieb,
        "training_history": training_history,
    }

    result_path = model_dir / f"{model_name}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    typer.echo(f"\nResults saved to: {result_path}")

    # W&B: log final test metrics + summary + finish
    if wandb_run is not None:
        import wandb
        # Log final test metrics as a regular log entry (appears in W&B charts)
        final_metrics = {
            k: v for k, v in result.items()
            if k not in ("training_history", "config", "model_name", "model_type",
                         "debiasing", "epochs", "best_epoch")
            and v is not None
        }
        wandb.log(final_metrics, step=global_step + 1)
        # Also set summary for sweep dashboard / runs table
        wandb_run.summary.update({
            k: v for k, v in result.items()
            if k != "training_history" and v is not None
        })
        if not _wandb_run_provided:
            wandb.finish()
            typer.echo("W&B run finished.")


if __name__ == "__main__":
    app()
