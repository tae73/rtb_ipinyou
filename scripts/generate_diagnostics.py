"""Generate 3-panel diagnostics figures for LGB, LR and Neural models."""

from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from src.features.engineering import load_feature_splits
from src.metrics.diagnostics_plot import plot_prediction_diagnostics

FEATURES_DIR = Path("data/ipinyou/prediction/features")
MODEL_DIR = Path("results/models")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_lgb_diagnostics() -> None:
    """LGB CTR (winners-only) 3-panel diagnostics."""
    lgb_path = MODEL_DIR / "lgb_ctr.txt"
    if not lgb_path.exists():
        print(f"[SKIP] {lgb_path} not found")
        return

    _, _, test_df, metadata = load_feature_splits(FEATURES_DIR)
    model = lgb.Booster(model_file=str(lgb_path))

    feature_info = metadata.get("feature_info", {})
    feature_cols = feature_info.get("categorical", []) + feature_info.get("numerical", [])
    feature_cols = [c for c in feature_cols if c in test_df.columns]

    # Winners only
    won_mask = test_df["win"] == 1
    X_test = test_df.loc[won_mask, feature_cols]
    y_test = test_df.loc[won_mask, "click"].values
    lgb_pred = model.predict(X_test)

    save_path = FIG_DIR / "03_lgb_ctr_diagnostics.png"
    plot_prediction_diagnostics(y_test, lgb_pred, "LGB CTR (winners only)", save_path=save_path)
    plt.close()
    # Also overwrite legacy notebook path (Figure 5 in prediction_report.md)
    legacy_path = FIG_DIR / "03_prediction_baseline_calibration.png"
    plot_prediction_diagnostics(y_test, lgb_pred, "LGB CTR (winners only)", save_path=legacy_path)
    plt.close()
    auc = roc_auc_score(y_test, lgb_pred)
    print(f"LGB CTR — AUC: {auc:.4f}, saved: {save_path}, {legacy_path}")


def generate_lr_diagnostics() -> None:
    """LR CTR_all 3-panel diagnostics (NB03 Cell 34 logic)."""
    lr_path = MODEL_DIR / "lr_ctr_all.joblib"
    if not lr_path.exists():
        print(f"[SKIP] {lr_path} not found")
        return

    _, _, test_df, metadata = load_feature_splits(FEATURES_DIR)
    artifact = joblib.load(lr_path)
    lr_model = artifact["model"]
    scaler = artifact["scaler"]

    feature_info = metadata.get("feature_info", {})
    cat_features = feature_info.get("categorical", [])
    num_features = feature_info.get("numerical", [])
    default_cols = [c for c in cat_features + num_features if c in test_df.columns]
    lr_feature_cols = artifact.get("feature_names", default_cols)

    X_test = scaler.transform(test_df[lr_feature_cols].values.astype(np.float32))
    y_test = test_df["click"].values
    lr_pred = lr_model.predict_proba(X_test)[:, 1]

    save_path = FIG_DIR / "03_lr_ctr_all_diagnostics.png"
    plot_prediction_diagnostics(y_test, lr_pred, "LR CTR_all (all bids)", save_path=save_path)
    plt.close()
    auc = roc_auc_score(y_test, lr_pred)
    print(f"LR CTR_all — AUC: {auc:.4f}, saved: {save_path}")


def generate_neural_diagnostics() -> None:
    """Neural model 3-panel diagnostics (NB04 Cell 27 logic)."""
    npz_models = {
        "ESMM-WC (Run J)": "esmmwc_test_predictions.npz",
        "ESCM2-WC(DR) (Run AL)": "escm2wc_dr_test_predictions.npz",
        "ESCM2-WC(DR)+ExtPS (Run AW)": "escm2wc_dr_extps_test_predictions.npz",
    }

    for name, fname in npz_models.items():
        npz_path = MODEL_DIR / fname
        if not npz_path.exists():
            print(f"[SKIP] {npz_path} not found — retrain with updated scripts/train.py")
            continue
        pred = np.load(npz_path)
        safe_name = fname.replace("_test_predictions.npz", "").replace("+", "_")
        save_path = FIG_DIR / f"04_{safe_name}_diagnostics.png"
        plot_prediction_diagnostics(
            pred["y_click"], pred["p_click_bid"], name, save_path=save_path,
        )
        plt.close()
        auc = roc_auc_score(pred["y_click"], pred["p_click_bid"])
        print(f"{name} — AUC: {auc:.4f}, saved: {save_path}")


if __name__ == "__main__":
    print("=== LGB CTR Diagnostics ===")
    generate_lgb_diagnostics()
    print("\n=== LR CTR_all Diagnostics ===")
    generate_lr_diagnostics()
    print("\n=== Neural Model Diagnostics ===")
    generate_neural_diagnostics()
    print("\nDone.")
