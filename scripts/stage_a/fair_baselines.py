"""Stage A — FAIR-split baselines (the first decisive signal).

Trains LR_ctr_all (StandardScaler + LogisticRegression) and LGB_ctr_all on the
FAIR per-advertiser-temporal split (data/ipinyou/prediction/features_fair/),
all-bids click target, then evaluates on FAIR test.

THE KEY QUESTION: does LR_ctr_all winners-only AUC on the FAIR split DROP from
the old 0.714 toward ~0.55? If so, the old 0.714 was the unseen-advertiser-2997
artifact (between-advertiser base-rate ranking that disappears once 2997 — and
the disjoint S2/S3 advertiser vocab — are folded into train).

PREPROCESSING (replicated EXACTLY, zero train/serve skew — mirrors
scripts/stage_a/build_unified_predictions.py + scripts/train.py baseline path):
  - feature_cols = categorical + numerical from feature_metadata.json (30 feats)
  - feature frame via load_feature_splits (nullable-int->0 fill, float keeps NaN)
  - str/object cols (slot_size_group, region_group) -> union(train+val+test)
    sorted cat.codes int32 (identical codes to training)
  - LR: StandardScaler.fit on train -> transform; LogisticRegression saga
        (matches scripts/train.py:517-526)
  - LGB: lgb.train binary/auc with declared categorical_feature; predict on RAW
        integer-coded features (matches scripts/train.py:410-448)

OUTPUTS:
  - results/stage_a/fair_baselines.json   (metrics + verdict)
  - results/stage_a/fair_baseline_preds.npz  (winners pCTR + y_win + y_click +
        advertiser; ALSO full all-bids pCTR for both models, same test row order)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.engineering import load_feature_splits
from src.metrics.evaluation import _numpy_roc_auc, compute_ieb
from src.metrics.calibration import quantile_reliability

pd_is_string = pd.api.types.is_string_dtype

PROJECT = Path("/home/mail-agent/project/rtb_ipinyou")
FAIR = PROJECT / "data" / "ipinyou" / "prediction" / "features_fair"
OUTDIR = PROJECT / "results" / "stage_a"
JSON_OUT = OUTDIR / "fair_baselines.json"
NPZ_OUT = OUTDIR / "fair_baseline_preds.npz"

# Old (adversarial-split) reference numbers for the verdict comparison.
OLD_LR_WINNERS_AUC = 0.7144
OLD_LGB_WINNERS_AUC = 0.4786
OLD_DECISIVE_ADVERTISER = 2997  # unseen high-CTR advertiser that carried LR's 0.714

# LR-fit subsample (stratified to keep all clicks). Train all-bids is ~90.6M rows;
# we subsample negatives for the LR fit cost while keeping 100% of positive clicks.
LR_FIT_TARGET_ROWS = 25_000_000
SEED = 42


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.intp)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return _numpy_roc_auc(y_true, y_score)


def _load_feature_cols() -> list[str]:
    with open(FAIR / "feature_metadata.json") as f:
        meta = json.load(f)
    fi = meta["feature_info"]
    return fi["categorical"] + fi["numerical"], fi["categorical"]


def _encode_str_cols(train_df, val_df, test_df, feature_cols):
    """Union(train+val+test) sorted cat.codes int32 for str/object feature cols.

    Mirrors build_unified_predictions.py: load_feature_splits leaves
    slot_size_group / region_group as StringDtype (dtype != 'object') so its
    internal branch skips them; we replicate the official union-coding here.
    """
    str_cols = [
        c for c in feature_cols
        if (test_df[c].dtype == "object") or pd_is_string(test_df[c])
    ]
    for col in str_cols:
        uniques = sorted(
            set(train_df[col].unique())
            | set(val_df[col].unique())
            | set(test_df[col].unique())
        )
        cat = pd.CategoricalDtype(categories=uniques)
        for d in (train_df, val_df, test_df):
            d[col] = d[col].astype(cat).cat.codes.astype("int32")
    return str_cols


def _per_advertiser_winners_auc(y_click, p, win_mask, advertiser):
    """Winners-only AUC per advertiser (within the win==1 subset)."""
    w_click = y_click[win_mask]
    w_p = p[win_mask]
    w_adv = advertiser[win_mask]
    rows = {}
    for adv in np.unique(w_adv):
        m = w_adv == adv
        n = int(m.sum())
        clk = int(w_click[m].sum())
        auc = _safe_auc(w_click[m], w_p[m]) if clk > 0 and clk < n else float("nan")
        rows[int(adv)] = {"n_won": n, "n_click": clk, "ctr": clk / n if n else 0.0, "winners_auc": auc}
    return rows


def _eval_model(name, p_all, y_click, y_win, advertiser):
    """All-bids AUC, winners-only AUC, quantile-ECE (winners + all), per-adv winners-AUC,
    and the same with the previously-decisive advertiser excluded (if present in won set)."""
    win_mask = y_win == 1
    w_click = y_click[win_mask]
    w_p = p_all[win_mask]

    all_auc = _safe_auc(y_click, p_all)
    win_auc = _safe_auc(w_click, w_p)

    q_all = quantile_reliability(y_click, p_all, n_bins=10)
    q_win = quantile_reliability(w_click, w_p, n_bins=10)

    per_adv = _per_advertiser_winners_auc(y_click, p_all, win_mask, advertiser)

    # Exclude previously-decisive advertiser (2997) from the WON subset, if present.
    won_advs = set(int(a) for a in np.unique(advertiser[win_mask]))
    excl = {}
    if OLD_DECISIVE_ADVERTISER in won_advs:
        keep = win_mask & (advertiser != OLD_DECISIVE_ADVERTISER)
        excl = {
            "advertiser_excluded": OLD_DECISIVE_ADVERTISER,
            "n_won": int(keep.sum()),
            "winners_auc": _safe_auc(y_click[keep], p_all[keep]),
        }
    else:
        excl = {
            "advertiser_excluded": OLD_DECISIVE_ADVERTISER,
            "note": f"advertiser {OLD_DECISIVE_ADVERTISER} not present in FAIR test WON subset; "
                    f"won advertisers = {sorted(won_advs)}",
        }

    return {
        "all_bids_auc": all_auc,
        "winners_auc": win_auc,
        "all_bids_quantile_ece": q_all.quantile_ece,
        "winners_quantile_ece": q_win.quantile_ece,
        "all_bids_ieb": compute_ieb(y_click, p_all),
        "winners_ieb": compute_ieb(w_click, w_p),
        "all_bids_pred_mean": float(p_all.mean()),
        "all_bids_true_mean": float(y_click.mean()),
        "winners_pred_mean": float(w_p.mean()),
        "winners_true_mean": float(w_click.mean()),
        "per_advertiser_winners_auc": per_adv,
        "winners_auc_excl_decisive_adv": excl,
    }


def main() -> None:
    t0 = time.time()
    feature_cols, categorical = _load_feature_cols()
    print(f"feature_cols ({len(feature_cols)}): {feature_cols}")
    print(f"categorical ({len(categorical)}): {categorical}")

    need = sorted(set(feature_cols + ["win", "click", "advertiser"]))
    print("Loading FAIR splits via load_feature_splits (zero-skew) ...")
    train_df, val_df, test_df, metadata = load_feature_splits(FAIR, columns=need)
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    categorical = [c for c in categorical if c in feature_cols]
    print(f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    str_cols = _encode_str_cols(train_df, val_df, test_df, feature_cols)
    print(f"str/object cols union-coded: {str_cols}")

    # Zero-skew fix: in the ORIGINAL features/ split, load_feature_splits returns
    # adexchange / slotvisibility / slotformat as int64 with nulls already filled
    # to 0 (_to_numpy_dtypes int-fill; documented baseline R4/R10). In the FAIR
    # parquet these categoricals are stored as float64, so they come back with
    # NaN. Replicate the SAME 0-fill -> int cast on any categorical feature col
    # that still carries NaN, so LR/LGB see identical inputs to the documented
    # baseline (0 is also the dominant real train code for these columns).
    nan_cat_cols = [
        c for c in categorical
        if hasattr(train_df[c], "isna") and (
            int(train_df[c].isna().sum()) or int(val_df[c].isna().sum()) or int(test_df[c].isna().sum())
        )
    ]
    for col in nan_cat_cols:
        for d in (train_df, val_df, test_df):
            d[col] = d[col].fillna(0).astype("int64")
    print(f"categorical cols 0-filled (NaN->0, int cast, baseline parity): {nan_cat_cols}")

    # ----- all-bids click target (ctr_all) -----
    y_train = train_df["click"].to_numpy().astype(np.int8)
    y_val = val_df["click"].to_numpy().astype(np.int8)
    y_test = test_df["click"].to_numpy().astype(np.int8)
    y_win_test = test_df["win"].to_numpy().astype(np.int8)
    adv_test = test_df["advertiser"].to_numpy().astype(np.int32)
    y_win_val = val_df["win"].to_numpy().astype(np.int8)  # for frozen-map val->test recalibration

    print(f"train clicks={int(y_train.sum()):,} ({y_train.mean():.6f}) | "
          f"test clicks={int(y_test.sum()):,} ({y_test.mean():.6f}) | "
          f"test won={int(y_win_test.sum()):,} winners-CTR={y_test[y_win_test==1].mean():.6f}")

    results = {
        "split": "features_fair (per_advertiser_temporal_0.70_0.15_0.15)",
        "target": "ctr_all (all-bids click)",
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "test_clicks": int(y_test.sum()),
        "test_won": int(y_win_test.sum()),
        "test_winners_ctr": float(y_test[y_win_test == 1].mean()),
        "test_won_advertisers": sorted(int(a) for a in np.unique(adv_test[y_win_test == 1])),
    }

    # =========================================================================
    # LR_ctr_all  (StandardScaler + LogisticRegression, stratified subsample fit)
    # =========================================================================
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("\n=== LR_ctr_all ===")
    Xtr_full = train_df[feature_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(Xtr_full).all():
        bad = [feature_cols[j] for j in range(Xtr_full.shape[1])
               if not np.isfinite(Xtr_full[:, j]).all()]
        raise ValueError(f"Non-finite values remain in LR train matrix cols: {bad}")

    # Stratified subsample: keep ALL positives, sample negatives to hit target rows.
    rng = np.random.RandomState(SEED)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    n_neg_keep = min(len(neg_idx), max(0, LR_FIT_TARGET_ROWS - len(pos_idx)))
    neg_sample = rng.choice(neg_idx, size=n_neg_keep, replace=False)
    fit_idx = np.concatenate([pos_idx, neg_sample])
    fit_idx.sort()
    print(f"LR fit subsample: {len(fit_idx):,} rows "
          f"(all {len(pos_idx):,} pos + {n_neg_keep:,} of {len(neg_idx):,} neg) "
          f"out of {len(y_train):,} train rows")

    # StandardScaler fit on the SAME subsample used for the LR fit (the rows the
    # model actually sees), matching train.py's fit_transform(X_train).
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(Xtr_full[fit_idx])
    y_fit = y_train[fit_idx]
    del Xtr_full

    lr = LogisticRegression(solver="saga", max_iter=100, tol=1e-4, C=1.0,
                            random_state=SEED, n_jobs=-1)
    t_lr = time.time()
    lr.fit(X_fit, y_fit)
    print(f"LR fit time: {time.time()-t_lr:.1f}s  n_iter={lr.n_iter_}")
    del X_fit, y_fit

    Xte = test_df[feature_cols].to_numpy(dtype=np.float32)
    lr_p_test = lr.predict_proba(scaler.transform(Xte))[:, 1].astype(np.float32)
    del Xte
    # VAL predictions too -> enables frozen-map val->test post-hoc recalibration.
    Xva = val_df[feature_cols].to_numpy(dtype=np.float32)
    lr_p_val = lr.predict_proba(scaler.transform(Xva))[:, 1].astype(np.float32)
    del Xva
    results["LR_ctr_all"] = _eval_model("LR", lr_p_test, y_test, y_win_test, adv_test)
    results["LR_ctr_all"]["lr_fit_subsample_rows"] = int(len(fit_idx))
    results["LR_ctr_all"]["lr_n_iter"] = int(lr.n_iter_[0])
    print(f"  all-bids AUC={results['LR_ctr_all']['all_bids_auc']:.4f}  "
          f"winners AUC={results['LR_ctr_all']['winners_auc']:.4f}")

    # =========================================================================
    # LGB_ctr_all  (full train, declared categoricals, raw integer features)
    # =========================================================================
    import lightgbm as lgb

    print("\n=== LGB_ctr_all ===")
    Xtr = train_df[feature_cols]
    Xval = val_df[feature_cols]
    Xte = test_df[feature_cols]
    cat_idx = [feature_cols.index(c) for c in categorical]

    train_data = lgb.Dataset(Xtr, label=y_train, categorical_feature=cat_idx, free_raw_data=False)
    val_data = lgb.Dataset(Xval, label=y_val, reference=train_data, free_raw_data=False)
    params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "learning_rate": 0.1, "max_depth": 6, "num_leaves": 31,
        "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
        "feature_fraction": 0.8, "verbose": -1, "seed": SEED, "num_threads": -1,
    }
    t_lgb = time.time()
    booster = lgb.train(
        params, train_data, num_boost_round=300,
        valid_sets=[train_data, val_data], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True),
                   lgb.log_evaluation(period=25)],
    )
    print(f"LGB train time: {time.time()-t_lgb:.1f}s  n_trees={booster.num_trees()}")
    lgb_p_test = booster.predict(Xte).astype(np.float32)
    lgb_p_val = booster.predict(Xval).astype(np.float32)  # for frozen-map val->test recalibration
    results["LGB_ctr_all"] = _eval_model("LGB", lgb_p_test, y_test, y_win_test, adv_test)
    results["LGB_ctr_all"]["lgb_n_trees"] = int(booster.num_trees())
    print(f"  all-bids AUC={results['LGB_ctr_all']['all_bids_auc']:.4f}  "
          f"winners AUC={results['LGB_ctr_all']['winners_auc']:.4f}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    lr_win = results["LR_ctr_all"]["winners_auc"]
    lgb_win = results["LGB_ctr_all"]["winners_auc"]
    drop = OLD_LR_WINNERS_AUC - lr_win
    confirmed = lr_win < 0.60  # dropped toward ~0.55 (away from 0.714)
    results["verdict"] = {
        "old_adversarial_split_LR_winners_auc": OLD_LR_WINNERS_AUC,
        "fair_split_LR_winners_auc": lr_win,
        "LR_winners_auc_drop": drop,
        "old_adversarial_split_LGB_winners_auc": OLD_LGB_WINNERS_AUC,
        "fair_split_LGB_winners_auc": lgb_win,
        "decisive_advertiser_2997_in_test_won": OLD_DECISIVE_ADVERTISER in results["test_won_advertisers"],
        "confirmed_artifact": bool(confirmed),
        "statement": (
            f"LR winners-AUC dropped {OLD_LR_WINNERS_AUC:.3f} -> {lr_win:.3f} "
            f"(Δ={drop:+.3f}) on the FAIR split. "
            + ("CONFIRMED: the old 0.714 was the unseen-advertiser-2997 / disjoint-vocab artifact. "
               if confirmed else
               "NOT confirmed: LR winners-AUC stays high on the FAIR split. ")
            + (f"Note: advertiser 2997 does NOT appear in the FAIR test WON subset "
               f"(won advertisers = {results['test_won_advertisers']}), so within-test "
               f"between-advertiser CTR spread is now narrow and the 2997 ranking axis is absent."
               if OLD_DECISIVE_ADVERTISER not in results["test_won_advertisers"]
               else "")
        ),
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {JSON_OUT}")

    # Save predictions for later neural comparison reuse (full test row order).
    # val arrays (lr_p_val/lgb_p_val/y_win_val/y_click_val) enable a frozen-map
    # val->test post-hoc recalibration without refitting the baselines.
    np.savez_compressed(
        NPZ_OUT,
        lr_p_all=lr_p_test,
        lgb_p_all=lgb_p_test,
        y_win=y_win_test,
        y_click=y_test,
        advertiser=adv_test,
        lr_p_val=lr_p_val,
        lgb_p_val=lgb_p_val,
        y_win_val=y_win_val,
        y_click_val=y_val,
    )
    print(f"Saved {NPZ_OUT}")
    print(f"\nTotal time: {time.time()-t0:.1f}s")

    # Console summary table
    print("\n================ FAIR-SPLIT BASELINE SUMMARY ================")
    print(f"{'metric':36s} {'LR_ctr_all':>14s} {'LGB_ctr_all':>14s}")
    for key, label in [
        ("all_bids_auc", "all-bids AUC"),
        ("winners_auc", "winners-only AUC"),
        ("all_bids_quantile_ece", "all-bids quantile-ECE"),
        ("winners_quantile_ece", "winners quantile-ECE"),
        ("winners_ieb", "winners IEB"),
    ]:
        print(f"{label:36s} {results['LR_ctr_all'][key]:>14.4f} {results['LGB_ctr_all'][key]:>14.4f}")
    print(f"\nVERDICT: {results['verdict']['statement']}")


if __name__ == "__main__":
    main()
