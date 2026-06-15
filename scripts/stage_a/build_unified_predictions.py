"""Stage A — Step 1: assemble a single unified test-predictions npz.

Builds ``results/stage_a/test_predictions_all.npz`` holding, for the 19,424,025
test rows (SAME row order as the neural npz and test.parquet), the predictions of
every model on a common index, plus slice/label columns.

Models and stored prediction arrays (all float32, shape (N,)):
  Neural (esmmwc / escm2wc_dr / escm2wc_dr_extps), from the saved npz:
    {model}_p_ctr        -> p_ctr        = P(click|win) winners-CTR tower
    {model}_p_click_bid  -> p_click_bid  = p_win * p_ctr = all-bids pCTR
  Baselines (computed here from test.parquet):
    lr_ctr_all   -> LogisticRegression all-bids pCTR
    lgb_ctr_all  -> LightGBM all-bids pCTR

Slice / label columns (compact dtypes):
    y_win (int8), y_click (int8), adexchange, advertiser, hour, payprice, bidprice.

PREPROCESSING (replicated exactly, zero train/serve skew):

  Feature order — ``feature_cols = categorical + numerical`` from
  ``feature_metadata.json["feature_info"]`` (scripts/train.py:361), identical to
  the joblib artifact's ``feature_names`` and the LGB booster's ``feature_name()``
  (both verified == the 30-feature list, same order).

  Feature MATRIX — we DO NOT read the raw parquet directly. Training built the
  feature frame via ``load_feature_splits`` (scripts/train.py:364), which applies
  TWO transforms the raw parquet lacks:
    (a) ``_to_numpy_dtypes`` (engineering.py:812) — ArrowDtype/nullable int cols are
        fillna(0)->int; float cols keep NaN. This fills the null cells in
        adexchange (2.0M nulls), slotvisibility/slotformat (100% null) with 0.
    (b) object-col integer coding (engineering.py:855-861) — slot_size_group /
        region_group (str) -> cat.codes using the UNION of train+val+test uniques,
        sorted. Codes therefore depend on all three splits, so we call the official
        ``load_feature_splits`` (loading the needed cols of all 3 splits) and keep
        only the returned test_df. This guarantees identical codes to training.

  LR_ctr_all (replicates scripts/train.py:517-520, 532-534; mirrored in
  src/serving/app.py:242-243, 299):
      X  = test_df[feature_cols].values.astype(np.float32)
      Xs = scaler.transform(X)                  # the saved StandardScaler
      p  = model.predict_proba(Xs)[:, 1]

  LGB_ctr_all (replicates scripts/train.py:454-456): predict on RAW (unscaled)
  features in feature_cols order:
      p = booster.predict(test_df[feature_cols])
  Categorical columns were declared at train time (scripts/train.py:407-410); the
  booster stores them, so the same integer-coded test_df[feature_cols] is passed
  (identical to training's X_test = test_df.loc[test_mask, feature_cols]).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.engineering import load_feature_splits

pd_is_string = pd.api.types.is_string_dtype

PROJECT = Path("/home/mail-agent/project/rtb_ipinyou")
MODELS = PROJECT / "results" / "models"
FEATURES = PROJECT / "data" / "ipinyou" / "prediction" / "features"
OUT = PROJECT / "results" / "stage_a" / "test_predictions_all.npz"

N_EXPECTED = 19_424_025
NEURAL = ["esmmwc", "escm2wc_dr", "escm2wc_dr_extps"]
SLICE_LABEL_COLS = ["adexchange", "advertiser", "hour", "payprice", "bidprice"]


def _load_feature_cols() -> list[str]:
    with open(FEATURES / "feature_metadata.json") as f:
        meta = json.load(f)
    fi = meta["feature_info"]
    return fi["categorical"] + fi["numerical"]


def main() -> None:
    feature_cols = _load_feature_cols()
    print(f"feature_cols ({len(feature_cols)}): {feature_cols}")

    # --- build the feature frame via the OFFICIAL loader (same as training:
    #     scripts/train.py:364). Loads needed cols of all 3 splits so object-col
    #     cat.codes use the identical train+val+test union; we keep only test_df. ---
    need = sorted(set(feature_cols + ["win", "click"] + SLICE_LABEL_COLS))
    print("Loading splits via load_feature_splits (zero-skew) ...")
    train_df, val_df, test_df, _ = load_feature_splits(FEATURES, columns=need)
    n = len(test_df)
    print(f"test split rows: {n:,}")
    assert n == N_EXPECTED, f"row count {n} != {N_EXPECTED}"

    # load_feature_splits encodes columns whose dtype == "object" via cat.codes
    # (engineering.py:855-861). Under the current pandas, slot_size_group /
    # region_group read back as the "str" StringDtype (dtype != "object"), so the
    # loader's branch SKIPS them. We replicate the SAME encoding here — union of
    # train+val+test uniques, sorted, cat.codes -> int32 — which is byte-identical
    # to training: verified train-split code means equal the saved StandardScaler
    # means exactly (slot_size_group 2.753114; region_group 1.298714).
    str_cols = [
        c for c in feature_cols
        if (test_df[c].dtype == "object") or pd_is_string(test_df[c])
    ]
    if str_cols:
        print(f"Encoding str/object feature cols via union cat.codes: {str_cols}")
    for col in str_cols:
        uniques = sorted(
            set(train_df[col].unique())
            | set(val_df[col].unique())
            | set(test_df[col].unique())
        )
        cat = pd.CategoricalDtype(categories=uniques)
        test_df[col] = test_df[col].astype(cat).cat.codes.astype("int32")
    del train_df, val_df

    out: dict[str, np.ndarray] = {}

    # --- neural predictions (from saved npz, same row order) ---
    for m in NEURAL:
        d = np.load(MODELS / f"{m}_test_predictions.npz")
        assert d["p_ctr"].shape[0] == N_EXPECTED, f"{m} length mismatch"
        # cross-check labels match parquet (sanity: same split / order)
        assert int(d["y_win"].sum()) == int(test_df["win"].sum()), f"{m} y_win mismatch"
        assert int(d["y_click"].sum()) == int(test_df["click"].sum()), f"{m} y_click mismatch"
        out[f"{m}_p_ctr"] = d["p_ctr"].astype(np.float32)
        out[f"{m}_p_click_bid"] = d["p_click_bid"].astype(np.float32)
        print(f"  {m}: p_ctr/p_click_bid loaded (sum y_win/y_click matched parquet)")

    # --- LR_ctr_all ---
    print("Computing LR_ctr_all ...")
    art = joblib.load(MODELS / "lr_ctr_all.joblib")
    lr_model, scaler, lr_feats = art["model"], art["scaler"], list(art["feature_names"])
    assert lr_feats == feature_cols, "LR feature_names order != feature_cols"
    X = test_df[feature_cols].values.astype(np.float32)
    Xs = scaler.transform(X)
    out["lr_ctr_all"] = lr_model.predict_proba(Xs)[:, 1].astype(np.float32)
    del X, Xs
    print("  LR_ctr_all done")

    # --- LGB_ctr_all (raw features) ---
    print("Computing LGB_ctr_all ...")
    booster = lgb.Booster(model_file=str(MODELS / "lgb_ctr_all.txt"))
    assert booster.feature_name() == feature_cols, "LGB feature_name order != feature_cols"
    out["lgb_ctr_all"] = booster.predict(test_df[feature_cols]).astype(np.float32)
    print("  LGB_ctr_all done")

    # --- slice / label columns (compact) ---
    out["y_win"] = test_df["win"].values.astype(np.int8)
    out["y_click"] = test_df["click"].values.astype(np.int8)
    out["adexchange"] = test_df["adexchange"].values.astype(np.int16)
    out["advertiser"] = test_df["advertiser"].values.astype(np.int32)
    out["hour"] = test_df["hour"].values.astype(np.int8)
    out["payprice"] = test_df["payprice"].values.astype(np.int32)
    out["bidprice"] = test_df["bidprice"].values.astype(np.int32)

    # --- verify all lengths ---
    for k, v in out.items():
        assert v.shape[0] == N_EXPECTED, f"{k} length {v.shape[0]} != {N_EXPECTED}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)
    print(f"\nSaved {OUT}")
    for k, v in out.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")


if __name__ == "__main__":
    main()
