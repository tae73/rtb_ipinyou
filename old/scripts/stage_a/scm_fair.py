"""Phase 1e — SCM / DAG causal analysis on the FAIR split (CPU, exploratory).

Honest framing: this is a structural-causal *diagnostic*, NOT an identified causal claim. iPinYou's
flat-bid logging (a handful of discrete bid levels) and won-only surplus (lost inventory censored)
mean the bid→surplus effect is not credibly identified; we report the DoWhy backdoor estimate plus
three refutation tests as robustness diagnostics, exactly as hypothesis-generating evidence.

Fair-split replacement for the original/unfair-split notebook 09b results.
Output: results/stage_a/scm_fair.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.bidding.value import ValueConfig
from src.causal.scm import build_rtb_dag, estimate_causal_effect, run_refutation_tests
from src.features.engineering import load_feature_splits

PROJECT = Path(__file__).resolve().parents[2]
FAIR = PROJECT / "data/ipinyou/prediction/features_fair"
PRED = PROJECT / "results/models/escm2wc_dr_fair_posw/escm2wc_dr_test_predictions.npz"
OUT = PROJECT / "results/stage_a/scm_fair.json"


def _num(x):
    try:
        return float(x)
    except Exception:
        return str(x)


def main() -> None:
    _, _, test_df, _ = load_feature_splits(
        str(FAIR), columns=["bidprice", "win", "click", "payprice", "slotprice",
                            "adexchange", "region", "advertiser"])
    p_ctr = np.asarray(np.load(PRED)["p_ctr"], dtype=np.float64)
    assert len(p_ctr) == len(test_df), (len(p_ctr), len(test_df))
    # drop rows with NaN/inf in any SCM column (a few rows have NaN adexchange/region) — keep df+preds aligned
    mask = np.isfinite(p_ctr)
    for c in ["bidprice", "win", "click", "payprice", "slotprice", "adexchange", "region", "advertiser"]:
        mask &= np.isfinite(pd.to_numeric(test_df[c], errors="coerce").values)
    test_df = test_df.loc[mask].reset_index(drop=True)
    p_ctr = p_ctr[mask]
    print(f"clean rows: {len(test_df):,} (dropped {int((~mask).sum()):,})")
    preds = {"p_ctr": p_ctr}
    cfg = ValueConfig()

    out = {"_meta": {
        "split": "features_fair (per-advertiser temporal)", "n_rows": int(len(test_df)),
        "model": "escm2wc_dr (all-bids p_ctr)", "estimator": "DoWhy backdoor.linear_regression",
        "framing": "EXPLORATORY / hypothesis-generating — NOT an identified causal claim. "
                   "Flat-bid logging + won-only (lost inventory censored) make bid→surplus "
                   "non-identified; refutation tests are robustness diagnostics only."}}

    for outcome in ("surplus", "win"):
        dag = build_rtb_dag(treatment="bid", outcome=outcome)
        est = estimate_causal_effect(test_df, dag, preds=preds, value_config=cfg, subsample_n=500_000)
        ed = est._asdict() if hasattr(est, "_asdict") else dict(est.__dict__)
        ref = run_refutation_tests(test_df, dag, preds=preds, value_config=cfg, subsample_n=200_000)
        rd = ref._asdict() if hasattr(ref, "_asdict") else dict(ref.__dict__)
        out[f"bid_to_{outcome}"] = {
            "estimate": {k: _num(v) for k, v in ed.items() if k not in ("dag",)},
            "refutation": json.loads(json.dumps(rd, default=_num)),
        }
        print(f"bid→{outcome}: estimate keys {list(ed.keys())}")

    out["dag_dot"] = build_rtb_dag(treatment="bid", outcome="surplus").graph_dot
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
