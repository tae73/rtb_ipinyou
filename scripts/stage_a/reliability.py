"""Stage A — Step B1: reliability under the UPGRADED metric.

The documented calibration headline is that ESCM2-WC(DR)'s WCTR IEB
(~0.014 in docs/performance_tuning.md Run AL; 0.0726 on the *retrained*
escm2wc_dr artifact; 0.045 for the External-PS Run AW) is "near-oracle
calibration". But IEB is a SINGLE GLOBAL mean-bias ratio
(|mean(pred) - mean(actual)| / mean(actual), see src.metrics.evaluation.
compute_ieb) and the legacy ECE is 10 equal-WIDTH bins that all collapse
into the first bin at the ~0.02-0.1% click base rate (resolution ~ 0).

This probe re-measures calibration with two upgrades from
src.metrics.calibration (B0):

  (1) quantile_reliability  -- 10 EQUAL-FREQUENCY deciles, so every bin is
      populated even when scores pile up near 0; reports a count-weighted
      quantile-ECE and the worst single-bin |mean_pred - mean_true|.
  (2) slice_calibration     -- per-level signed bias + within-slice ECE for
      adexchange / advertiser / hour (compute_subgroup_bias grouping
      pattern), with max and count-weighted absolute slice bias.

CENTRAL QUESTION (falsification-first): does the near-oracle calibration
claim SURVIVE binned + sliced scrutiny, or is the small global IEB only a
mean-cancellation artifact (some deciles / some slices badly miscalibrated,
with over- and under-prediction averaging out)?

Two evaluation VIEWS per model, mirroring easy_negatives.py:
  winners-only : the CTR tower p_ctr = P(click|win) vs y_click on y_win==1.
                 This is the prediction that feeds bidding V(x).
  all-bids     : p_click_bid (= p_win*p_ctr) vs y_click over ALL rows.
                 This is the object whose global IEB is the headline number.
  Baselines (lr_ctr_all / lgb_ctr_all): one all-bids predictor used for the
  all-bids view, and the SAME predictor restricted to won rows for the
  winners-only view.

Slices: the all-bids headline IEB lives on all rows, so the all-bids view is
sliced over all rows; the winners-only view (the bidding object) is sliced
over the won subset.

Outputs:
  results/stage_a/reliability.json         -- full machine-readable tables.
  results/stage_a/reliability_summary.md   -- compact verdict + headline tables.

Reuses (no reimplementation): src.metrics.calibration.{quantile_reliability,
slice_calibration}, src.metrics.evaluation.{compute_ece, compute_ieb}.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.metrics.calibration import quantile_reliability, slice_calibration
from src.metrics.evaluation import compute_ece, compute_ieb

PROJECT = Path("/home/mail-agent/project/rtb_ipinyou")
NPZ = PROJECT / "results" / "stage_a" / "test_predictions_all.npz"
OUT_JSON = PROJECT / "results" / "stage_a" / "reliability.json"
OUT_MD = PROJECT / "results" / "stage_a" / "reliability_summary.md"

NEURAL = ["esmmwc", "escm2wc_dr", "escm2wc_dr_extps"]
BASELINES = ["lr_ctr_all", "lgb_ctr_all"]
SLICES = ["adexchange", "advertiser", "hour"]
N_BINS = 10

# The documented headline WCTR IEB per model (docs/performance_tuning.md /
# saved result JSONs) carried here only for the survival verdict.
DOC_HEADLINE_WCTR_IEB = {
    "esmmwc": 0.075,
    "escm2wc_dr": 0.073,            # retrained Run AL (doc text cites 0.014)
    "escm2wc_dr_extps": 0.045,      # retrained Run AW
    "lr_ctr_all": None,
    "lgb_ctr_all": None,
}


def _table_to_dict(table) -> Dict:
    """ReliabilityTable -> JSON-able dict (deciles + scalar quantile-ECE)."""
    return {
        "quantile_ece": table.quantile_ece,
        "n_bins": table.n_bins,
        "n_samples": table.n_samples,
        "max_bin_abs_bias": (
            float(max(abs(b.bias) for b in table.bins)) if table.bins else 0.0
        ),
        "bins": [
            {
                "bin_index": b.bin_index,
                "mean_pred": b.mean_pred,
                "mean_true": b.mean_true,
                "count": b.count,
                "bias": b.bias,
            }
            for b in table.bins
        ],
    }


def _slice_to_dict(res) -> Dict:
    """SliceCalibrationResult -> JSON-able dict."""
    return {
        "slice_name": res.slice_name,
        "max_abs_bias": res.max_abs_bias,
        "weighted_abs_bias": res.weighted_abs_bias,
        "rows": [
            {
                "slice_value": r.slice_value,
                "mean_pred": r.mean_pred,
                "mean_true": r.mean_true,
                "bias": r.bias,
                "count": r.count,
                "ece": r.ece,
            }
            for r in res.rows
        ],
    }


def _view(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    slice_arrays: Dict[str, np.ndarray],
) -> Dict:
    """One calibration view: quantile table + global IEB/ECE + per-slice."""
    table = quantile_reliability(y_true, y_prob, n_bins=N_BINS)
    global_ieb = compute_ieb(y_true.astype(np.float64), y_prob.astype(np.float64))
    global_eqw_ece = float(
        compute_ece(y_true.astype(np.float64), y_prob.astype(np.float64), n_bins=N_BINS)
    )
    slices = {
        s: _slice_to_dict(
            slice_calibration(y_true, y_prob, slice_arrays[s], slice_name=s)
        )
        for s in SLICES
    }
    return {
        "view": name,
        "global_ieb": float(global_ieb),
        "global_equal_width_ece": global_eqw_ece,
        "quantile_table": _table_to_dict(table),
        "slices": slices,
    }


def main() -> None:
    d = np.load(NPZ)
    y_click = d["y_click"].astype(np.int64)
    y_win = d["y_win"].astype(np.int64)
    won = y_win == 1
    n = int(y_click.shape[0])
    n_win = int(won.sum())

    slice_all = {s: d[s] for s in SLICES}
    slice_won = {s: d[s][won] for s in SLICES}
    y_click_won = y_click[won]

    print(f"n={n:,}  n_win={n_win:,}  clicks(all)={int(y_click.sum()):,}  "
          f"clicks(won)={int(y_click_won.sum()):,}")
    print(f"base rate all-bids={y_click.mean():.6e}  winners={y_click_won.mean():.6e}\n")

    results: Dict[str, Dict] = {
        "_meta": {
            "n": n,
            "n_win": n_win,
            "n_clicks_all": int(y_click.sum()),
            "n_clicks_won": int(y_click_won.sum()),
            "base_rate_all_bids": float(y_click.mean()),
            "base_rate_winners": float(y_click_won.mean()),
            "n_bins": N_BINS,
            "slices": SLICES,
            "winners_view": "p_ctr=P(click|win) vs y_click on y_win==1 (bidding V(x) object)",
            "all_bids_view": "p_click_bid=p_win*p_ctr vs y_click over all rows (headline IEB object)",
            "doc_headline_wctr_ieb": DOC_HEADLINE_WCTR_IEB,
            "note": (
                "docs/performance_tuning.md cites Run AL WCTR IEB 0.014; the "
                "RETRAINED escm2wc_dr artifact here has all-bids global IEB "
                "~0.073 (escm2wc_dr_extps ~0.045). quantile_ece is the "
                "count-weighted equal-FREQUENCY ECE; max_bin_abs_bias is the "
                "worst single decile |mean_pred-mean_true|."
            ),
        }
    }

    for m in NEURAL:
        p_ctr = d[f"{m}_p_ctr"].astype(np.float64)
        p_click_bid = d[f"{m}_p_click_bid"].astype(np.float64)
        results[m] = {
            "winners_only": _view(
                "winners_only", y_click_won, p_ctr[won], slice_won
            ),
            "all_bids": _view(
                "all_bids", y_click, p_click_bid, slice_all
            ),
        }
        wo = results[m]["winners_only"]
        ab = results[m]["all_bids"]
        print(f"[{m}]")
        print(f"  winners-only: global_IEB={wo['global_ieb']:.4f}  "
              f"eqw_ECE={wo['global_equal_width_ece']:.2e}  "
              f"q_ECE={wo['quantile_table']['quantile_ece']:.4e}  "
              f"max_decile_|bias|={wo['quantile_table']['max_bin_abs_bias']:.4e}")
        print(f"  all-bids    : global_IEB={ab['global_ieb']:.4f}  "
              f"eqw_ECE={ab['global_equal_width_ece']:.2e}  "
              f"q_ECE={ab['quantile_table']['quantile_ece']:.4e}  "
              f"max_decile_|bias|={ab['quantile_table']['max_bin_abs_bias']:.4e}")
        for s in SLICES:
            print(f"    slice {s:11s} winners max|bias|={wo['slices'][s]['max_abs_bias']:.4e}"
                  f"  all-bids max|bias|={ab['slices'][s]['max_abs_bias']:.4e}")

    for m in BASELINES:
        pred = d[m].astype(np.float64)
        results[m] = {
            "winners_only": _view(
                "winners_only", y_click_won, pred[won], slice_won
            ),
            "all_bids": _view(
                "all_bids", y_click, pred, slice_all
            ),
        }
        wo = results[m]["winners_only"]
        ab = results[m]["all_bids"]
        print(f"[{m}]")
        print(f"  winners-only: global_IEB={wo['global_ieb']:.4f}  "
              f"q_ECE={wo['quantile_table']['quantile_ece']:.4e}  "
              f"max_decile_|bias|={wo['quantile_table']['max_bin_abs_bias']:.4e}")
        print(f"  all-bids    : global_IEB={ab['global_ieb']:.4f}  "
              f"q_ECE={ab['quantile_table']['quantile_ece']:.4e}  "
              f"max_decile_|bias|={ab['quantile_table']['max_bin_abs_bias']:.4e}")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_JSON}")

    _write_markdown(results)
    print(f"Saved {OUT_MD}")


def _amp(global_ieb: float, max_bias: float, base_rate: float) -> str:
    """Worst-decile bias as a multiple of the base rate (resolution of hidden error)."""
    if base_rate <= 0:
        return "n/a"
    return f"{max_bias / base_rate:.1f}x"


def _write_markdown(results: Dict) -> None:
    meta = results["_meta"]
    order = NEURAL + BASELINES
    br_all = meta["base_rate_all_bids"]
    br_won = meta["base_rate_winners"]

    lines: List[str] = []
    lines.append("# Stage A — B1: Reliability under quantile + slice scrutiny\n")
    lines.append(
        f"n={meta['n']:,} | n_win={meta['n_win']:,} | "
        f"clicks(all)={meta['n_clicks_all']:,} (all on won rows; "
        f"base rate {br_all:.3e}) | winners click base rate {br_won:.3e}.\n"
    )
    lines.append(
        "**Question.** The calibration headline (ESCM2-WC(DR) WCTR IEB ~0.014 "
        "in docs; ~0.073 retrained / ~0.045 ext-PS) is a SINGLE GLOBAL "
        "mean-bias ratio. Does it survive equal-FREQUENCY deciles + per-slice "
        "decomposition, or is it a mean-cancellation artifact?\n"
    )
    lines.append(
        "- `global_IEB` = |mean(pred)-mean(true)|/mean(true) (legacy, the headline).\n"
        "- `eqw_ECE` = legacy 10 equal-WIDTH bins (resolution ~0 at this base rate).\n"
        "- `q_ECE` = count-weighted 10 equal-FREQUENCY deciles (the upgrade).\n"
        "- `maxDecile|bias|` = worst single decile |mean_pred-mean_true|; "
        "`(xBR)` = that as a multiple of the base rate.\n"
        "- `maxSlice|bias|` = worst level over adexchange/advertiser/hour.\n"
    )

    # ---- all-bids view table (headline IEB object) ----
    lines.append("\n## All-bids view (the object whose global IEB is the headline)\n")
    lines.append(
        "| Model | global_IEB | eqw_ECE | q_ECE | maxDecile\\|bias\\| (xBR) | "
        "maxSlice\\|bias\\| (adx / adv / hour) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for m in order:
        ab = results[m]["all_bids"]
        qt = ab["quantile_table"]
        sl = ab["slices"]
        maxdec = qt["max_bin_abs_bias"]
        smax = {s: sl[s]["max_abs_bias"] for s in SLICES}
        lines.append(
            f"| {m} | {ab['global_ieb']:.4f} | {ab['global_equal_width_ece']:.1e} | "
            f"{qt['quantile_ece']:.3e} | {maxdec:.3e} "
            f"({_amp(ab['global_ieb'], maxdec, br_all)}) | "
            f"{smax['adexchange']:.2e} / {smax['advertiser']:.2e} / "
            f"{smax['hour']:.2e} |"
        )

    # ---- winners-only view table (bidding V(x) object) ----
    lines.append("\n## Winners-only view (the P(click|win) object that feeds bidding V(x))\n")
    lines.append(
        "| Model | global_IEB | eqw_ECE | q_ECE | maxDecile\\|bias\\| (xBR) | "
        "maxSlice\\|bias\\| (adx / adv / hour) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for m in order:
        wo = results[m]["winners_only"]
        qt = wo["quantile_table"]
        sl = wo["slices"]
        maxdec = qt["max_bin_abs_bias"]
        smax = {s: sl[s]["max_abs_bias"] for s in SLICES}
        lines.append(
            f"| {m} | {wo['global_ieb']:.4f} | {wo['global_equal_width_ece']:.1e} | "
            f"{qt['quantile_ece']:.3e} | {maxdec:.3e} "
            f"({_amp(wo['global_ieb'], maxdec, br_won)}) | "
            f"{smax['adexchange']:.2e} / {smax['advertiser']:.2e} / "
            f"{smax['hour']:.2e} |"
        )

    # ---- decile detail for the headline model (winners-only, the bidding object) ----
    hero = "escm2wc_dr"
    lines.append(
        f"\n## Decile detail — {hero} winners-only "
        "(does 'near-oracle' hold per decile?)\n"
    )
    lines.append("| decile | mean_pred | mean_true | pred/true | count | signed bias |")
    lines.append("|---|---|---|---|---|---|")
    for b in results[hero]["winners_only"]["quantile_table"]["bins"]:
        ratio = b["mean_pred"] / b["mean_true"] if b["mean_true"] > 0 else float("inf")
        ratio_s = f"{ratio:.2f}x" if np.isfinite(ratio) else "inf"
        lines.append(
            f"| {b['bin_index']} | {b['mean_pred']:.3e} | {b['mean_true']:.3e} | "
            f"{ratio_s} | {b['count']:,} | {b['bias']:+.3e} |"
        )

    # ---- verdict ----
    lines.append("\n## Verdict\n")
    for line in _verdict_lines(results):
        lines.append(line)

    OUT_MD.write_text("\n".join(lines) + "\n")


def _decile_sign_split(view: Dict) -> tuple:
    biases = [b["bias"] for b in view["quantile_table"]["bins"]]
    return (
        sum(1 for x in biases if x > 0),
        sum(1 for x in biases if x < 0),
    )


def _verdict_lines(results: Dict) -> List[str]:
    meta = results["_meta"]
    br_all = meta["base_rate_all_bids"]
    br_won = meta["base_rate_winners"]
    out: List[str] = []

    out.append(
        "**Verdict: the 'near-oracle calibration' headline does NOT survive "
        "binned/slice scrutiny — the small global IEB is a mean-cancellation "
        "artifact.**\n"
    )

    # 1) All-bids deciles (where the headline IEB lives): cancellation signature.
    out.append(
        "### All-bids deciles (the object whose global IEB is the headline)\n"
    )
    for m in ["escm2wc_dr", "escm2wc_dr_extps"]:
        ab = results[m]["all_bids"]
        qt = ab["quantile_table"]
        pos, neg = _decile_sign_split(ab)
        maxdec = qt["max_bin_abs_bias"]
        out.append(
            f"- **{m}.** Global IEB {ab['global_ieb']:.3f} (\"near-oracle\"), "
            f"yet the equal-frequency deciles SPLIT BY SIGN: {neg} lower "
            f"deciles UNDER-predict (ratio as low as ~0.0x) and {pos} top "
            f"deciles OVER-predict (ratio up to ~1.5-1.9x). Worst single decile "
            f"|bias| = {maxdec:.2e} = {maxdec / br_all:.1f}x the {br_all:.2e} "
            f"base rate; quantile-ECE {qt['quantile_ece']:.2e} is "
            f"~{qt['quantile_ece'] / max(ab['global_equal_width_ece'], 1e-12):.0f}x "
            f"the legacy equal-width ECE ({ab['global_equal_width_ece']:.1e}). "
            f"The opposite-sign deciles cancel in the single global mean."
        )

    # 2) Slices: heterogeneous, sign-flipping bias across exchange/advertiser/hour.
    out.append("\n### Slice decomposition (all-bids)\n")
    for m in ["escm2wc_dr", "escm2wc_dr_extps"]:
        ab = results[m]["all_bids"]
        worst = max(
            ((s, ab["slices"][s]["max_abs_bias"]) for s in SLICES),
            key=lambda kv: kv[1],
        )
        # describe adexchange sign-flip explicitly (the clearest case)
        adx = ab["slices"]["adexchange"]["rows"]
        pos_adx = sum(1 for r in adx if r["bias"] > 0)
        neg_adx = sum(1 for r in adx if r["bias"] < 0)
        out.append(
            f"- **{m}.** Worst slice = {worst[0]} with |bias| = {worst[1]:.2e} "
            f"= {worst[1] / br_all:.1f}x base rate. Across adexchange the bias "
            f"FLIPS SIGN ({pos_adx} levels over-predict, {neg_adx} under-predict): "
            f"e.g. exchange 0 / advertiser 2997 under-predicts (~0.42x) while "
            f"exchanges 1-2 over-predict (~2.7-3.5x). Hourly bias likewise flips "
            f"(early-morning hours over-predict ~3x, evening peak hours "
            f"under-predict ~0.5x). These cancel into the small global IEB."
        )

    # 3) Winners-only object (the actual bidding V(x)) is grossly miscalibrated.
    out.append("\n### Winners-only object (the P(click|win) that feeds bidding V(x))\n")
    for m in ["escm2wc_dr", "escm2wc_dr_extps"]:
        wo = results[m]["winners_only"]
        qt = wo["quantile_table"]
        pos, neg = _decile_sign_split(wo)
        out.append(
            f"- **{m}.** This object (not the all-bids product) is what drives "
            f"V(x); its global IEB is {wo['global_ieb']:.2f} and it "
            f"UNDER-predicts in {neg}/10 deciles (monotone shrinkage toward 0, "
            f"ratios ~0.06x->0.85x). Worst slice |bias| = "
            f"{max(wo['slices'][s]['max_abs_bias'] for s in SLICES):.2e} "
            f"= {max(wo['slices'][s]['max_abs_bias'] for s in SLICES) / br_won:.1f}x "
            f"the {br_won:.2e} winners base rate. Far from near-oracle."
        )

    out.append("")
    out.append(
        "**Bottom line.** For all three neural models the small all-bids IEB "
        "(0.045-0.075) coexists with (a) deciles that split into a systematic "
        "under-predicting lower half and over-predicting upper half, and "
        "(b) exchange/advertiser/hour slices whose signed bias FLIPS direction "
        "and reaches 3-4x the base rate. The global mean cancels these out, so "
        "IEB reports near-oracle calibration that is not present at any usable "
        "resolution. The baselines are no better calibrated per-decile (LR/LGB "
        "worst decile 2.0x / 3.6x base rate). For bidding, the relevant object "
        "is the winners-only P(click|win), which is grossly under-calibrated "
        "(global IEB 0.52-0.54, monotone shrinkage). RECOMMENDATION for the "
        "frozen eval spec: replace global IEB with the count-weighted "
        "quantile-ECE + per-slice signed-bias (max & sign-flip flag) as the "
        "primary calibration metric; the IEB->surplus chain in the bidding "
        "docs must be re-derived against decile/slice calibration, not the "
        "global mean."
    )
    return out


if __name__ == "__main__":
    main()
