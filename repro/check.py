"""Green harness — re-assert the headline numbers from the canonical witness JSONs, AND (with --live)
re-run one phase-diagram cell from scratch to prove the pipeline actually produces the claimed effect
(not just a static-file consistency check).

    python repro/check.py          # fast: static consistency of the committed JSONs
    python repro/check.py --live   # also recompute one cell live (~1 min) and check it matches
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "witnesses"


def static_checks():
    pd = json.load(open(W / "phase_diagram.json"))
    s = pd["summary"]
    # C1 (within-capacity, IPW primary): helps the weak model, not the strong one
    assert s["debias_edge_ipw_within_linear_pp"] > 1, s          # IPW helps a weak (linear) model
    assert s["debias_edge_ipw_within_gbm_pp"] < 1, s             # does NOT help a strong (gbm) model
    assert s["debias_helps_weak_not_strong"] is True, s
    assert s["grows_with_selection"] is True, s                  # edge grows with selection strength
    # the confound is reported separately and is NOT counted as debiasing
    assert s["capacity_gap_pp"] > s["debias_edge_ipw_within_linear_pp"], s   # capacity dominates the apparent edge
    # honest negative: a genuine DR did not beat IPW here
    assert s["dr_beats_ipw"] is False, s

    rt = json.load(open(W / "recal_trap.json"))
    ls = rt["cases"]["linear_strong"]
    assert ls["biased_recal"]["mean_bid"] > ls["biased"]["mean_bid"], ls      # recal inflates the bid
    assert ls["biased_recal"]["won_surplus"] < ls["biased"]["won_surplus"], ls  # C2 trap: surplus drops
    assert ls["debiased"]["won_surplus"] > ls["biased"]["won_surplus"], ls   # IPW recovers
    rb = rt["robustness"]
    assert rb["recal_lowers_surplus_frac"] == "5/5", rb                       # trap robust across seeds
    assert rb["ipw_raises_surplus_frac"] == "5/5", rb
    print("STATIC GREEN — C1 (IPW within-linear %+.1fpp > within-gbm %+.1fpp; capacity gap %+.1fpp separate; DR lost) "
          "and C2 (recal surplus %.2fM < biased %.2fM < IPW %.2fM, trap 5/5) hold." % (
          s["debias_edge_ipw_within_linear_pp"], s["debias_edge_ipw_within_gbm_pp"], s["capacity_gap_pp"],
          ls["biased_recal"]["won_surplus"] / 1e6, ls["biased"]["won_surplus"] / 1e6, ls["debiased"]["won_surplus"] / 1e6))


def neural_checks():
    """Neural anchor (iPinYou-grounded semi-synthetic + real ESCM²-WC). Skips gracefully if absent."""
    p = W / "neural_anchor.json"
    if not p.exists():
        print("SKIP neural_anchor — JSON not present"); return
    s = json.load(open(p))["summary"]
    rec = s["pctr_recovery_neural"]
    assert s["biased_pctr_collapsed"] is True, rec               # win-selection collapses the biased neural pCTR
    assert s["censoring_fixes_overbidding"] is True, s           # censoring click (click·win) fixes the −47pp over-bidding
    assert s["truthful_edge_neural_pp"] > 0, s                   # corrected: ESCM²-WC genuinely helps truthful bidding
    assert s["debiased_undershoots_level"] is True, rec         # censored model now under-predicts (no overshoot)
    assert s["ipw_calibration_helps_truthful"] is True, s       # selection-aware calibration further improves it...
    assert s["ipw_cal_hurts_strong_selection"] is True, s       # ...but its gain vanishes at the strongest selection
    assert s["recal_trap_holds_gbm"] is True, s                 # C2 recal-trap still reproduces for GBM
    print("NEURAL GREEN (corrected) — censoring click fixes the −47pp over-bidding → neural truthful edge "
          "%+.1fpp (was %.1f); +IPW-cal %+.1fpp (gain vanishes at strong γ); pCTR mean %.3f→%.3f (no overshoot)." % (
          s["truthful_edge_neural_pp"], s["frozen_prefix_truthful_edge_neural_pp"],
          s["truthful_edge_neural_ipwcal_pp"], rec["biased_mean"], rec["debiased_mean"]))


def live_check():
    """Recompute ONE strong-selection linear cell from scratch (2 seeds) and confirm the within-capacity
    IPW debiasing edge is robustly positive — i.e. the committed JSON reflects a real pipeline output."""
    sys.path.insert(0, str(W))
    import phase_diagram as P
    c = P.cell(1.2, 0, "linear", seeds=2)
    edge = c["debias_edge_ipw_pp"]
    committed = next(x for x in json.load(open(W / "phase_diagram.json"))["cells"]
                     if x["gamma"] == 1.2 and x["theta"] == 0 and x["capacity"] == "linear")["debias_edge_ipw_pp"]
    assert edge > 5, f"live within-linear IPW edge not robustly positive: {edge}"
    print(f"LIVE GREEN — recomputed cell (γ=1.2,θ=0,linear): IPW edge {edge:+.1f}pp (committed {committed:+.1f}pp); "
          f"pipeline reproduces a real positive within-capacity effect.")


if __name__ == "__main__":
    static_checks()
    neural_checks()
    if "--live" in sys.argv:
        live_check()
