"""Green harness — re-assert the headline numbers from the canonical witness JSONs (unmodified)."""
import json
from pathlib import Path

W = Path(__file__).resolve().parents[1] / "witnesses"


def main():
    pd = json.load(open(W / "phase_diagram.json"))
    s = pd["summary"]
    assert s["mean_edge_vs_linear_pp"] > 10, s            # robust vs linear baseline
    assert s["mean_edge_vs_gbm_pp"] < 2, s                # NOT robust vs strong GBM baseline
    assert s["mean_edge_vs_linear_pp"] > s["mean_edge_vs_gbm_pp"] + 1, s  # C1 asymmetry
    assert s["asymmetry_holds"] is True

    rt = json.load(open(W / "recal_trap.json"))["cases"]["linear_strong"]
    assert rt["biased_recal"]["mean_bid"] > rt["biased"]["mean_bid"], rt      # recal inflates the bid
    assert rt["biased_recal"]["won_surplus"] < rt["biased"]["won_surplus"], rt  # C2 trap: surplus drops
    assert rt["debiased"]["won_surplus"] > rt["biased"]["won_surplus"], rt   # debiasing recovers

    print("GREEN — C1 (edge vs linear %.1fpp > vs gbm %.1fpp) and C2 (recal surplus %.1fM < biased %.1fM < debiased %.1fM) hold." % (
        s["mean_edge_vs_linear_pp"], s["mean_edge_vs_gbm_pp"],
        rt["biased_recal"]["won_surplus"] / 1e6, rt["biased"]["won_surplus"] / 1e6, rt["debiased"]["won_surplus"] / 1e6))


if __name__ == "__main__":
    main()
