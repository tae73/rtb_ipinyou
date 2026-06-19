# MANIFEST — canonical map

정본 수치는 아래 witness JSON뿐. 문서·figure는 거기서 재생성, 직접 수정 금지.

## Claims → witnesses → figures
| claim | headline (canonical) | witness (canonical JSON) | code | figure |
|---|---|---|---|---|
| de-risk GO (debiasing > recalibration on full inventory) | GO | `witnesses/probe_debiasing_bidding_value.json` | `witnesses/probe_debiasing_bidding_value.py` | — |
| **C1** within-capacity competitor-strength asymmetry | IPW +4.4pp vs weak / −1.9pp vs strong; capacity gap +26.3pp separate; DR −2.6pp (lost to IPW) | `witnesses/phase_diagram.json` | `witnesses/phase_diagram.py` | `witnesses/figures/fig_phase_diagram.png` |
| **C2** recalibration trap (over-bids marginal inventory) | surplus 4.31M→3.26M (recal) vs 5.92M (IPW); 5/5 seeds | `witnesses/recal_trap.json` | `witnesses/recal_trap.py` | `witnesses/figures/fig_recal_trap.png` |
| **Neural anchor** — REAL iPinYou features + the REAL ESCM²-WC (Flax) | −47pp over-bidding was a **censoring bug**; fixed (`click·win`) → ESCM²-WC helps truthful bidding **+7.2pp** (+11 w/ calibration). 2×2 test: selection-aware IPW beats naive **only at weak selection** (ESS-gated) — *accuracy ≠ bidding value*. pre-fix frozen in `_meta` | `witnesses/neural_anchor.json` | `witnesses/neural_anchor.py` | `witnesses/figures/fig_neural_anchor.png` |

> **Edge is WITHIN-CAPACITY** (regret(biased) − regret(debiased) at the same model class). `capacity_gap_pp`
> in the JSON is the GBM>LR model-class effect, reported separately — it is **not** debiasing. Primary
> debiaser = IPW; DR is reported as an honest negative (it did not beat IPW here).

## Documents
| file | role |
|---|---|
| `README.md` / `README.ko.md` | flagship front (EN / KO twin) |
| `concept.md` | 1-pager (motivation → claim) |
| `methods.md` | testbed · models · results (numbers = witness JSON) |
| `review.md` | honest scoping · prior-art positioning · limitations · path-to-full |
| `old/` | **Foundation** — the iPinYou debiasing portfolio (real-world anchor: robust vs LR, not vs LGB) |

## Reproduce
`repro/` imports the canonical JSON unmodified and re-asserts the headline numbers (green harness).
Regenerate from scratch: `python witnesses/phase_diagram.py && python witnesses/recal_trap.py &&
python witnesses/figures/make_figures.py`.

## Real-world anchor (reused, unmodified)
`old/results/stage_a/*.json` (iPinYou fair-split: neural−LR robust, neural−LGB not robust, I²=0.82),
`old/results/market_price_cdf/` (market calibration). See [`old/`](old/) for the full study.
