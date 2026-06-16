# MANIFEST — canonical map

정본 수치는 아래 witness JSON뿐. 문서·figure는 거기서 재생성, 직접 수정 금지.

## Claims → witnesses → figures
| claim | witness (canonical JSON) | code | figure |
|---|---|---|---|
| de-risk GO (debiasing > recalibration on full inventory) | `witnesses/probe_debiasing_bidding_value.json` | `witnesses/probe_debiasing_bidding_value.py` | — |
| **C1** competitor-strength asymmetry (robust vs LR, not vs GBM) | `witnesses/phase_diagram.json` | `witnesses/phase_diagram.py` | `witnesses/figures/fig_phase_diagram.png` |
| **C2** recalibration trap (over-bids marginal inventory) | `witnesses/recal_trap.json` | `witnesses/recal_trap.py` | `witnesses/figures/fig_recal_trap.png` |

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
