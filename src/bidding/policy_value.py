"""Full-inventory policy-value projection for second-price RTB bidding.

Estimates the value V(π) of a bidding policy π (context → bid) over ALL bids (won + lost), not just
the won subset. Under SECOND-PRICE auctions the surplus is **exactly observable** wherever the policy
bid is determinable from the logged market price, and only requires a market model on the censored
increment where the policy bids ABOVE the logged bid:

  market price m_i:  won  → m_i = payprice_i (observed exactly)
                     lost → m_i > logged_bid_i (right-censored)
  surplus_i(b) = (click_i·CPC − m_i)·1(m_i ≤ b_i)        [second-price: pay the market price]

  • won row:                 s_i = (click_i·CPC − payprice_i)·1(payprice_i ≤ b_i)   → EXACT
  • lost row, b_i ≤ logged:  m_i > logged ≥ b_i → s_i = 0                           → EXACT
  • lost row, b_i > logged:  win iff logged < m_i ≤ b_i (censored region)           → MODELED (F,pCTR)

So V(π) = **V_exact** (observed) + **V_model** (extrapolated increment, only where π bids above the
logged flat bid). For shaded/truthful policies (bid ≈ V ≈ 160 < logged 227–300) V is almost entirely
EXACT — the won-only limitation barely binds, and the result is honest. The market model `F(b|x)` is
needed only for the incremental aggressive-bid region (`scripts/stage_a/policy_value.py`).

This is NOT off-policy evaluation (deterministic flat logging → no propensity). It is a structural,
mostly-observable policy-value projection; the modeled component is reported separately and bounded.

Reuses `src/win_rate/survival.py` (the F(b|x) engine) and `src/bidding/simulator.py` bootstrap CIs.
"""

from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Market model: per-segment F(b|x) = P(market_price <= b | x), from KM survival
# ---------------------------------------------------------------------------

class MarketModel:
    """Per-segment market-price CDF F(b|x) with win-prob and truncated-spend lookups.

    Each segment holds a shared ``price_grid`` and ``cdf`` (monotone, F(p)=P(market<=p)).
    ``win_prob(seg, b)`` = F(b|seg); ``spend_le(seg, b)`` = ∫₀^b m·dF(m|seg) (expected second-price
    spend for the part of the market below b — already weighted by win prob). A ``__default__``
    segment is the fallback for unseen/under-populated segments.
    """

    def __init__(self, grid: np.ndarray, cdf_by_seg: Dict[str, np.ndarray], default_key: str = "__default__"):
        self.grid = np.asarray(grid, dtype=np.float64)
        self.default_key = default_key
        # precompute cumulative truncated spend ∫₀^b m dF per segment on the grid
        m_mid = 0.5 * (self.grid[:-1] + self.grid[1:])
        self._cdf: Dict[str, np.ndarray] = {}
        self._cumspend: Dict[str, np.ndarray] = {}
        for k, cdf in cdf_by_seg.items():
            cdf = np.asarray(cdf, dtype=np.float64)
            self._cdf[k] = cdf
            dF = np.diff(cdf)
            cum = np.concatenate([[0.0], np.cumsum(m_mid * dF)])
            self._cumspend[k] = cum

    def _key(self, seg: str) -> str:
        return seg if seg in self._cdf else self.default_key

    def win_prob(self, seg_keys: np.ndarray, bids: np.ndarray) -> np.ndarray:
        """F(b|seg) per row (vectorized over rows sharing segments)."""
        out = np.empty(len(bids), dtype=np.float64)
        seg_keys = np.asarray(seg_keys)
        for k in np.unique(seg_keys):
            kk = self._key(str(k))
            m = seg_keys == k
            out[m] = np.interp(bids[m], self.grid, self._cdf[kk], left=0.0, right=self._cdf[kk][-1])
        return out

    def spend_le(self, seg_keys: np.ndarray, bids: np.ndarray) -> np.ndarray:
        """∫₀^b m·dF(m|seg) per row (expected second-price spend on market below b)."""
        out = np.empty(len(bids), dtype=np.float64)
        seg_keys = np.asarray(seg_keys)
        for k in np.unique(seg_keys):
            kk = self._key(str(k))
            m = seg_keys == k
            out[m] = np.interp(bids[m], self.grid, self._cumspend[kk],
                               left=0.0, right=self._cumspend[kk][-1])
        return out


# ---------------------------------------------------------------------------
# Policy-value projection
# ---------------------------------------------------------------------------

class PolicyValueResult(NamedTuple):
    total: float              # V(π) = V_exact + V_model
    v_exact: float            # observable component (won rows + exact-lost-zeros)
    v_model: float            # extrapolated increment (lost rows, policy bids above logged)
    frac_value_modeled: float # |v_model| / (|v_exact| + |v_model|)
    n_extrapolated: int       # # rows needing the market model
    frac_rows_extrapolated: float
    n_winnable_exact: int     # won rows the policy re-wins (payprice <= policy bid)
    s_vec: np.ndarray         # per-row surplus contribution (for bootstrap CI)


def project_policy_value(
    pctr: np.ndarray,
    policy_bid: np.ndarray,
    logged_bid: np.ndarray,
    win: np.ndarray,
    payprice: np.ndarray,
    click: np.ndarray,
    seg_keys: np.ndarray,
    market_model: Optional[MarketModel],
    cpc: float = 200_000.0,
) -> PolicyValueResult:
    """Second-price full-inventory value of policy bids ``policy_bid`` over all rows.

    EXACT where determinable from the log; MODELED only on lost rows the policy bids above the logged
    bid (requires ``market_model``). Returns per-row ``s_vec`` for cluster-bootstrap CIs.

    Args:
        pctr: P(click|win, x) per row (the value model, ideally recalibrated).
        policy_bid: π(x) — the policy's bid per row.
        logged_bid: the original logged bid per row (defines the censoring threshold).
        win, payprice, click: logged outcomes (payprice/click meaningful on won rows).
        seg_keys: market-segment key per row (for F(b|x)).
        market_model: MarketModel for the extrapolated increment (may be None → model term = 0,
            i.e. a strict observable lower bound).
        cpc: CPC target (value per click).
    """
    pctr = np.asarray(pctr, np.float64)
    b = np.asarray(policy_bid, np.float64)
    lb = np.asarray(logged_bid, np.float64)
    win = np.asarray(win).astype(bool)
    pay = np.asarray(payprice, np.float64)
    clk = np.asarray(click, np.float64)
    n = len(b)

    s = np.zeros(n, dtype=np.float64)

    # (1) WON rows: exact second-price surplus = (click·CPC − payprice) if policy re-wins (pay≤bid).
    rewin = win & (pay <= b)
    s[rewin] = clk[rewin] * cpc - pay[rewin]
    n_winnable_exact = int(rewin.sum())

    # (2) LOST rows with policy bid ≤ logged bid → still lose → s = 0 (already zero). EXACT.
    # (3) LOST rows with policy bid > logged bid → censored increment → MODELED.
    extra = (~win) & (b > lb)
    n_extra = int(extra.sum())
    if n_extra > 0 and market_model is not None:
        sk = seg_keys[extra]
        F_b = market_model.win_prob(sk, b[extra])
        F_lb = market_model.win_prob(sk, lb[extra])
        spend_b = market_model.spend_le(sk, b[extra])
        spend_lb = market_model.spend_le(sk, lb[extra])
        # expected surplus over the censored window (logged_bid, policy_bid]
        s[extra] = pctr[extra] * cpc * (F_b - F_lb) - (spend_b - spend_lb)

    v_exact = float(s[~extra].sum())
    v_model = float(s[extra].sum())
    denom = abs(v_exact) + abs(v_model)
    return PolicyValueResult(
        total=v_exact + v_model,
        v_exact=v_exact,
        v_model=v_model,
        frac_value_modeled=(abs(v_model) / denom) if denom > 0 else 0.0,
        n_extrapolated=n_extra,
        frac_rows_extrapolated=n_extra / n if n else 0.0,
        n_winnable_exact=n_winnable_exact,
        s_vec=s,
    )


def contextual_optimal_bid(
    values: np.ndarray,
    seg_keys: np.ndarray,
    market_model: MarketModel,
    n_candidates: int = 300,
    max_bid: float = 300.0,
) -> np.ndarray:
    """Second-price contextual optimal bid: argmax_b [V·F(b|x) − ∫₀^b m·dF(m|x)].

    Uses the market model's win-prob and truncated-spend on a shared candidate grid (≤ max_bid =
    the action-support ceiling). Vectorized per segment.
    """
    values = np.asarray(values, np.float64)
    seg_keys = np.asarray(seg_keys)
    grid = np.linspace(1.0, max_bid, n_candidates)
    out = np.empty(len(values), np.float64)
    for k in np.unique(seg_keys):
        m = seg_keys == k
        kk = market_model._key(str(k))
        F = np.interp(grid, market_model.grid, market_model._cdf[kk], left=0.0, right=market_model._cdf[kk][-1])
        spend = np.interp(grid, market_model.grid, market_model._cumspend[kk],
                          left=0.0, right=market_model._cumspend[kk][-1])
        # surplus matrix (rows in segment × candidates): V_i·F(b) − spend(b)
        surplus = values[m][:, None] * F[None, :] - spend[None, :]
        out[m] = grid[np.argmax(surplus, axis=1)]
    return out
