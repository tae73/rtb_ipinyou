"""Tests for src.bidding.policy_value — second-price full-inventory policy value.

Pins the EXACT/MODELED decomposition and the contextual optimal bid:
  - where the policy is determinable from the log (won rows, or lost rows bid <= logged), the
    estimate is EXACT (== ground truth) and needs no market model;
  - the modeled increment (lost rows bid above logged) matches the market model's expectation;
  - contextual optimal bid recovers the closed-form second-price optimum (b* = V for a uniform market).
"""

import numpy as np
import pytest

from src.bidding.policy_value import (
    MarketModel,
    PolicyValueResult,
    contextual_optimal_bid,
    project_policy_value,
)

CPC = 200_000.0
GRID = np.linspace(0.0, 300.0, 301)


def _const_market_model(c: float) -> MarketModel:
    """Deterministic market: price == c in every segment (F = step at c)."""
    cdf = (GRID >= c).astype(np.float64)
    return MarketModel(GRID, {"__default__": cdf})


def _uniform_market_model(M: float = 300.0) -> MarketModel:
    cdf = np.clip(GRID / M, 0.0, 1.0)
    return MarketModel(GRID, {"__default__": cdf})


def test_exact_when_policy_below_logged_no_model_needed():
    """Policy bids below the logged bid everywhere -> value is fully EXACT (market model unused)."""
    rng = np.random.default_rng(0)
    n = 5000
    logged = np.full(n, 280.0)
    market = rng.uniform(0, 400, n)          # true market price
    win = market <= logged
    payprice = np.where(win, market, 0.0)
    click = np.where(win & (rng.random(n) < 0.01), 1.0, 0.0)
    pctr = np.full(n, 0.01)
    policy = np.full(n, 90.0)                 # 90 < 280 logged → no extrapolation
    seg = np.array(["__default__"] * n)

    res = project_policy_value(pctr, policy, logged, win, payprice, click, seg,
                               market_model=None, cpc=CPC)
    assert isinstance(res, PolicyValueResult)
    assert res.n_extrapolated == 0 and res.v_model == 0.0
    # ground truth: surplus = (click*CPC - market)*(market <= policy)
    gt = ((click * CPC - market) * (market <= policy)).sum()
    assert res.total == pytest.approx(gt, rel=1e-9)
    # only re-wins the won rows with payprice <= 90
    assert res.n_winnable_exact == int((win & (payprice <= 90.0)).sum())


def test_lost_below_logged_is_exact_zero():
    """A lost row with policy bid <= logged bid contributes exactly 0 (still loses)."""
    res = project_policy_value(
        pctr=np.array([0.01]), policy_bid=np.array([100.0]), logged_bid=np.array([200.0]),
        win=np.array([0]), payprice=np.array([0.0]), click=np.array([0.0]),
        seg_keys=np.array(["__default__"]), market_model=_const_market_model(150.0), cpc=CPC)
    # market > 200 (lost), policy 100 < 200 → not extrapolated, surplus 0
    assert res.n_extrapolated == 0
    assert res.total == 0.0


def test_modeled_increment_matches_market_expectation():
    """Lost row, policy bids ABOVE logged into the censored region → modeled surplus = pctr*CPC - c."""
    c = 150.0
    # lost at logged=120 (<150 so we lost); policy bids 250 (>150) → wins the increment, pays c
    res = project_policy_value(
        pctr=np.array([0.02]), policy_bid=np.array([250.0]), logged_bid=np.array([120.0]),
        win=np.array([0]), payprice=np.array([0.0]), click=np.array([0.0]),
        seg_keys=np.array(["__default__"]), market_model=_const_market_model(c), cpc=CPC)
    assert res.n_extrapolated == 1
    # F(250)-F(120) = 1-0 = 1 ; spend(250)-spend(120) = c-0
    assert res.v_model == pytest.approx(0.02 * CPC - c, abs=2.0)  # grid discretization tol
    assert res.frac_value_modeled == pytest.approx(1.0)


def test_contextual_optimal_bid_uniform_market_is_truthful():
    """Uniform market F(b)=b/M -> second-price optimum b* = V (truthful), capped at max_bid."""
    mm = _uniform_market_model(300.0)
    V = np.array([50.0, 100.0, 250.0, 400.0])
    seg = np.array(["__default__"] * 4)
    b = contextual_optimal_bid(V, seg, mm, n_candidates=300, max_bid=300.0)
    # b* = min(V, 300) within grid resolution (~1.0)
    assert b[0] == pytest.approx(50.0, abs=2.0)
    assert b[1] == pytest.approx(100.0, abs=2.0)
    assert b[2] == pytest.approx(250.0, abs=2.0)
    assert b[3] == pytest.approx(300.0, abs=2.0)   # V>max_bid → clipped to ceiling


def test_market_model_winprob_and_spend():
    """MarketModel.win_prob = F(b); spend_le = ∫₀^b m dF (= c at the step for a constant market)."""
    mm = _const_market_model(70.0)
    seg = np.array(["a", "b"])  # unseen segs → default
    wp = mm.win_prob(seg, np.array([50.0, 100.0]))
    assert wp[0] == pytest.approx(0.0) and wp[1] == pytest.approx(1.0)
    sp = mm.spend_le(seg, np.array([50.0, 100.0]))
    assert sp[0] == pytest.approx(0.0) and sp[1] == pytest.approx(70.0, abs=1.0)


def test_policy_value_deterministic_s_vec_shape():
    rng = np.random.default_rng(1)
    n = 1000
    logged = np.full(n, 280.0); market = rng.uniform(0, 400, n)
    win = market <= logged
    res = project_policy_value(np.full(n, 0.01), np.full(n, 90.0), logged, win,
                               np.where(win, market, 0.0), np.zeros(n),
                               np.array(["__default__"] * n), None, CPC)
    assert res.s_vec.shape == (n,)
    assert res.total == pytest.approx(float(res.s_vec.sum()))
