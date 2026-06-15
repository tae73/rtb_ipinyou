"""Pure-numpy tests for src.bidding (value, shading, simulator).

Fast, deterministic, no JAX, no disk-data dependency. A synthetic monotone
market-price CDF F(p) = clip(p / 200, 0, 1) is used throughout because it has a
clean closed-form argmax for the first-price surplus (V - b) * F(b):

    surplus(b) = (V - b) * b / 200  =>  b* = V / 2   (for V <= 200)

Coverage:
  (a) optimal_bid_vectorized: grid argmax of (V-b)*F(b), b* ~ V/2, b <= V.
  (b) compute_impression_values: linear/monotone in pCTR.
  (c) run_auction_simulation / compute_simulation_metrics: win-accounting
      identities on a tiny synthetic auction.
  (d) load_exchange_cdfs / load_market_cdf / MarketCDF: basic behavior;
      disk artifacts tested only if results/market_price_cdf/ exists (skip).
"""

import os

import numpy as np
import pytest

from src.bidding.shading import (
    MarketCDF,
    _interpolate_cdf,
    linear_bid,
    load_exchange_cdfs,
    load_market_cdf,
    optimal_bid_vectorized,
    percentile_bid,
)
from src.bidding.simulator import (
    SurplusGapCI,
    cluster_bootstrap_surplus_gap,
    compute_simulation_metrics,
    paired_bootstrap_surplus_gap,
    run_auction_simulation,
)
from src.bidding.value import (
    ValueConfig,
    compute_impression_values,
    compute_impression_values_cpa,
)


SEED = 0

# Synthetic monotone CDF: F(p) = clip(p / 200, 0, 1) on a fine grid in [0, 300].
_PRICE_GRID = np.linspace(0.0, 300.0, 301)
_CDF = np.clip(_PRICE_GRID / 200.0, 0.0, 1.0)

_CDF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "market_price_cdf",
)


@pytest.fixture
def linear_cdf() -> MarketCDF:
    """Synthetic monotone market CDF F(p) = clip(p/200, 0, 1) (closed-form b*=V/2)."""
    return MarketCDF(
        price_grid=_PRICE_GRID.copy(),
        cdf=_CDF.copy(),
        median_price=100.0,
        source="synthetic_linear",
    )


# =============================================================================
# (a) optimal_bid_vectorized
# =============================================================================


def test_optimal_bid_is_grid_argmax_and_bounded(linear_cdf):
    """b* is the grid argmax of (V-b)*F(b), never exceeds V, surplus >= 0."""
    values = np.array([10.0, 50.0, 100.0, 180.0], dtype=np.float64)
    bids, surplus = optimal_bid_vectorized(
        values, linear_cdf, n_candidates=1000, min_bid=1.0, max_bid=300.0
    )

    assert bids.shape == values.shape
    assert surplus.shape == values.shape

    # Hard cap: bid never exceeds value (margin > 0 mask).
    assert np.all(bids <= values + 1e-9)
    # Bounds and non-negative surplus.
    assert np.all(bids >= 1.0)
    assert np.all(bids <= 300.0)
    assert np.all(surplus >= 0.0)

    # Brute-force the argmax of (V-b)*F(b) on the SAME grid the function uses
    # (linspace(min_bid, max_bid, n_candidates)) -> exact match. Then confirm
    # the returned surplus also dominates a denser grid up to grid resolution.
    same_grid = np.linspace(1.0, 300.0, 1000)
    f_same = _interpolate_cdf(same_grid, linear_cdf)
    grid_res = (300.0 - 1.0) / (1000 - 1)
    dense = np.linspace(1.0, 300.0, 6000)
    f_dense = _interpolate_cdf(dense, linear_cdf)
    for v, b, s in zip(values, bids, surplus):
        # Exact grid argmax: returned surplus == best over the function's own grid.
        margin = v - same_grid
        surp_same = np.where(margin > 0, margin * f_same, -np.inf)
        assert s >= surp_same.max() - 1e-9
        # vs a denser grid: surplus is a smooth concave function of b, so the
        # surplus gap between adjacent grid points is bounded ~ |dS/db| * grid_res.
        # Use a generous slack proportional to the grid spacing.
        margin_d = v - dense
        surp_dense = np.where(margin_d > 0, margin_d * f_dense, -np.inf)
        assert s >= surp_dense.max() - (v * grid_res)
        # The surplus recomputed at the returned bid matches the reported one.
        recomputed = max((v - b) * float(_interpolate_cdf(np.array([b]), linear_cdf)[0]), 0.0)
        assert abs(recomputed - s) < 1e-6


def test_optimal_bid_closed_form_half_value(linear_cdf):
    """For F(p)=p/200 the optimum is b* = V/2 (V <= 200), within grid resolution."""
    values = np.array([10.0, 50.0, 100.0, 250.0], dtype=np.float64)
    n_candidates = 1000
    bids, _ = optimal_bid_vectorized(
        values, linear_cdf, n_candidates=n_candidates, min_bid=1.0, max_bid=300.0
    )
    grid_res = (300.0 - 1.0) / (n_candidates - 1)

    # Closed form b* = V/2 holds while V <= 200 (interior optimum below F=1 region).
    for v, b in zip(values, bids):
        if v <= 200.0:
            assert abs(b - v / 2.0) <= grid_res + 1e-6
        else:
            # b <= V still required; optimum drifts but stays below V.
            assert b <= v + 1e-9


def test_optimal_bid_batching_is_invariant(linear_cdf):
    """Batching (batch_size) does not change the result."""
    rng = np.random.default_rng(SEED)
    values = rng.uniform(5.0, 250.0, size=257).astype(np.float64)

    bids_full, surplus_full = optimal_bid_vectorized(
        values, linear_cdf, n_candidates=500, batch_size=100_000
    )
    bids_small, surplus_small = optimal_bid_vectorized(
        values, linear_cdf, n_candidates=500, batch_size=16
    )
    assert np.array_equal(bids_full, bids_small)
    assert np.allclose(surplus_full, surplus_small, atol=1e-12)


def test_optimal_bid_never_exceeds_value_random(linear_cdf):
    """Across random values, the b <= V invariant holds (margin>0 mask)."""
    rng = np.random.default_rng(SEED + 1)
    values = rng.uniform(0.5, 300.0, size=512).astype(np.float64)
    bids, surplus = optimal_bid_vectorized(values, linear_cdf, n_candidates=400)
    assert np.all(bids <= values + 1e-9)
    assert np.all(surplus >= 0.0)


# =============================================================================
# (b) compute_impression_values: linear / monotone in pCTR
# =============================================================================


def test_value_is_exactly_linear_in_pctr():
    """CPC value == p_ctr * cpc_target exactly; zero pCTR -> zero value."""
    p_ctr = np.array([0.0, 0.001, 0.004, 0.01], dtype=np.float64)
    cfg = ValueConfig(goal_type="CPC", cpc_target=200_000.0)
    res = compute_impression_values(p_ctr, cfg)

    expected = p_ctr * cfg.cpc_target
    assert np.array_equal(res.values, expected)
    assert res.values[0] == 0.0  # zero pCTR -> zero value
    # Summary stats consistent with the array.
    assert res.mean_value == pytest.approx(float(expected.mean()))
    assert res.median_value == pytest.approx(float(np.median(expected)))
    assert res.std_value == pytest.approx(float(np.std(expected)))
    assert 0.0 <= res.pct_above_market_median <= 1.0


def test_value_doubling_pctr_doubles_value():
    """Doubling pCTR doubles the value (homogeneity / monotonicity)."""
    p_ctr = np.array([0.001, 0.002, 0.005], dtype=np.float64)
    cfg = ValueConfig(goal_type="CPC", cpc_target=123_456.0)
    base = compute_impression_values(p_ctr, cfg).values
    doubled = compute_impression_values(2.0 * p_ctr, cfg).values
    assert np.allclose(doubled, 2.0 * base)

    # Strictly monotone non-decreasing in pCTR.
    sorted_pctr = np.sort(p_ctr)
    sorted_vals = compute_impression_values(sorted_pctr, cfg).values
    assert np.all(np.diff(sorted_vals) >= 0.0)


def test_value_pct_above_market_median():
    """pct_above_market_median equals the fraction of values strictly above it."""
    p_ctr = np.array([0.0, 0.0005, 0.001, 0.01], dtype=np.float64)
    cfg = ValueConfig(goal_type="CPC", cpc_target=200_000.0)
    market_median = 100.0  # values = [0, 100, 200, 2000]; strictly above -> 2/4
    res = compute_impression_values(p_ctr, cfg, market_median=market_median)
    expected = float(np.mean((p_ctr * cfg.cpc_target) > market_median))
    assert res.pct_above_market_median == pytest.approx(expected)
    assert res.pct_above_market_median == pytest.approx(0.5)


def test_value_cpm_is_constant():
    """CPM goal gives a constant value cpm_target/1000 regardless of pCTR."""
    p_ctr = np.array([0.001, 0.5, 0.9], dtype=np.float64)
    cfg = ValueConfig(goal_type="CPM", cpm_target=80_000.0)
    res = compute_impression_values(p_ctr, cfg)
    assert np.allclose(res.values, 80_000.0 / 1000.0)


def test_value_unknown_goal_raises():
    """CPA (and any unknown goal_type) raises ValueError in compute_impression_values."""
    p_ctr = np.array([0.001, 0.002])
    with pytest.raises(ValueError):
        compute_impression_values(p_ctr, ValueConfig(goal_type="CPA"))
    with pytest.raises(ValueError):
        compute_impression_values(p_ctr, ValueConfig(goal_type="bogus"))


def test_value_cpa_is_bilinear():
    """CPA value == p_ctr * p_cvr * cpa_target."""
    p_ctr = np.array([0.01, 0.02], dtype=np.float64)
    p_cvr = np.array([0.1, 0.5], dtype=np.float64)
    cfg = ValueConfig(goal_type="CPA", cpa_target=500_000.0)
    res = compute_impression_values_cpa(p_ctr, p_cvr, cfg)
    assert np.allclose(res.values, p_ctr * p_cvr * 500_000.0)


# =============================================================================
# (c) Win-accounting identities: run_auction_simulation / compute_simulation_metrics
# =============================================================================


def _tiny_auction():
    """A tiny deterministic auction with a mix of wins and losses, and one click."""
    #            win   win   lose  win   lose
    bids = np.array([100.0, 80.0, 30.0, 200.0, 10.0])
    market_prices = np.array([60.0, 80.0, 50.0, 120.0, 70.0])
    values = np.array([150.0, 90.0, 40.0, 300.0, 20.0])
    clicks = np.array([1, 0, 1, 1, 0], dtype=np.int32)  # click on a loser too
    return bids, market_prices, values, clicks


def test_auction_win_accounting_first_price():
    """First-price: wins {0,1}, losers zeroed, won payment == bid, surplus == V - pay."""
    bids, mp, values, clicks = _tiny_auction()
    res = run_auction_simulation(bids, mp, values, clicks, auction_type="first_price")

    expected_wins = (bids >= mp).astype(np.int32)
    assert np.array_equal(res.wins, expected_wins)
    assert set(np.unique(res.wins)).issubset({0, 1})

    win_mask = res.wins.astype(bool)
    lose_mask = ~win_mask
    # Losers fully zeroed.
    assert np.all(res.payments[lose_mask] == 0.0)
    assert np.all(res.surplus[lose_mask] == 0.0)
    assert np.all(res.clicks[lose_mask] == 0)
    # First-price: won payment == own bid.
    assert np.allclose(res.payments[win_mask], bids[win_mask])
    # Surplus identity on wins.
    assert np.allclose(res.surplus[win_mask], values[win_mask] - res.payments[win_mask])
    # Clicks only counted on wins: row 2 (loser) had a click but is zeroed.
    assert np.array_equal(res.clicks, clicks * expected_wins)


def test_auction_second_price_pay_market_and_le_bid():
    """Second-price: won payment == market_price and <= bid; spend <= first-price."""
    bids, mp, values, clicks = _tiny_auction()
    res2 = run_auction_simulation(bids, mp, values, clicks, auction_type="second_price")
    res1 = run_auction_simulation(bids, mp, values, clicks, auction_type="first_price")

    win_mask = res2.wins.astype(bool)
    assert np.allclose(res2.payments[win_mask], mp[win_mask])
    assert np.all(res2.payments[win_mask] <= bids[win_mask] + 1e-9)
    # Identical wins both auction types; second-price spend <= first-price spend.
    assert np.array_equal(res2.wins, res1.wins)
    assert res2.payments.sum() <= res1.payments.sum() + 1e-9


def test_auction_unknown_type_raises():
    """Unknown auction_type raises ValueError."""
    bids, mp, values, clicks = _tiny_auction()
    with pytest.raises(ValueError):
        run_auction_simulation(bids, mp, values, clicks, auction_type="dutch")


def test_simulation_metrics_identities_first_price():
    """Metrics match the underlying AuctionResult sums; first-price overpayment >= 0."""
    bids, mp, values, clicks = _tiny_auction()
    res = run_auction_simulation(bids, mp, values, clicks, auction_type="first_price")
    cpc_target = 200_000.0
    m = compute_simulation_metrics(res, values, mp, "fp", cpc_target=cpc_target)

    assert m.n_bids == len(bids)
    assert m.n_wins == int(res.wins.sum())
    assert m.win_rate == pytest.approx(m.n_wins / m.n_bids)
    assert m.total_clicks == int(res.clicks.sum())
    assert m.total_spend == pytest.approx(float(res.payments.sum()))
    assert m.total_surplus == pytest.approx(float(res.surplus.sum()))

    assert m.avg_cpm == pytest.approx(m.total_spend / m.n_wins)
    assert m.click_rate == pytest.approx(m.total_clicks / m.n_wins)

    # ROI and CPC identities.
    assert m.roi == pytest.approx(m.total_clicks * cpc_target / m.total_spend)
    if m.total_clicks > 0:
        assert m.avg_cpc == pytest.approx(m.total_spend / m.total_clicks)

    # First-price overpayment ratio = mean((bid - mp)/mp) over wins, mp>0 -> >= 0.
    assert m.overpayment_ratio >= 0.0


def test_simulation_metrics_second_price_pays_market_price():
    """Second-price spends the market price exactly on each win (vs the bid)."""
    bids, mp, values, clicks = _tiny_auction()
    res = run_auction_simulation(bids, mp, values, clicks, auction_type="second_price")
    m = compute_simulation_metrics(res, values, mp, "sp")

    win_mask = res.wins.astype(bool)
    # Spend identity: second-price total spend == sum of market prices over wins.
    assert m.total_spend == pytest.approx(float(mp[win_mask].sum()))
    # overpayment_ratio is computed from the original bids (not payments), so it
    # still reflects bid - market_price over wins (>= 0 here since bids >= mp).
    expected_over = float(np.mean((bids[win_mask] - mp[win_mask]) / mp[win_mask]))
    assert m.overpayment_ratio == pytest.approx(expected_over)
    assert m.overpayment_ratio >= 0.0


def test_simulation_metrics_zero_wins_yields_inf():
    """No wins -> spend/surplus zero; avg_cpc and roi are inf."""
    bids = np.array([1.0, 2.0, 3.0])
    mp = np.array([100.0, 200.0, 300.0])  # all losses
    values = np.array([50.0, 60.0, 70.0])
    clicks = np.array([1, 0, 1], dtype=np.int32)
    res = run_auction_simulation(bids, mp, values, clicks, auction_type="first_price")

    assert res.wins.sum() == 0
    m = compute_simulation_metrics(res, values, mp, "none")
    assert m.n_wins == 0
    assert m.total_spend == 0.0
    assert m.total_surplus == 0.0
    assert m.win_rate == 0.0
    assert m.avg_cpm == 0.0
    assert m.click_rate == 0.0
    assert np.isinf(m.avg_cpc)
    assert np.isinf(m.roi)


# =============================================================================
# (d) load_exchange_cdfs / load_market_cdf / MarketCDF basic behavior
# =============================================================================


def test_market_cdf_interpolation_and_helpers(linear_cdf):
    """_interpolate_cdf clips to [0, cdf[-1]], left=0; linear/percentile helpers sane."""
    # At grid points F(p) = p/200; below grid -> 0, above grid -> cdf[-1].
    pts = np.array([-5.0, 0.0, 100.0, 200.0, 1000.0])
    f = _interpolate_cdf(pts, linear_cdf)
    assert f[0] == 0.0                      # left clip
    assert f[1] == pytest.approx(0.0)
    assert f[2] == pytest.approx(0.5)       # 100/200
    assert f[3] == pytest.approx(1.0)       # 200/200
    assert f[4] == pytest.approx(linear_cdf.cdf[-1])  # right clip to last cdf value
    assert np.all(np.diff(f) >= -1e-12)     # non-decreasing

    # linear_bid: with alpha <= 1 and unclipped range, bid == alpha * V.
    values = np.array([10.0, 50.0, 100.0])
    lb = linear_bid(values, alpha=0.8, min_bid=1.0, max_bid=300.0)
    assert np.allclose(lb, 0.8 * values)

    # percentile_bid: bid <= V; higher target percentile -> non-decreasing target price.
    pb_lo = percentile_bid(values, linear_cdf, target_percentile=0.25)
    pb_hi = percentile_bid(values, linear_cdf, target_percentile=0.75)
    assert np.all(pb_lo <= values + 1e-9)
    assert np.all(pb_hi <= values + 1e-9)
    # For the larger value, the binding cap is the percentile target price, which
    # must be non-decreasing in the target percentile.
    assert pb_hi[-1] >= pb_lo[-1] - 1e-9


def test_load_exchange_cdfs_with_synthetic_npz(tmp_path):
    """load_exchange_cdfs scans km_cdf_exchange_<id>.npz; load_market_cdf parses keys."""
    grid = _PRICE_GRID.copy()
    cdf = _CDF.copy()
    for ex_id in (1, 2):
        np.savez(
            tmp_path / f"km_cdf_exchange_{ex_id}.npz",
            price_grid=grid,
            cdf=cdf,
            survival=1.0 - cdf,
            median_price=float(ex_id * 50),
        )
    # A non-matching file must be ignored by the scanner.
    np.savez(tmp_path / "km_cdf_overall.npz", price_grid=grid, cdf=cdf,
             survival=1.0 - cdf, median_price=100.0)

    cdfs = load_exchange_cdfs(str(tmp_path))
    assert set(cdfs.keys()) == {"1", "2"}
    for ex_id, mcdf in cdfs.items():
        assert isinstance(mcdf, MarketCDF)
        assert np.array_equal(mcdf.price_grid, grid)
        assert np.array_equal(mcdf.cdf, cdf)
        assert np.all(np.diff(mcdf.price_grid) >= 0)
        assert np.all(np.diff(mcdf.cdf) >= -1e-12)
    assert cdfs["2"].median_price == pytest.approx(100.0)

    # load_market_cdf derives source from the filename.
    single = load_market_cdf(str(tmp_path / "km_cdf_exchange_1.npz"))
    assert single.source == "km_cdf_exchange_1"
    assert single.median_price == pytest.approx(50.0)


@pytest.mark.skipif(
    not os.path.isdir(_CDF_DIR),
    reason="results/market_price_cdf/ artifacts not present",
)
def test_load_real_market_cdf_artifacts_if_present():
    """If real KM CDF artifacts exist, they load with monotone grid/cdf in [0,1]."""
    cdfs = load_exchange_cdfs(_CDF_DIR)
    if not cdfs:
        pytest.skip("no km_cdf_exchange_*.npz files in artifact dir")
    for ex_id, mcdf in cdfs.items():
        assert isinstance(mcdf, MarketCDF)
        assert mcdf.price_grid.ndim == 1 and mcdf.cdf.ndim == 1
        assert mcdf.price_grid.shape == mcdf.cdf.shape
        assert np.all(np.diff(mcdf.price_grid) >= 0)
        assert np.all(np.diff(mcdf.cdf) >= -1e-9)
        assert mcdf.cdf.min() >= -1e-9
        assert mcdf.cdf.max() <= 1.0 + 1e-9
        # Optimal bids against a real CDF still respect the b <= V cap.
        v = np.array([50.0, 150.0, 280.0])
        bids, surplus = optimal_bid_vectorized(v, mcdf, n_candidates=500)
        assert np.all(bids <= v + 1e-9)
        assert np.all(surplus >= 0.0)


# ---------------------------------------------------------------------------
# (e) Bootstrap CIs on realized-surplus GAPS (Stage B2 inference)
# ---------------------------------------------------------------------------

def _rare_click_surplus_vectors(n=200000, n_click=400, edge=120, seed=1):
    """Synthetic per-row realized-surplus vectors for two cells (A, B).

    Mimics the RTB regime: a few rare CLICKED rows worth +CPC each dominate the
    gap, atop millions of small payment-only costs. Cell A wins `edge` MORE of
    the clicked rows than B -> a true positive gap of ~edge*CPC.
    """
    rng = np.random.default_rng(seed)
    CPC = 200_000.0
    s_a = -rng.uniform(0, 50, size=n)      # payment-only cost on every row
    s_b = -rng.uniform(0, 50, size=n)
    clicked = np.arange(n_click)           # the rare clicked rows
    s_a[clicked] = CPC - rng.uniform(0, 80, size=n_click)   # A wins all n_click clicks
    s_b[clicked] = CPC - rng.uniform(0, 80, size=n_click)
    s_b[clicked[:edge]] = -rng.uniform(0, 80, size=edge)    # B loses `edge` of them
    adv = rng.integers(0, 6, size=n)       # 6 advertiser clusters
    return s_a, s_b, adv, CPC, edge


def test_paired_bootstrap_detects_true_positive_gap():
    """A strong +edge*CPC gap: point > 0, CI excludes 0, p_gt_0 ~ 1."""
    s_a, s_b, _, CPC, edge = _rare_click_surplus_vectors()
    res = paired_bootstrap_surplus_gap(s_a, s_b, n_boot=2000, seed=0)
    assert isinstance(res, SurplusGapCI)
    assert res.point == pytest.approx(float((s_a - s_b).sum()))
    assert res.point > 0
    assert res.ci95_lo > 0          # CI excludes 0
    assert res.p_gt_0 > 0.99
    # gap is dominated by the ~edge clicked-row wins (order edge*CPC)
    assert res.point == pytest.approx(edge * CPC, rel=0.2)


def test_paired_bootstrap_zero_signal_straddles_zero():
    """When both cells have the same surplus law, the gap CI straddles 0."""
    rng = np.random.default_rng(3)
    CPC = 200_000.0
    n, n_click = 100000, 300
    base = -rng.uniform(0, 50, size=n)
    s_a = base.copy(); s_b = base.copy()
    # identical clicked rows for both -> no systematic gap, only noise
    cl = np.arange(n_click)
    s_a[cl] = CPC; s_b[cl] = CPC
    # add independent jitter on a few clicked rows (symmetric)
    s_a[cl[:40]] -= rng.uniform(0, CPC, size=40)
    s_b[cl[40:80]] -= rng.uniform(0, CPC, size=40)
    res = paired_bootstrap_surplus_gap(s_a, s_b, n_boot=2000, seed=0)
    assert res.ci95_lo < 0 < res.ci95_hi      # straddles 0
    assert 0.1 < res.p_gt_0 < 0.9


def test_paired_bootstrap_deterministic_and_heavy_cap_consistent():
    """Same seed -> identical; exact (heavy_cap>=n) ~ hybrid (heavy_cap<n) gap CI."""
    s_a, s_b, _, _, _ = _rare_click_surplus_vectors(n=50000, n_click=300)
    r1 = paired_bootstrap_surplus_gap(s_a, s_b, n_boot=1000, seed=7)
    r2 = paired_bootstrap_surplus_gap(s_a, s_b, n_boot=1000, seed=7)
    assert r1 == r2  # determinism

    exact = paired_bootstrap_surplus_gap(s_a, s_b, n_boot=4000, seed=1, heavy_cap=10**9)
    hybrid = paired_bootstrap_surplus_gap(s_a, s_b, n_boot=4000, seed=1, heavy_cap=2000)
    assert exact.point == pytest.approx(hybrid.point)
    # CLT bulk vs exact tail agree on the CI to within a few percent of the width
    width = exact.ci95_hi - exact.ci95_lo
    assert abs(exact.ci95_lo - hybrid.ci95_lo) < 0.1 * width
    assert abs(exact.ci95_hi - hybrid.ci95_hi) < 0.1 * width


def test_cluster_bootstrap_is_conservative_when_signal_concentrated():
    """When the gap rides on ONE advertiser, cluster bootstrap is far wider.

    The whole +edge*CPC advantage lives in cluster 0. The paired row bootstrap
    sees it as a stable signal; the cluster bootstrap resamples the 6 clusters,
    so cluster 0 appears 0..k times -> the gap swings between ~0 and a multiple
    of the signal -> a much wider, conservative interval. This is exactly why
    the advertiser-cluster CI is the bound a decision claim must clear.
    """
    rng = np.random.default_rng(5)
    CPC = 200_000.0
    n, n_click, edge = 120000, 360, 120
    s_a = -rng.uniform(0, 50, size=n)
    s_b = -rng.uniform(0, 50, size=n)
    cl = np.arange(n_click)
    s_a[cl] = CPC; s_b[cl] = CPC
    s_b[cl[:edge]] = -rng.uniform(0, 80, size=edge)   # A's advantage = `edge` clicks
    adv = rng.integers(0, 6, size=n)
    adv[cl[:edge]] = 0                                 # concentrate the advantage in adv 0

    paired = paired_bootstrap_surplus_gap(s_a, s_b, n_boot=2000, seed=0)
    clus = cluster_bootstrap_surplus_gap(s_a, s_b, adv, n_boot=2000, seed=0)
    clus2 = cluster_bootstrap_surplus_gap(s_a, s_b, adv, n_boot=2000, seed=0)
    assert clus == clus2                              # determinism
    assert clus.point == pytest.approx(paired.point)
    assert clus.method == "cluster"
    # Signal concentrated in one of 6 clusters -> cluster CI strictly wider.
    assert (clus.ci95_hi - clus.ci95_lo) > (paired.ci95_hi - paired.ci95_lo)
    # ...and conservative enough that dropping adv 0 pulls the low end down to ~0.
    assert clus.ci95_lo < paired.ci95_lo


def test_bootstrap_empty_input():
    """Empty vectors -> zero gap, no raise."""
    z = np.array([])
    r = paired_bootstrap_surplus_gap(z, z, n_boot=10)
    assert r.point == 0.0 and r.ci95_lo == 0.0 and r.ci95_hi == 0.0
