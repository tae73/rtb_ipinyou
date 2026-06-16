"""Tests for the FastAPI RTB serving demo (src.serving.app).

Uses fastapi.testclient.TestClient (in-process, no network). Verifies:
  - /healthz == 200 with model_loaded true and n_cdfs > 0
  - /bid on a valid synthetic payload returns:
        * pctr in [0, 1]
        * finite bid, 0 <= bid <= value  (shading never bids above V)
        * a regime string ('competitive' | 'floor_bound')
        * positive latency_ms
  - a malformed payload yields 422 (Pydantic validation)

Artifact loading at startup is real (loads lr_ctr_all.joblib + market CDFs); the
module-scoped client fixture pays that cost once and is marked `slow`.
"""

import math
import warnings

import pytest

# sklearn version-mismatch unpickling warnings are noise here.
warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Module-scoped TestClient; triggers (slow) artifact loading once."""
    with TestClient(app) as c:
        yield c


# Minimal valid synthetic feature payload. Missing features fall back to the
# training-set mean (neutral), so an empty dict is also valid; here we set a few
# named features to exercise the assembly path.
_VALID_PAYLOAD = {
    "features": {
        "adexchange": 1,
        "advertiser": 1458,
        "slotwidth": 300,
        "slotheight": 250,
        "slotprice": 50.0,
        "bidprice": 277.0,
        "hour": 12,
    },
    "adexchange": 1,
    "cpc_target": 200000.0,
    "strategy": "optimal",
}


@pytest.mark.slow
def test_healthz_ok(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_cdfs"] >= 1


@pytest.mark.slow
def test_bid_valid_payload(client: TestClient) -> None:
    resp = client.post("/bid", json=_VALID_PAYLOAD)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    pctr = body["pctr"]
    value = body["value"]
    bid = body["bid"]

    # pctr in [0, 1]
    assert 0.0 <= pctr <= 1.0

    # finite, non-negative bid, never above the impression value
    assert math.isfinite(bid)
    assert bid >= 0.0
    assert bid <= value + 1e-6  # shading never bids above V

    # value consistency: V = pctr * cpc_target
    assert value == pytest.approx(pctr * _VALID_PAYLOAD["cpc_target"], rel=1e-6)

    # shading factor and win prob in sane ranges
    assert 0.0 <= body["shading_factor"] <= 1.0 + 1e-6
    assert 0.0 <= body["expected_win_prob"] <= 1.0

    # regime is a known string
    assert body["regime"] in {"competitive", "floor_bound"}

    # latency is reported and positive
    assert body["latency_ms"] >= 0.0


@pytest.mark.slow
def test_bid_empty_features_uses_defaults(client: TestClient) -> None:
    """Empty features dict is valid: all features default to train means."""
    resp = client.post("/bid", json={"features": {}, "adexchange": 2})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 0.0 <= body["pctr"] <= 1.0
    assert body["bid"] <= body["value"] + 1e-6


@pytest.mark.slow
def test_bid_different_exchanges_select_cdf(client: TestClient) -> None:
    """adexchange selects the exchange-conditional CDF source."""
    for ex in (1, 2, 3):
        resp = client.post("/bid", json={"features": {}, "adexchange": ex})
        assert resp.status_code == 200, resp.text
        assert resp.json()["exchange_cdf"].endswith(str(ex))


@pytest.mark.slow
def test_bid_low_value_never_exceeds_value(client: TestClient) -> None:
    """When V(x) < shading min_bid floor (1.0), bid must still be <= value.

    With a tiny cpc_target, value = pctr * cpc_target falls below the shading
    layer's min_bid=1.0 floor. The serving layer must cap the bid at the value
    so it never bids above the impression value (the documented invariant).
    """
    for strategy in ("optimal", "linear", "percentile", "dual_regime"):
        payload = {
            "features": {},
            "adexchange": 1,
            "cpc_target": 1.0,  # value = pctr * 1.0 << 1.0
            "strategy": strategy,
        }
        if strategy == "dual_regime":
            payload["slotprice"] = 50.0
        resp = client.post("/bid", json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["value"] < 1.0  # confirms we are below the min_bid floor
        assert body["bid"] <= body["value"] + 1e-9, (strategy, body)
        assert body["bid"] >= 0.0
        assert 0.0 <= body["expected_win_prob"] <= 1.0
        assert 0.0 <= body["shading_factor"] <= 1.0 + 1e-9


@pytest.mark.slow
def test_bid_malformed_payload_422(client: TestClient) -> None:
    """Wrong types -> Pydantic 422 (graceful validation)."""
    bad = {
        "features": "not-a-dict",  # must be a mapping
        "adexchange": "not-an-int",  # must be int-coercible
        "cpc_target": -5,  # must be > 0
    }
    resp = client.post("/bid", json=bad)
    assert resp.status_code == 422


@pytest.mark.slow
def test_bid_negative_cpc_target_422(client: TestClient) -> None:
    """cpc_target must be > 0."""
    resp = client.post("/bid", json={"features": {}, "cpc_target": 0})
    assert resp.status_code == 422
