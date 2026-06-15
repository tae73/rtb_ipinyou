"""FastAPI RTB Bidder — lightweight serving demo (CPU-only, <100ms).

Demonstrates the full Real-Time Bidding loop for a first-price auction:

    feature store inputs
        -> featurize (replicate baseline LR train preprocessing EXACTLY)
        -> pCTR        (sklearn LogisticRegression, lr_ctr_all.joblib)
        -> value V(x)  (src.bidding.value.compute_impression_values, CPC)
        -> shaded bid  (src.bidding.shading, exchange-conditional optimal
                        first-price shading via Kaplan-Meier market CDFs)
        -> BidResponse

This is a solo side-project demo, NOT a production system. It exists to show
production-shaped capability: artifacts loaded once at startup, train/serve
parity, structured logging, health checks, graceful validation, and sub-100ms
latency on CPU.

--------------------------------------------------------------------------------
TRAIN/SERVE PARITY (no skew)
--------------------------------------------------------------------------------
The pCTR model is the well-calibrated baseline ``lr_ctr_all`` LogisticRegression
trained in ``scripts/train.py`` (``baseline`` command, ``--task ctr_all``,
``include_lr=True``). Its preprocessing, replicated here verbatim, is:

  1. feature order = feature_info["categorical"] + feature_info["numerical"]
     (17 categorical + 13 numerical = 30 features), taken from the joblib
     artifact's saved ``feature_names`` (identical to feature_metadata.json).
  2. ALL 30 features (categorical included — they are integer-coded, NOT
     one-hot) are assembled as a raw float32 row vector in that order.
  3. The single saved ``StandardScaler`` (fit on the training set) transforms
     the full 30-vector: ``scaler.transform(X.astype(np.float32))``.
     => normalization_stats in feature_metadata.json are NOT applied separately;
        the StandardScaler is the sole normalizer (it covers both cat & num).
  4. pCTR = ``model.predict_proba(Xs)[:, 1]``.

The artifact is ``{"model": LogisticRegression, "scaler": StandardScaler,
"feature_names": List[str]}``; ``model.n_features_in_ == 30`` is asserted
against ``len(feature_names)`` at startup.

--------------------------------------------------------------------------------
REQUEST CONTRACT (POST /bid)
--------------------------------------------------------------------------------
A real feature store supplies model-ready features. The request accepts a
``features`` dict mapping the 30 named features to numeric values. Any feature
omitted defaults to the training-set mean for that column (from the scaler),
which standardizes to ~0 — a neutral, well-defined fallback. ``adexchange``
(also a model feature) additionally selects the exchange-conditional market
CDF. Optional ``cpc_target`` overrides the CPC value goal.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.bidding.shading import (
    MarketCDF,
    ShadingConfig,
    _interpolate_cdf,
    compute_shaded_bids,
    load_exchange_cdfs,
    load_market_cdf,
)
from src.bidding.value import ValueConfig, compute_impression_values

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rtb.serving")

# --------------------------------------------------------------------------- #
# Artifact paths (relative to project root; resolved at import time)
# --------------------------------------------------------------------------- #
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LR_PATH = _PROJECT_ROOT / "results" / "models" / "lr_ctr_all.joblib"
_META_PATH = (
    _PROJECT_ROOT / "data" / "ipinyou" / "prediction" / "features" / "feature_metadata.json"
)
_CDF_DIR = _PROJECT_ROOT / "results" / "market_price_cdf"

# CPC value goal default (CPM per click); see ValueConfig / NB05 Section 11.
_DEFAULT_CPC_TARGET = 200_000.0


# --------------------------------------------------------------------------- #
# Artifact bundle (loaded once at startup)
# --------------------------------------------------------------------------- #
class _Artifacts:
    """Holds the loaded model + scaler + feature schema + market CDFs."""

    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.feature_index: Dict[str, int] = {}
        self.default_vector: Optional[np.ndarray] = None  # scaler means (neutral fallback)
        self.overall_cdf: Optional[MarketCDF] = None
        self.exchange_cdfs: Dict[str, MarketCDF] = {}
        self.loaded = False

    @property
    def n_cdfs(self) -> int:
        return len(self.exchange_cdfs) + (1 if self.overall_cdf is not None else 0)


ARTIFACTS = _Artifacts()


def load_artifacts() -> _Artifacts:
    """Load model, scaler, feature schema and market CDFs into ARTIFACTS.

    Idempotent: safe to call multiple times (TestClient + uvicorn both trigger
    startup). Raises FileNotFoundError with a clear message if artifacts are
    missing.
    """
    if ARTIFACTS.loaded:
        return ARTIFACTS

    if not _LR_PATH.exists():
        raise FileNotFoundError(
            f"pCTR model artifact missing: {_LR_PATH}. "
            "Train it via: python scripts/train.py baseline "
            "--data-dir data/ipinyou/prediction/features "
            "--model-dir results/models --task ctr_all"
        )
    if not _CDF_DIR.exists():
        raise FileNotFoundError(f"Market CDF directory missing: {_CDF_DIR}")

    logger.info("Loading LR pCTR artifact: %s", _LR_PATH)
    bundle = joblib.load(_LR_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = list(bundle["feature_names"])

    # Train/serve parity guard: assembled vector length must equal n_features_in_.
    n_in = int(getattr(model, "n_features_in_", len(feature_names)))
    if n_in != len(feature_names):
        raise ValueError(
            f"Train/serve skew: model.n_features_in_={n_in} != "
            f"len(feature_names)={len(feature_names)}"
        )

    ARTIFACTS.model = model
    ARTIFACTS.scaler = scaler
    ARTIFACTS.feature_names = feature_names
    ARTIFACTS.feature_index = {name: i for i, name in enumerate(feature_names)}
    # Neutral fallback: training-set mean per column (standardizes to ~0).
    ARTIFACTS.default_vector = np.asarray(scaler.mean_, dtype=np.float64).copy()

    logger.info("Loading market CDFs from: %s", _CDF_DIR)
    overall_path = _CDF_DIR / "km_cdf_overall.npz"
    if overall_path.exists():
        ARTIFACTS.overall_cdf = load_market_cdf(str(overall_path))
    ARTIFACTS.exchange_cdfs = load_exchange_cdfs(str(_CDF_DIR))
    if ARTIFACTS.overall_cdf is None:
        # Fall back to any exchange CDF as the common reference if no overall.
        if ARTIFACTS.exchange_cdfs:
            ARTIFACTS.overall_cdf = next(iter(ARTIFACTS.exchange_cdfs.values()))
        else:
            raise FileNotFoundError(f"No market CDFs found in {_CDF_DIR}")

    ARTIFACTS.loaded = True
    logger.info(
        "Artifacts loaded: model_features=%d, exchange_cdfs=%s",
        len(feature_names),
        sorted(ARTIFACTS.exchange_cdfs.keys()),
    )
    return ARTIFACTS


# --------------------------------------------------------------------------- #
# Request / Response schemas
# --------------------------------------------------------------------------- #
class BidRequest(BaseModel):
    """Bid request from the feature store / upstream bidder.

    Fields:
        features: Mapping of the 30 model feature names to numeric values
            (categorical features are integer-coded, NOT one-hot). Any omitted
            feature defaults to its training-set mean (neutral, ~0 after
            standardization). Extra/unknown keys are ignored with a warning.
        adexchange: Ad exchange id (1/2/3) used to select the exchange-conditional
            market-price CDF for shading. If provided in `features` too, the
            top-level field takes precedence for CDF selection.
        slotprice: Optional floor price; only used by the dual_regime strategy.
        cpc_target: Optional CPC value goal (CPM per click); defaults to 200000.
        strategy: Shading strategy: 'optimal' (default, exchange-conditional),
            'linear', 'percentile', or 'dual_regime'.
    """

    features: Dict[str, float] = Field(
        default_factory=dict,
        description="Model-ready named features (subset allowed; missing -> train mean).",
    )
    adexchange: int = Field(1, description="Ad exchange id (1/2/3) for CDF selection.")
    slotprice: Optional[float] = Field(
        None, description="Floor price (dual_regime strategy only)."
    )
    cpc_target: float = Field(
        _DEFAULT_CPC_TARGET, gt=0, description="CPC value goal (CPM per click)."
    )
    strategy: str = Field("optimal", description="optimal|linear|percentile|dual_regime")


class BidResponse(BaseModel):
    """Bid decision returned to the upstream bidder."""

    pctr: float = Field(..., description="Predicted click-through rate in [0, 1].")
    value: float = Field(..., description="Impression value V(x) = pCTR * cpc_target (CPM).")
    bid: float = Field(..., description="Shaded first-price bid (CPM), 0 <= bid <= value.")
    shading_factor: float = Field(..., description="bid / value (0-1).")
    expected_win_prob: float = Field(..., description="F(bid) from market CDF.")
    regime: str = Field(..., description="'competitive' or 'floor_bound'.")
    exchange_cdf: str = Field(..., description="Source CDF used for this exchange.")
    latency_ms: float = Field(..., description="Server-side handling latency (ms).")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_cdfs: int


# --------------------------------------------------------------------------- #
# Featurization (replicates baseline LR training preprocessing EXACTLY)
# --------------------------------------------------------------------------- #
def _assemble_feature_vector(features: Dict[str, float], art: _Artifacts) -> np.ndarray:
    """Build the raw 30-d feature row in training order, then StandardScale.

    Mirrors scripts/train.py LR path:
        X = df[categorical + numerical].values.astype(np.float32)
        Xs = scaler.transform(X)            # the saved StandardScaler
    Missing features default to the training-set column mean (neutral).
    """
    raw = art.default_vector.copy()  # (30,) start from training means
    unknown = [k for k in features if k not in art.feature_index]
    if unknown:
        logger.warning("Ignoring unknown feature keys: %s", unknown)
    for name, val in features.items():
        idx = art.feature_index.get(name)
        if idx is not None:
            raw[idx] = float(val)
    raw = raw.reshape(1, -1).astype(np.float32)
    return art.scaler.transform(raw)  # (1, 30) standardized


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="RTB iPinYou First-Price Bidder",
    description="Demo: features -> pCTR -> V(x) -> shaded first-price bid.",
    version="0.1.0",
)


@app.on_event("startup")
def _startup() -> None:
    load_artifacts()


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness/readiness probe."""
    return HealthResponse(
        status="ok" if ARTIFACTS.loaded else "loading",
        model_loaded=ARTIFACTS.loaded and ARTIFACTS.model is not None,
        n_cdfs=ARTIFACTS.n_cdfs,
    )


@app.post("/bid", response_model=BidResponse)
def bid(req: BidRequest) -> BidResponse:
    """Run the full RTB loop and return a shaded first-price bid."""
    t0 = time.perf_counter()
    art = load_artifacts()  # idempotent; ensures artifacts present

    if art.model is None or art.scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # 1) Featurize with exact training preprocessing.
    try:
        x = _assemble_feature_vector(req.features, art)
    except Exception as exc:  # noqa: BLE001 - surface as clean 400
        raise HTTPException(status_code=400, detail=f"Featurization failed: {exc}") from exc

    # 2) pCTR.
    pctr = float(art.model.predict_proba(x)[:, 1][0])
    pctr_arr = np.array([pctr], dtype=np.float64)

    # 3) Value V(x) = pCTR * cpc_target (CPC goal).
    value_cfg = ValueConfig(goal_type="CPC", cpc_target=req.cpc_target)
    v = compute_impression_values(pctr_arr, value_cfg)
    values = v.values  # (1,)

    # 4) Exchange-conditional first-price shading.
    ex_str = str(int(req.adexchange))
    cdf_source = art.exchange_cdfs.get(ex_str)
    used_cdf_name = cdf_source.source if cdf_source is not None else art.overall_cdf.source

    cfg = ShadingConfig(strategy=req.strategy, exchange_conditional=True)
    try:
        if req.strategy == "dual_regime":
            slot = np.array(
                [req.slotprice if req.slotprice is not None else 0.0], dtype=np.float64
            )
            result = compute_shaded_bids(values, art.overall_cdf, cfg, slotprice=slot)
        else:
            result = compute_shaded_bids(
                values,
                art.overall_cdf,
                cfg,
                exchange_ids=np.array([int(req.adexchange)]),
                exchange_cdfs=art.exchange_cdfs,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bid_val = float(result.bids[0])
    value_val = float(values[0])
    shading_factor = float(result.shading_factors[0])
    win_prob = float(result.expected_win_prob[0])
    regime_str = "floor_bound" if int(result.regime[0]) == 1 else "competitive"

    # Invariant guard: never bid above the impression value. The shading layer
    # applies a global ``min_bid`` floor (ShadingConfig.min_bid=1.0), so when
    # V(x) < min_bid (low pCTR and/or small cpc_target) the floored bid can
    # exceed V. Bidding above value is never economically rational, so we cap
    # the returned bid at V and recompute the dependent fields consistently.
    if bid_val > value_val:
        bid_val = value_val
        shading_factor = 1.0 if value_val > 0.0 else 0.0
        win_prob = float(_interpolate_cdf(np.array([bid_val]), cdf_source or art.overall_cdf)[0])

    latency_ms = (time.perf_counter() - t0) * 1000.0

    logger.info(
        "bid adexchange=%s pctr=%.6f value=%.2f bid=%.2f regime=%s latency_ms=%.2f",
        req.adexchange,
        pctr,
        value_val,
        bid_val,
        regime_str,
        latency_ms,
    )

    return BidResponse(
        pctr=pctr,
        value=value_val,
        bid=bid_val,
        shading_factor=shading_factor,
        expected_win_prob=win_prob,
        regime=regime_str,
        exchange_cdf=used_cdf_name,
        latency_ms=latency_ms,
    )
