# RTB iPinYou — First-Price Bidder (serving demo)

A lightweight FastAPI service that runs the full Real-Time Bidding loop on CPU:

```
features -> pCTR -> value V(x) -> shaded first-price bid -> response
```

- **pCTR**: baseline `lr_ctr_all` (sklearn `LogisticRegression`, AUC ~0.7687,
  well-calibrated, all-bids CTR). 30 features, StandardScaler — serving
  preprocessing replicates `scripts/train.py` exactly (no train/serve skew).
- **Value**: `V(x) = pCTR × cpc_target` (CPC goal, default 200000 CPM/click) via
  `src.bidding.value.compute_impression_values`.
- **Bid**: exchange-conditional optimal first-price shading
  `b* = argmax_b (V − b)·F(b)` using Kaplan-Meier market-price CDFs
  (`src.bidding.shading`). `bid ≤ V` always.

> Solo side-project demo of production-shaped capability — not a real prod system.

## Run

```bash
# Local (project venv)
make serve                       # uvicorn src.serving.app:app on :8000

# Docker (from project root)
docker build -f mlops/serving/Dockerfile -t rtb-bidder:demo .
docker run --rm -p 8000:8000 rtb-bidder:demo
```

## Endpoints

- `GET /healthz` → `{status, model_loaded, n_cdfs}`
- `POST /bid` → `{pctr, value, bid, shading_factor, expected_win_prob, regime, exchange_cdf, latency_ms}`

The request `features` map accepts any subset of the 30 model features
(categorical features are integer-coded, not one-hot); omitted features default
to the training-set mean. `adexchange` selects the exchange-conditional CDF.

## curl example

```bash
curl -s localhost:8000/bid -H 'content-type: application/json' -d '{
  "features": {"adexchange": 1, "advertiser": 1458, "slotwidth": 300,
               "slotheight": 250, "slotprice": 50, "bidprice": 277, "hour": 12},
  "adexchange": 1,
  "cpc_target": 200000,
  "strategy": "optimal"
}'
# -> {"pctr":0.000343,"value":68.53,"bid":31.23,"shading_factor":0.456,
#     "expected_win_prob":0.30,"regime":"competitive",
#     "exchange_cdf":"km_cdf_exchange_1","latency_ms":0.49}
```

## Latency

Measured in-process (TestClient, 200 `/bid` calls, CPU): **median 1.34 ms,
P95 1.47 ms** — well under the 100 ms target (CPU LR + numpy shading; the
heavy artifact load happens once at startup).
