# RTB iPinYou — FastAPI first-price bidder (CPU-only serving demo).
#
# Lightweight image: CPU wheels only (no JAX/GPU). pCTR comes from the baseline
# sklearn LogisticRegression (lr_ctr_all.joblib); shading is pure numpy.
#
# Build (from project root):
#   docker build -f mlops/serving/Dockerfile -t rtb-bidder:demo .
# Run:
#   docker run --rm -p 8000:8000 rtb-bidder:demo
# Probe:
#   curl localhost:8000/healthz

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install uv for fast, reproducible installs.
RUN pip install --no-cache-dir uv

# Copy project metadata first to leverage Docker layer caching.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package (CPU core deps) plus the serving extra.
# onnxruntime is part of the [serving] extra but unused by this demo; install
# the explicit runtime deps to keep the image lean.
RUN uv pip install --system --no-cache . fastapi uvicorn

# Model + market-CDF artifacts (real, not stubbed).
COPY results/models/lr_ctr_all.joblib ./results/models/lr_ctr_all.joblib
COPY results/market_price_cdf ./results/market_price_cdf
COPY data/ipinyou/prediction/features/feature_metadata.json \
     ./data/ipinyou/prediction/features/feature_metadata.json

EXPOSE 8000

# Single worker is plenty for a CPU LR demo; sub-100ms P95 per /bid.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
