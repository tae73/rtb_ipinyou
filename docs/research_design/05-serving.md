# SP5: 실시간 Serving

---

## 개요

| 항목 | 내용 |
|------|------|
| **목적** | ESCM²-WC(DR) 및 입찰 모델의 Production 배포 |
| **선행 조건** | SP1-SP4 완료 |
| **핵심 산출물** | Feature Store, Model Server, RTBBidder, Monitoring |

---

## Part A: RTB Serving 특수성

### A-1. 요구사항 비교

| 항목 | 일반 ML Serving | RTB Serving |
|------|----------------|-------------|
| **Latency** | 100-500ms | **<100ms** (보통 10-50ms) |
| **QPS** | 수천-수만 | **수십만-수백만** |
| **SLA** | 99% | **99.9%+** |
| **Timeout 시** | Retry 가능 | **입찰 기회 영구 손실** |
| **Feature 시점** | 요청 시점 | Bid 시점 (100ms 전) |

### A-2. Latency Budget

```
전체 예산: 100ms (Ad Exchange timeout)

분배:
┌────────────────────────────────────┬───────────────┐
│ Component                          │ Budget        │
├────────────────────────────────────┼───────────────┤
│ Network (Exchange → DSP)           │ 10ms          │
│ Feature Lookup                     │ 5-10ms        │
│ Model Inference                    │ 5-10ms        │
│   ├─ ESCM²-WC (p_win, p_ctr,    │   3-5ms       │
│   │    p_click_bid)               │               │
│   └─ Win Propensity (optional)    │   2-3ms       │
│ Bid Computation                    │ 2-3ms         │
│   ├─ Value calculation            │   1ms         │
│   ├─ Bid shading                  │   1ms         │
│   └─ Pacing adjustment            │   1ms         │
│ Budget & Frequency Check           │ 2-3ms         │
│ Network (DSP → Exchange)           │ 10ms          │
│ Buffer                             │ 10-20ms       │
├────────────────────────────────────┼───────────────┤
│ Total                              │ ~50-70ms      │
└────────────────────────────────────┴───────────────┘
```

### A-3. 전체 Serving Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RTB Serving Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Ad Exchange                                                             │
│       │                                                                  │
│       ▼                                                                  │
│  ┌──────────┐    ┌────────────────────────────────────────────────────┐ │
│  │   Bid    │    │                   DSP Server                        │ │
│  │ Request  │───▶│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────┐ │ │
│  │ (<100ms) │    │  │ Feature │  │  Model  │  │   Bid   │  │Budget │ │ │
│  └──────────┘    │  │ Lookup  │─▶│ Predict │─▶│ Compute │─▶│ Check │ │ │
│                  │  │ (~5ms)  │  │ (~5ms)  │  │ (~2ms)  │  │(~2ms) │ │ │
│                  │  └────┬────┘  └────┬────┘  └────┬────┘  └───┬───┘ │ │
│                  │       │           │           │           │       │ │
│                  │       ▼           ▼           ▼           ▼       │ │
│                  │  ┌────────────────────────────────────────────┐   │ │
│                  │  │            Supporting Services              │   │ │
│                  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐      │   │ │
│                  │  │  │ Feature │ │  Model  │ │ Budget  │      │   │ │
│                  │  │  │  Store  │ │  Store  │ │  Store  │      │   │ │
│                  │  │  │ (Redis) │ │(S3/ONNX)│ │ (Redis) │      │   │ │
│                  │  │  └─────────┘ └─────────┘ └─────────┘      │   │ │
│                  │  └────────────────────────────────────────────┘   │ │
│                  └────────────────────────────────────────────────────┘ │
│                                    │                                     │
│                                    ▼                                     │
│                              ┌──────────┐                                │
│                              │   Bid    │                                │
│                              │ Response │                                │
│                              └──────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part B: Feature Store

### B-1. Feature 분류

| 분류 | 예시 | 갱신 주기 | 저장소 |
|------|------|----------|--------|
| **Static** | Campaign settings, User demographics | 일/주 | Redis |
| **Batch** | User history, DR weights, CATE | 시간 | Redis + HDFS |
| **Near-realtime** | Session context, Recent clicks | 분 | Redis Streams |
| **Realtime** | Bid request context | 즉시 | Request payload |

### B-2. Redis Schema 설계

```python
# Feature Store Redis Schema

# 1. User Features
# Key: user:{user_id}
# Value: JSON
{
    "user:12345": {
        "usertag": [101, 203, 405],
        "region": "110000",
        "device_type": "mobile",
        "historical_ctr": 0.012,
        "historical_cvr": 0.005,
        "segment": "high_value"
    }
}

# 2. Campaign Features
# Key: campaign:{campaign_id}
{
    "campaign:2997": {
        "advertiser": 2997,
        "industry": "ecommerce",
        "goal_type": "CPA",
        "target_cpa": 50.0,
        "daily_budget": 100000,
        "bid_floor": 10.0
    }
}

# 3. DR Weights (Precomputed) - 핵심!
# Key: dr_weight:{user_segment}:{hour}:{exchange}
{
    "dr_weight:high_value:14:tanx": 1.5,
    "dr_weight:low_value:14:tanx": 2.8,
    "dr_weight:high_value:14:youku": 1.3
}

# 4. Market Price Distribution (for bid shading)
# Key: market_dist:{hour}:{exchange}
{
    "market_dist:14:tanx": {
        "median": 85,
        "p25": 50,
        "p75": 120,
        "mean": 95
    }
}

# 5. CATE (Segment average)
# Key: cate:{campaign}:{segment}
{
    "cate:2997:high_value": 0.023,
    "cate:2997:low_value": 0.008
}

# 6. Budget State (Real-time)
# Key: budget:{campaign_id}:{date}
{
    "budget:2997:2026-02-04": 75000
}
```

### B-3. Feature Store Client

```python
import redis
import json
from typing import Dict, Any, Optional

class RTBFeatureStore:
    """RTB용 Feature Store Client"""

    def __init__(self, redis_host: str, redis_port: int,
                 local_cache_ttl: int = 60):
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
            socket_timeout=0.005,  # 5ms timeout
            socket_connect_timeout=0.005
        )
        self.local_cache = {}
        self.local_cache_ttl = local_cache_ttl

    def get_user_features(self, user_id: str) -> Dict[str, Any]:
        """User features 조회"""
        key = f"user:{user_id}"

        # Local cache check
        cached = self._get_from_local_cache(key)
        if cached:
            return cached

        # Redis lookup
        try:
            data = self.redis.get(key)
            if data:
                features = json.loads(data)
                self._set_local_cache(key, features)
                return features
        except redis.exceptions.TimeoutError:
            pass

        # Default for unknown users
        return self._default_user_features()

    def get_campaign_features(self, campaign_id: str) -> Dict[str, Any]:
        """Campaign features 조회"""
        key = f"campaign:{campaign_id}"
        try:
            data = self.redis.get(key)
            return json.loads(data) if data else {}
        except:
            return {}

    def get_dr_weight(self, user_segment: str, hour: int,
                      exchange: str, default: float = 1.0) -> float:
        """
        Precomputed DR weight 조회

        중요: DR weight는 실시간 계산 불가
        → 배치로 사전 계산하여 저장
        """
        key = f"dr_weight:{user_segment}:{hour}:{exchange}"
        try:
            weight = self.redis.get(key)
            return float(weight) if weight else default
        except:
            return default

    def get_market_distribution(self, hour: int, exchange: str) -> Dict:
        """Market price distribution 조회"""
        key = f"market_dist:{hour}:{exchange}"
        try:
            data = self.redis.get(key)
            if data:
                return json.loads(data)
        except:
            pass
        return {"median": 70, "p25": 40, "p75": 120, "mean": 80}

    def get_cate(self, campaign: str, segment: str) -> float:
        """Segment CATE 조회"""
        key = f"cate:{campaign}:{segment}"
        try:
            cate = self.redis.get(key)
            return float(cate) if cate else 0.0
        except:
            return 0.0

    def get_budget_remaining(self, campaign_id: str, date: str) -> float:
        """남은 예산 조회"""
        key = f"budget:{campaign_id}:{date}"
        try:
            budget = self.redis.get(key)
            return float(budget) if budget else 0.0
        except:
            return 0.0

    def decrement_budget(self, campaign_id: str, date: str,
                         amount: float) -> float:
        """예산 차감 (atomic)"""
        key = f"budget:{campaign_id}:{date}"
        try:
            return self.redis.incrbyfloat(key, -amount)
        except:
            return -1

    def _get_from_local_cache(self, key: str) -> Optional[Dict]:
        """Local cache 조회"""
        import time
        cached = self.local_cache.get(key)
        if cached and time.time() - cached['ts'] < self.local_cache_ttl:
            return cached['value']
        return None

    def _set_local_cache(self, key: str, value: Any):
        """Local cache 저장"""
        import time
        self.local_cache[key] = {'value': value, 'ts': time.time()}

    def _default_user_features(self) -> Dict:
        """Unknown user의 기본 features"""
        return {
            "usertag": [],
            "region": "unknown",
            "device_type": "unknown",
            "historical_ctr": 0.01,
            "historical_cvr": 0.005,
            "segment": "default"
        }
```

### B-4. DR Weight 사전 계산

```python
class DRWeightPrecomputer:
    """
    DR Weight 배치 사전 계산

    실시간 계산 불가 → 배치로 계산 후 Feature Store 저장
    """

    def __init__(self, propensity_model, feature_store):
        self.propensity_model = propensity_model
        self.feature_store = feature_store

    def compute_and_store(self, data, segments=['hour', 'exchange', 'user_segment']):
        """
        세그먼트별 평균 DR weight 계산 및 저장

        주기: 매 시간 또는 매일
        """
        import numpy as np

        for segment_values, group in data.groupby(segments):
            # Propensity 계산
            X = group[self.propensity_model.feature_cols]
            bid = group['bid_price'].values
            propensity = self.propensity_model.predict(X, bid)

            # IPW weight
            weights = 1.0 / np.clip(propensity, 0.05, 0.95)

            # Stabilize
            marginal = group['win'].mean()
            weights = marginal * weights

            # Clip
            weights = np.clip(weights, 0.5, 5.0)

            # 평균 weight 저장
            avg_weight = weights.mean()
            hour, exchange, user_segment = segment_values

            key = f"dr_weight:{user_segment}:{hour}:{exchange}"
            self.feature_store.redis.set(key, avg_weight, ex=3600)  # 1시간 TTL

        print(f"Stored DR weights for {len(data.groupby(segments))} segments")
```

---

## Part C: Model Serving

### C-1. Model Export

```python
import torch
import onnx
import onnxruntime as ort

class ModelExporter:
    """Model을 Serving format으로 변환"""

    @staticmethod
    def export_lightgbm_to_onnx(model, output_path: str, n_features: int):
        """LightGBM → ONNX"""
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        onnx.save_model(onnx_model, output_path)

    @staticmethod
    def export_pytorch_to_onnx(model: torch.nn.Module,
                                example_input: torch.Tensor,
                                output_path: str):
        """PyTorch → ONNX"""
        model.eval()

        torch.onnx.export(
            model,
            example_input,
            output_path,
            input_names=['features'],
            output_names=['p_win', 'p_ctr', 'p_click_bid'],
            dynamic_axes={
                'features': {0: 'batch_size'},
                'p_win': {0: 'batch_size'},
                'p_ctr': {0: 'batch_size'},
                'p_click_bid': {0: 'batch_size'}
            },
            opset_version=13
        )

    @staticmethod
    def quantize_onnx(model_path: str, output_path: str):
        """ONNX INT8 Quantization"""
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            model_path,
            output_path,
            weight_type=QuantType.QUInt8
        )
```

### C-2. Model Server

```python
import onnxruntime as ort
import numpy as np
from typing import Dict
import threading

class RTBModelServer:
    """
    RTB Model Server

    Features:
    - Multi-model serving
    - Thread-safe inference
    - Warmup for consistent latency
    """

    def __init__(self, model_configs: Dict[str, str]):
        """
        Args:
            model_configs: {model_name: model_path}
        """
        self.sessions = {}

        for name, path in model_configs.items():
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 1

            self.sessions[name] = ort.InferenceSession(
                path, sess_options=sess_options
            )

        self._warmup()

    def _warmup(self, n_iterations: int = 100):
        """Warmup to reduce cold-start latency"""
        for name, session in self.sessions.items():
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            shape = [1 if isinstance(d, str) or d is None else d
                     for d in input_shape]
            dummy = np.random.randn(*shape).astype(np.float32)

            for _ in range(n_iterations):
                session.run(None, {input_name: dummy})

    def predict(self, model_name: str, features: np.ndarray) -> np.ndarray:
        """Single prediction"""
        session = self.sessions[model_name]
        input_name = session.get_inputs()[0].name

        features = features.astype(np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        outputs = session.run(None, {input_name: features})
        return outputs[0]

    def predict_escm2wc(self, features: np.ndarray) -> Dict[str, float]:
        """ESCM²-WC 예측 (p_win, p_ctr, p_click_bid)"""
        # ESCM²-WC 모델 3개 출력
        outputs = self.predict('escm2wc', features)

        return {
            'p_win': outputs[0][0],
            'p_ctr': outputs[0][1],
            'p_click_bid': outputs[0][2]
        }

    def health_check(self) -> Dict[str, bool]:
        """Health check"""
        status = {}
        for name in self.sessions:
            try:
                session = self.sessions[name]
                input_shape = session.get_inputs()[0].shape
                shape = [1 if isinstance(d, str) or d is None else d
                         for d in input_shape]
                dummy = np.random.randn(*shape).astype(np.float32)
                self.predict(name, dummy)
                status[name] = True
            except Exception:
                status[name] = False
        return status
```

---

## Part D: 통합 Bidder Service

```python
from dataclasses import dataclass
from typing import Optional, Dict
import time

@dataclass
class BidRequest:
    """Bid Request"""
    bid_id: str
    user_id: str
    campaign_id: str
    exchange: str
    hour: int
    weekday: int
    slot_width: int
    slot_height: int
    floor_price: float
    region: str

@dataclass
class BidResponse:
    """Bid Response"""
    bid_id: str
    bid_price: float
    campaign_id: str
    latency_ms: float
    debug: Optional[Dict] = None


class RTBBidder:
    """
    통합 RTB Bidder Service

    Components:
    - Feature Store (Redis)
    - Model Server (ONNX)
    - Bid Logic (Value × Shade × Pace)
    """

    def __init__(self, feature_store: RTBFeatureStore,
                 model_server: RTBModelServer,
                 config: Dict):
        self.feature_store = feature_store
        self.model_server = model_server
        self.config = config

    def process_bid_request(self, request: BidRequest) -> Optional[BidResponse]:
        """Bid request 처리"""
        start_time = time.perf_counter()
        debug = {}

        try:
            # 1. Feature Lookup
            t0 = time.perf_counter()
            user_features = self.feature_store.get_user_features(request.user_id)
            campaign_features = self.feature_store.get_campaign_features(
                request.campaign_id
            )
            debug['feature_lookup_ms'] = (time.perf_counter() - t0) * 1000

            # 2. DR Weight 조회 (사전 계산된)
            dr_weight = self.feature_store.get_dr_weight(
                user_features.get('segment', 'default'),
                request.hour,
                request.exchange
            )
            debug['dr_weight'] = dr_weight

            # 3. Feature Vector 구성
            feature_vector = self._build_feature_vector(
                request, user_features, campaign_features
            )

            # 4. Model Prediction (ESCM²-WC)
            t0 = time.perf_counter()
            predictions = self.model_server.predict_escm2wc(feature_vector)
            debug['model_predict_ms'] = (time.perf_counter() - t0) * 1000
            debug.update(predictions)

            # 5. Value 계산: V(x) = debiased_pCTR × CPC_target
            cpc_target = campaign_features.get('target_cpc', 10)
            value = predictions['p_ctr'] * cpc_target
            debug['value'] = value

            # 6. Bid Shading
            market_dist = self.feature_store.get_market_distribution(
                request.hour, request.exchange
            )
            shade = self._calculate_shading(value, market_dist)
            debug['shade'] = shade

            # 7. Pacing
            remaining_budget = self.feature_store.get_budget_remaining(
                request.campaign_id,
                time.strftime('%Y-%m-%d')
            )
            pace = self._calculate_pacing(remaining_budget, request.hour)
            debug['pace'] = pace

            # 8. Final Bid
            bid_price = value * shade * pace
            bid_price = max(bid_price, request.floor_price)
            debug['final_bid'] = bid_price

            # 9. Budget Check
            if remaining_budget < bid_price:
                debug['no_bid_reason'] = 'insufficient_budget'
                return None

            # 10. Total latency
            total_ms = (time.perf_counter() - start_time) * 1000

            return BidResponse(
                bid_id=request.bid_id,
                bid_price=bid_price,
                campaign_id=request.campaign_id,
                latency_ms=total_ms,
                debug=debug
            )

        except Exception as e:
            debug['error'] = str(e)
            return None

    def _build_feature_vector(self, request, user_features, campaign_features):
        """Feature vector 구성"""
        import numpy as np

        features = []
        # User features
        features.append(user_features.get('historical_ctr', 0.01))
        features.append(user_features.get('historical_cvr', 0.005))
        # Context features
        features.append(request.hour)
        features.append(request.weekday)
        features.append(request.slot_width * request.slot_height)
        # ... 추가 features

        return np.array(features, dtype=np.float32)

    def _calculate_shading(self, value: float, market_dist: Dict) -> float:
        """Bid shading factor"""
        median_market = market_dist.get('median', 70)

        if value > median_market * 2:
            return 0.7
        elif value > median_market:
            return 0.8
        else:
            return 0.9

    def _calculate_pacing(self, remaining_budget: float, hour: int) -> float:
        """Budget pacing multiplier"""
        remaining_hours = max(24 - hour, 1)
        ideal_hourly = remaining_budget / remaining_hours

        # 간단한 규칙 기반
        if remaining_budget > ideal_hourly * 2:
            return 1.2
        elif remaining_budget < ideal_hourly * 0.5:
            return 0.7
        else:
            return 1.0

    def on_win_notification(self, bid_id: str, campaign_id: str,
                            winning_price: float):
        """Win notification 처리 (비동기)"""
        date = time.strftime('%Y-%m-%d')
        self.feature_store.decrement_budget(campaign_id, date, winning_price)
```

---

## Part E: Monitoring & Alerting

### E-1. Key Metrics

| 카테고리 | 메트릭 | 임계값 | 설명 |
|----------|--------|--------|------|
| **Latency** | P50 | <20ms | 중간값 |
| | P95 | <50ms | 95분위 |
| | P99 | <80ms | 99분위 |
| **Throughput** | QPS | >100K | 초당 처리량 |
| **Error** | Error rate | <0.1% | 오류율 |
| **Model** | Feature miss rate | <5% | Feature 조회 실패 |
| **Business** | Win rate | 10-30% | 낙찰률 |
| | Budget utilization | 80-100% | 예산 소진율 |

### E-2. Monitoring Implementation

```python
from collections import deque
import numpy as np
import time

class RTBMonitor:
    """RTB Serving 모니터링"""

    def __init__(self, window_size: int = 10000):
        self.latencies = deque(maxlen=window_size)
        self.errors = 0
        self.successes = 0
        self.bids = 0
        self.wins = 0

        # Alert thresholds
        self.thresholds = {
            'p95_latency_ms': 50,
            'p99_latency_ms': 80,
            'error_rate': 0.001
        }

    def record_request(self, latency_ms: float, success: bool, bid_made: bool):
        """요청 기록"""
        self.latencies.append(latency_ms)
        if success:
            self.successes += 1
        else:
            self.errors += 1
        if bid_made:
            self.bids += 1

    def record_win(self):
        """Win 기록"""
        self.wins += 1

    def get_metrics(self) -> Dict:
        """현재 메트릭"""
        latencies = list(self.latencies)
        total = self.successes + self.errors

        if not latencies:
            return {}

        return {
            'latency_p50_ms': np.percentile(latencies, 50),
            'latency_p95_ms': np.percentile(latencies, 95),
            'latency_p99_ms': np.percentile(latencies, 99),
            'error_rate': self.errors / max(total, 1),
            'bid_rate': self.bids / max(total, 1),
            'win_rate': self.wins / max(self.bids, 1)
        }

    def check_alerts(self) -> list:
        """Alert 확인"""
        metrics = self.get_metrics()
        alerts = []

        if metrics.get('latency_p95_ms', 0) > self.thresholds['p95_latency_ms']:
            alerts.append(f"High P95 latency: {metrics['latency_p95_ms']:.1f}ms")

        if metrics.get('error_rate', 0) > self.thresholds['error_rate']:
            alerts.append(f"High error rate: {metrics['error_rate']*100:.2f}%")

        return alerts
```

### E-3. Model Drift Detection

```python
from scipy import stats

class ModelDriftDetector:
    """Model drift 탐지"""

    def __init__(self, window_size: int = 100000):
        self.reference_predictions = None
        self.current_predictions = deque(maxlen=window_size)

    def set_reference(self, predictions: np.ndarray):
        """Reference 설정 (배포 시점)"""
        self.reference_predictions = predictions

    def add_prediction(self, prediction: float):
        """새 예측 추가"""
        self.current_predictions.append(prediction)

    def check_drift(self) -> Dict:
        """Drift 검사"""
        if self.reference_predictions is None:
            return {'status': 'no_reference'}

        if len(self.current_predictions) < 1000:
            return {'status': 'insufficient_data'}

        current = np.array(self.current_predictions)

        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(
            self.reference_predictions, current
        )

        drift_detected = ks_pvalue < 0.05

        return {
            'status': 'drift_detected' if drift_detected else 'ok',
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue
        }
```

### E-4. Calibrated Drift Monitoring (EDA-driven)

```
EDA Baseline:
- S2→S3 market price KS D=0.1294 (자연 temporal drift)
- 이 값을 drift monitoring의 calibrated threshold로 활용

Calibrated Thresholds:
- KS D > 0.10: ⚠️ Warning (자연 변동 수준 접근)
- KS D > 0.15: 🚨 Critical (자연 변동 초과 → retraining 트리거)
- KS D > 0.20: 🔴 Severe (모델 재학습 즉시 필요)

적용:
- pCTR prediction 분포, feature 분포 모두에 동일 threshold 적용
- ModelDriftDetector.check_drift()에 threshold 파라미터 추가
```

### E-5. Exchange-aware Cache TTL (EDA-driven)

```
EDA Finding:
- Exchange별 price 분포/경쟁 강도 상이
- Ex3: Floor-heavy → price 분포 안정적 (longer TTL 가능)
- Ex1: No floor → price 분포 변동 크다 (shorter TTL 필요)

Cache TTL 차등 설정:
| Exchange | 특성 | Feature Store TTL |
|----------|------|-------------------|
| Ex1 (no floor) | 변동 큼 | 15분 |
| Ex2 (moderate) | 중간 | 30분 |
| Ex3 (active floor) | 안정적 | 60분 |

적용: RTBFeatureStore.get_market_distribution()에서 exchange별 TTL 적용
```

### E-6. IVT Domain Blocklist (EDA-driven)

```
EDA Finding:
- 76 zero-win domains (7.16M bids, 전체의 5.5%)
- 이 도메인들에서 bid 요청은 처리하지만 낙찰 가능성 0%
- → 불필요한 latency/compute cost 발생

Production 적용:
1. Known IVT domain blocklist 유지 (Redis Set)
2. Bid request 수신 시 domain 체크 (O(1) lookup)
3. Blocklist 도메인 → bid 자체를 skip → latency/cost 절감
4. Blocklist 주기적 갱신 (weekly batch job)

기대 효과:
- ~5.5% bid 요청 즉시 skip
- Model inference 비용 절감
- P95 latency 개선
```

---

## Part F: 실무적 문제

### F-1. DR Weight 실시간 계산 불가

```
문제:
- DR weight = f(propensity, outcome_model)
- Propensity 계산: 추가 모델 추론 필요
- 실시간으로 하면 latency 초과

해결:
1. 배치 사전 계산 (매 시간)
2. Segment별 평균 weight 저장
3. Serving 시 lookup만 수행

Trade-off:
- 정확도 ↓ (개별 weight가 아닌 segment 평균)
- 속도 ↑ (lookup only)
```

### F-2. Feature 조회 실패 시 Graceful Degradation

```python
def get_features_with_fallback(feature_store, request):
    """Feature 조회 실패 시 fallback"""
    features = {}

    # User features
    try:
        features['user'] = feature_store.get_user_features(request.user_id)
    except:
        features['user'] = DEFAULT_USER_FEATURES  # Fallback

    # DR weight
    try:
        features['dr_weight'] = feature_store.get_dr_weight(...)
    except:
        features['dr_weight'] = 1.0  # Neutral weight

    return features
```

### F-3. Canary Deployment

```
Model 업데이트 시 Canary 배포:

1. Canary (5% traffic)
   - 새 모델 적용
   - 메트릭 모니터링

2. 검증 (24시간)
   - Latency OK?
   - Win rate 유지?
   - Error rate OK?

3. Gradual rollout
   - 5% → 25% → 50% → 100%

4. Rollback trigger
   - P95 latency > 80ms (5분 지속)
   - Error rate > 1%
   - Win rate 50% 이상 하락
```

---

## 산출물

| 산출물 | 경로 | 설명 |
|--------|------|------|
| Feature Store | `src/serving/feature_store.py` | Redis 기반 |
| Model Server | `src/serving/model_server.py` | ONNX Runtime |
| RTBBidder | `src/serving/rtb_bidder.py` | 통합 Bidder |
| Monitoring | `src/serving/monitoring.py` | 모니터링 |
| ONNX 모델 | `models/escm2wc.onnx` | 배포용 모델 |
| Docker | `docker/Dockerfile` | 컨테이너화 |

---

## 핵심 요약

1. **Latency**: <50ms (P95), 각 컴포넌트 budget 관리
2. **Feature Store**: Redis 기반, DR weight 사전 계산
3. **Model Serving**: ONNX + Quantization, Warmup 필수
4. **DR Weight**: 실시간 계산 불가 → 배치 사전 계산
5. **Monitoring**: Latency, Error rate, Drift 실시간 감시
6. **Deployment**: Canary → Gradual rollout
