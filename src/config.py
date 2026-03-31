"""Configuration and result types using NamedTuple pattern."""

from typing import NamedTuple, Optional, Tuple, List, Dict, Any


# =============================================================================
# Data Configurations
# =============================================================================

class DataConfig(NamedTuple):
    """Data loading and preprocessing configuration."""
    data_dir: str = "data/ipinyou"
    output_dir: str = "results/processed"
    seasons: Tuple[str, ...] = ("2", "3")
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42


class FeatureConfig(NamedTuple):
    """Feature engineering configuration."""
    # Categorical features (embedding lookup / LightGBM categorical)
    categorical_features: Tuple[str, ...] = (
        "region", "city", "adexchange", "advertiser",
        "slotwidth", "slotheight", "slotvisibility", "slotformat", "slot_size_group",
        "hour", "minute", "weekday",
        "is_weekend", "is_peak_hour", "region_group",
    )
    # Numerical features (Z-score normalized for neural models)
    numerical_features: Tuple[str, ...] = (
        "bidprice", "slotprice", "bid_floor_ratio",
        "slot_area", "slot_area_log", "slot_aspect_ratio",
        "hour_sin", "hour_cos", "region_freq",
    )
    # Usertag multi-hot encoding
    usertag_vocab_size: int = 10000
    # Embedding dimensions
    embedding_dim: int = 16
    # Unknown token index
    unk_idx: int = 0
    # Normalization method for neural models
    normalization_method: str = "zscore"


# =============================================================================
# Training Configurations
# =============================================================================

class TrainingConfig(NamedTuple):
    """Training configuration."""
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 50
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    early_stopping_patience: int = 5
    # Checkpointing
    checkpoint_dir: str = "results/models"
    save_every_n_epochs: int = 5
    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = True
    wandb_project: str = "rtb-ipinyou"
    wandb_run_name: Optional[str] = None
    eval_every: int = 1
    quiet: bool = False


class OptimizerConfig(NamedTuple):
    """Optimizer configuration."""
    optimizer: str = "adamw"  # 'adam', 'adamw', 'sgd'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    # Scheduler
    scheduler: str = "cosine"  # 'constant', 'cosine', 'linear'
    warmup_steps: int = 1000
    total_steps: int = 100000


# =============================================================================
# Distributed Training Configuration
# =============================================================================

class DistributedConfig(NamedTuple):
    """Distributed training configuration for JAX SPMD."""
    enabled: bool = False                          # False = single-device (backward compatible)
    num_devices: Optional[int] = None              # None = auto-detect all devices
    mesh_axis_name: str = "data"
    per_device_batch_size: int = 4096              # global = per_device x num_devices
    lr_scaling: str = "linear"                     # 'linear', 'sqrt', 'none'
    warmup_steps: int = 1000
    scheduler: str = "cosine"                      # 'constant', 'cosine', 'linear'
    gradient_clip: float = 1.0
    weight_decay: float = 1e-5
    checkpoint_enabled: bool = True
    checkpoint_dir: str = "results/checkpoints"
    checkpoint_every_n_epochs: int = 5
    resume_from: Optional[str] = None
    drop_last_batch: bool = True


# =============================================================================
# Serving Configurations
# =============================================================================

class ServingConfig(NamedTuple):
    """Serving configuration for FastAPI + ONNX."""
    model_path: str = "results/models/escm2_wc.onnx"
    host: str = "0.0.0.0"
    port: int = 8000
    # Latency targets
    p50_target_ms: float = 20.0
    p95_target_ms: float = 50.0
    p99_target_ms: float = 80.0
    # ONNX optimization
    use_quantization: bool = True
    quantization_type: str = "int8"  # 'int8' or 'fp16'
    # Feature store
    feast_repo_path: str = "configs/feast"
    redis_host: str = "localhost"
    redis_port: int = 6379


class BiddingConfig(NamedTuple):
    """Bidding strategy configuration.

    All monetary values in CPM units.
    CPC_target = 200,000 CPM/click (NB05 Section 11 기준).
    """
    # Value computation (CPC-based, CVR near-trivial)
    goal_type: str = "CPC"                      # CPC, CPA, CPM
    cpc_target: float = 200_000.0               # CPM per click
    # Bid shading
    shading_strategy: str = "optimal"           # optimal, linear, percentile, dual_regime
    shading_factor: float = 0.8                 # for linear strategy
    min_bid: float = 1.0
    max_bid: float = 300.0
    exchange_conditional: bool = True
    # Budget pacing
    daily_budget: float = 10_000.0
    pacing_type: str = "pid"                    # pid, throttle, uniform, wr_weighted
    pid_kp: float = 0.5
    pid_ki: float = 0.1
    pid_kd: float = 0.1
    multiplier_range: Tuple[float, float] = (0.3, 2.0)
    # Simulation
    auction_type: str = "first_price"           # first_price, second_price


# =============================================================================
# Result Types
# =============================================================================

class WCPredictionMetrics(NamedTuple):
    """Bid→Win→Click prediction metrics.

    WCTR = Win-Click Through Rate = P(Win) × P(Click|Win)
    원 ESCM² 논문의 CTCVR에 해당하는 Bid→Win→Click 퍼널 버전.
    """
    win_auc: float       # P(Win|X, bid) AUC — all bids
    ctr_auc: float       # P(Click|X, Win=1) AUC — won impressions only (biased)
    wctr_auc: float      # P(Win) × P(Click|Win) AUC — all bids
    ctr_ece: float       # CTR calibration error (won impressions)
    wctr_ece: float      # WCTR calibration error (all bids)
    ctr_ieb: float       # CTR Inherent Estimation Bias: |E[p_ctr] - E[click|win=1]| / E[click|win=1]
    wctr_ieb: float      # WCTR IEB: |E[wctr] - E[click]| / E[click]


class BiddingMetrics(NamedTuple):
    """Bidding performance metrics."""
    win_rate: float
    avg_cpm: float
    avg_cpc: float
    total_spend: float
    total_clicks: int
    roi: float


class LatencyMetrics(NamedTuple):
    """Serving latency metrics."""
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float
    error_rate: float
