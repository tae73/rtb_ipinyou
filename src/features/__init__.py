"""Feature engineering modules.

Provides functions for:
- Time features (hour, weekday, is_peak, cyclical encoding)
- Slot features (area, aspect ratio, size group)
- Region features (frequency encoding, CTR by region)
- Competition features (bid aggressiveness, market price stats)
- Usertag features (multi-hot, hashing, embeddings)
"""

from .engineering import (
    # Time features
    add_time_features,
    # Slot features
    add_slot_features,
    # Region features
    add_region_features,
    compute_region_stats,
    # Competition features
    add_competition_features,
    compute_market_stats,
    # Main pipeline
    engineer_features,
    get_feature_info,
    # Data splitting
    split_temporal,
    split_by_days,
    # I/O
    save_feature_splits,
    load_feature_splits,
    # Target encoding
    target_encode_kfold,
    # Types
    FeatureInfo,
    TargetEncodingResult,
)

from .usertag import (
    # Parsing
    parse_usertag,
    parse_usertag_series,
    parse_usertag_series_parallel,
    # Vocabulary
    build_vocab,
    save_vocab,
    load_vocab,
    # Encoding
    encode_multihot,
    encode_multihot_sparse,
    encode_count,
    encode_hashing,
    # Feature columns
    add_usertag_features,

    # Analysis
    compute_tag_stats,
    compute_tag_stats_parallel,
    compute_coverage,
    # Types
    UsertagVocab,
    UsertagEncodingResult,
    ParseResult,
)

__all__ = [
    # Engineering
    "add_time_features",
    "add_slot_features",
    "add_region_features",
    "compute_region_stats",
    "add_competition_features",
    "compute_market_stats",
    "engineer_features",
    "get_feature_info",
    "split_temporal",
    "split_by_days",
    "save_feature_splits",
    "load_feature_splits",
    "target_encode_kfold",
    "FeatureInfo",
    "TargetEncodingResult",
    # Usertag
    "parse_usertag",
    "parse_usertag_series",
    "parse_usertag_series_parallel",
    "build_vocab",
    "save_vocab",
    "load_vocab",
    "encode_multihot",
    "encode_multihot_sparse",
    "encode_count",
    "encode_hashing",
    "add_usertag_features",

    "compute_tag_stats",
    "compute_tag_stats_parallel",
    "compute_coverage",
    "UsertagVocab",
    "UsertagEncodingResult",
    "ParseResult",
]
