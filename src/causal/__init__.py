"""Causal inference modules for RTB (SP4).

Modules:
  - cate: Multi-outcome CATE estimation (CausalForestDML)
  - scm: DAG, DoWhy estimation, refutation, model-based counterfactual
"""

from .cate import (
    ADVERTISER_TAXONOMY,
    DEFAULT_OUTCOMES,
    CATEConfig,
    CATEResult,
    MediationResult,
    MultiOutcomeCATEResult,
    OutcomeSpec,
    assign_advertiser_taxonomy,
    compute_segment_cate_summary,
    StratifiedCATEResult,
    TLearnerCATEResult,
    build_strata,
    estimate_cate,
    estimate_mediation,
    estimate_multi_outcome_cate,
    estimate_stratified_cate,
    estimate_tlearner_cate,
    prepare_cate_data,
    validate_decomposition,
)
from .scm import (
    CounterfactualResult,
    DAGSpec,
    RefutationResult,
    SCMResult,
    build_rtb_dag,
    estimate_causal_effect,
    run_refutation_tests,
    simulate_counterfactual,
    simulate_counterfactual_scenarios,
)
