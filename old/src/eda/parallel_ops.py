"""Parallel and optimized operations for EDA notebooks.

Provides:
- parallel_groupbys(): Run independent groupby aggregations concurrently
  using ThreadPoolExecutor (pandas groupby releases GIL during C-level ops).
- parse_weekday_from_timestamp(): Extract weekday from timestamp strings
  using unique-value mapping (~30 unique dates vs 65M rows).
"""

from typing import Any, Callable, Dict, List, NamedTuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Parallel Groupby
# =============================================================================

class GroupbySpec(NamedTuple):
    """Specification for a single groupby operation."""
    name: str
    by: Any                                  # groupby key(s)
    agg_spec: Dict[str, Any]                 # kwargs for .agg()
    post_fn: Optional[Callable] = None       # optional post-processing
    sort_by: Optional[str] = None            # optional sort column
    ascending: bool = False
    head_n: Optional[int] = None             # optional .head(n)


class GroupbyResults(NamedTuple):
    """Container for parallel groupby results."""
    results: Dict[str, pd.DataFrame]
    n_specs: int
    n_success: int


def _execute_single_groupby(
    df: pd.DataFrame,
    spec: GroupbySpec,
) -> pd.DataFrame:
    """Execute a single groupby spec and return the result DataFrame."""
    result = df.groupby(spec.by).agg(**spec.agg_spec)

    if spec.post_fn is not None:
        result = spec.post_fn(result)

    if spec.sort_by is not None:
        result = result.sort_values(spec.sort_by, ascending=spec.ascending)

    if spec.head_n is not None:
        result = result.head(spec.head_n)

    return result


def parallel_groupbys(
    df: pd.DataFrame,
    specs: List[GroupbySpec],
    max_workers: Optional[int] = None,
) -> GroupbyResults:
    """Run independent groupby operations concurrently via ThreadPoolExecutor.

    Pandas groupby with built-in aggregations (sum, count, mean) releases
    the GIL during C-level execution, so threads achieve real parallelism
    without Ray serialization overhead (the 65M-row DataFrame stays in
    shared process memory).

    Falls back to sequential execution if len(specs) <= 1.

    Args:
        df: Input DataFrame (shared across all groupbys)
        specs: List of GroupbySpec defining each aggregation
        max_workers: Thread pool size (default: len(specs))

    Returns:
        GroupbyResults with results dict keyed by spec.name
    """
    if len(specs) <= 1:
        results = {}
        for spec in specs:
            results[spec.name] = _execute_single_groupby(df, spec)
        return GroupbyResults(
            results=results, n_specs=len(specs), n_success=len(specs)
        )

    if max_workers is None:
        max_workers = len(specs)

    logger.info(f"Running {len(specs)} groupbys in parallel ({max_workers} threads)")

    results: Dict[str, pd.DataFrame] = {}
    n_success = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(_execute_single_groupby, df, spec): spec.name
            for spec in specs
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
                n_success += 1
            except Exception as e:
                logger.error(f"Groupby '{name}' failed: {e}")

    return GroupbyResults(
        results=results, n_specs=len(specs), n_success=n_success
    )


# =============================================================================
# Optimized Datetime Parsing
# =============================================================================

def parse_weekday_from_timestamp(
    series: pd.Series,
    date_slice: slice = slice(None, 8),
    date_format: str = "%Y%m%d",
) -> pd.Series:
    """Extract day-of-week from timestamp strings via unique-value mapping.

    The iPinYou dataset spans ~30-60 unique dates across 65M+ rows.
    Instead of calling pd.to_datetime on every row, we:
    1. Extract the date substring (first 8 chars of timestamp)
    2. Get unique dates (~30 values)
    3. Parse only unique dates → weekday
    4. Map back to full Series (vectorized)

    Args:
        series: Series of timestamp strings (e.g., "20130607000000123")
        date_slice: Slice to extract date portion (default: first 8 chars)
        date_format: strftime format of the date portion

    Returns:
        Series of int weekday values (0=Mon, 6=Sun)
    """
    date_strs = series.astype(str).str[date_slice]

    unique_dates = date_strs.unique()
    logger.info(
        f"Mapping {len(series):,} rows via {len(unique_dates)} unique dates"
    )

    weekday_map = {
        d: pd.Timestamp(d).dayofweek
        for d in unique_dates
        if d and d != "nan"
    }

    return date_strs.map(weekday_map)
