"""EDA utility modules.

Provides parallel/optimized operations for exploratory data analysis:
- parallel_groupbys(): Run independent groupby operations concurrently
- parse_weekday_from_timestamp(): Optimized weekday extraction via unique-value map
"""

from .parallel_ops import (
    GroupbySpec,
    GroupbyResults,
    parallel_groupbys,
    parse_weekday_from_timestamp,
)

__all__ = [
    "GroupbySpec",
    "GroupbyResults",
    "parallel_groupbys",
    "parse_weekday_from_timestamp",
]
