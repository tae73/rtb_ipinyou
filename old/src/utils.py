"""General utility functions for RTB pipeline.

This module provides:
- Timer decorator and context manager for performance measurement
- Common helper functions
"""

from typing import Callable, Optional
from functools import wraps
import time


# =============================================================================
# Timer Utilities
# =============================================================================

def format_duration(seconds: float) -> str:
    """Format duration in h:m:s format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "0:01:23" or "1:30:45"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def timer(func: Callable) -> Callable:
    """Decorator to measure and print function execution time.

    Prints timing in h:m:s format after function completes.

    Example:
        >>> @timer
        ... def slow_function():
        ...     time.sleep(2)
        >>> slow_function()
        [slow_function] completed in 0:00:02
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"[{func.__name__}] completed in {format_duration(elapsed)}")
        return result
    return wrapper


def timer_verbose(name: Optional[str] = None) -> Callable:
    """Decorator factory for timer with custom name.

    Args:
        name: Custom name to display (default: function name)

    Example:
        >>> @timer_verbose("Data Loading")
        ... def load_data():
        ...     pass
        >>> load_data()
        [Data Loading] completed in 0:00:01
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            display_name = name or func.__name__
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"[{display_name}] completed in {format_duration(elapsed)}")
            return result
        return wrapper
    return decorator


class TimerContext:
    """Context manager for timing code blocks.

    Example:
        >>> with TimerContext("Processing"):
        ...     time.sleep(1)
        [Processing] completed in 0:00:01

        >>> with TimerContext("Task") as t:
        ...     time.sleep(1)
        >>> print(t.elapsed)  # Access elapsed time
        1.00123
    """

    def __init__(self, name: str = "Block"):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        print(f"[{self.name}] completed in {format_duration(self.elapsed)}")
