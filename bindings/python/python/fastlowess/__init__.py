"""
fastlowess: High-performance LOWESS Smoothing for Python.

A high-performance LOWESS (Locally Weighted Scatterplot Smoothing) implementation
with parallel execution via Rayon and NumPy integration. Built on top of the
fastLowess Rust crate.
"""

from .__version__ import __version__

from ._core import (
    LowessResult,
    Diagnostics,
    Lowess,
    OnlineLowess,
    StreamingLowess,
)

__all__ = [
    "LowessResult",
    "Diagnostics",
    "Lowess",
    "OnlineLowess",
    "StreamingLowess",
    "__version__",
]
