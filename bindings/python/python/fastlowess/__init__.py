"""
fastlowess: High-performance LOWESS Smoothing for Python.

A high-performance LOWESS (Locally Weighted Scatterplot Smoothing) implementation
with parallel execution via Rayon and NumPy integration. Built on top of the
fastLowess Rust crate.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np

from .__version__ import __version__

if TYPE_CHECKING:
    from ._core import Diagnostics as _CoreDiagnostics
    from ._core import LowessResult as _CoreLowessResult

    Diagnostics = _CoreDiagnostics
    LowessResult = _CoreLowessResult
else:
    _core = import_module("._core", __package__)
    Diagnostics = getattr(_core, "Diagnostics")
    LowessResult = getattr(_core, "LowessResult")

_Lowess = getattr(import_module("._core", __package__), "Lowess")
_OnlineLowess = getattr(import_module("._core", __package__), "OnlineLowess")
_StreamingLowess = getattr(import_module("._core", __package__), "StreamingLowess")


def _as_float64_vector(values: Any, name: str) -> np.ndarray:
    """Convert an arbitrary 1D array-like input into a float64 NumPy array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise TypeError(f"argument '{name}' must be 1-dimensional")
    return array


class Lowess:
    """Python wrapper that accepts array-like inputs for batch LOWESS fitting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create a LOWESS model with the same options as the native binding."""
        self._inner = _Lowess(*args, **kwargs)

    def fit(self, x: Any, y: Any) -> LowessResult:
        """Fit the model after coercing `x` and `y` to 1D float64 arrays."""
        return self._inner.fit(_as_float64_vector(x, "x"), _as_float64_vector(y, "y"))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return repr(self._inner)


class StreamingLowess:
    """Python wrapper that accepts array-like inputs for streaming LOWESS."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create a streaming LOWESS processor with the native binding options."""
        self._inner = _StreamingLowess(*args, **kwargs)

    def process_chunk(self, x: Any, y: Any) -> LowessResult:
        """Process one chunk after coercing `x` and `y` to 1D float64 arrays."""
        return self._inner.process_chunk(
            _as_float64_vector(x, "x"),
            _as_float64_vector(y, "y"),
        )

    def finalize(self) -> LowessResult:
        """Flush buffered streaming state and return the final LOWESS result."""
        return self._inner.finalize()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return repr(self._inner)


class OnlineLowess:
    """Python wrapper that accepts array-like inputs for online LOWESS updates."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create an online LOWESS processor with the native binding options."""
        self._inner = _OnlineLowess(*args, **kwargs)

    def update(self, x: float, y: float) -> float | None:
        """Update the model with a single point and return a smoothed value if ready."""
        return self._inner.update(x, y)

    def add_points(self, x: Any, y: Any) -> LowessResult:
        """Add a batch of points after coercing `x` and `y` to 1D float64 arrays."""
        return self._inner.add_points(
            _as_float64_vector(x, "x"),
            _as_float64_vector(y, "y"),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return repr(self._inner)


__all__ = [
    "LowessResult",
    "Diagnostics",
    "Lowess",
    "OnlineLowess",
    "StreamingLowess",
    "__version__",
]
