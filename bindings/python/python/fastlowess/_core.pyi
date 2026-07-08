"""Type stubs for fastlowess._core native extension module."""

# pylint: disable=unnecessary-ellipsis,unused-argument

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

class Diagnostics:
    """Diagnostic statistics for LOWESS fit quality."""

    @property
    def rmse(self) -> float:
        """Root Mean Squared Error."""
        ...

    @property
    def mae(self) -> float:
        """Mean Absolute Error."""
        ...

    @property
    def r_squared(self) -> float:
        """R-squared (coefficient of determination)."""
        ...

    @property
    def aic(self) -> float | None:
        """Akaike Information Criterion."""
        ...

    @property
    def aicc(self) -> float | None:
        """Corrected AIC."""
        ...

    @property
    def effective_df(self) -> float | None:
        """Effective degrees of freedom."""
        ...

    @property
    def residual_sd(self) -> float:
        """Residual standard deviation."""
        ...

class LowessResult:
    """Result from LOWESS smoothing."""

    @property
    def x(self) -> NDArray[np.float64]:
        """Sorted x values."""
        ...

    @property
    def y(self) -> NDArray[np.float64]:
        """Smoothed y values."""
        ...

    @property
    def standard_errors(self) -> NDArray[np.float64] | None:
        """Standard errors (if computed)."""
        ...

    @property
    def confidence_lower(self) -> NDArray[np.float64] | None:
        """Lower confidence interval bounds."""
        ...

    @property
    def confidence_upper(self) -> NDArray[np.float64] | None:
        """Upper confidence interval bounds."""
        ...

    @property
    def prediction_lower(self) -> NDArray[np.float64] | None:
        """Lower prediction interval bounds."""
        ...

    @property
    def prediction_upper(self) -> NDArray[np.float64] | None:
        """Upper prediction interval bounds."""
        ...

    @property
    def residuals(self) -> NDArray[np.float64] | None:
        """Residuals (original y - smoothed y)."""
        ...

    @property
    def robustness_weights(self) -> NDArray[np.float64] | None:
        """Robustness weights from final iteration."""
        ...

    @property
    def diagnostics(self) -> Diagnostics | None:
        """Diagnostic metrics."""
        ...

    @property
    def iterations_used(self) -> int | None:
        """Number of iterations performed."""
        ...

    @property
    def fraction_used(self) -> float:
        """Fraction used for smoothing."""
        ...

    @property
    def cv_scores(self) -> NDArray[np.float64] | None:
        """CV scores for tested fractions."""
        ...

class Lowess:
    """Batch LOWESS model — configure once, fit many times."""

    def __init__(
        self,
        fraction: float = 0.67,
        iterations: int = 3,
        delta: float | None = None,
        weight_function: str = "tricube",
        robustness_method: str = "bisquare",
        scaling_method: str = "mad",
        boundary_policy: str = "extend",
        confidence_intervals: float | None = None,
        prediction_intervals: float | None = None,
        return_diagnostics: bool = False,
        return_residuals: bool = False,
        return_robustness_weights: bool = False,
        zero_weight_fallback: str = "use_local_mean",
        auto_converge: float | None = None,
        cv_fractions: Sequence[float] | None = None,
        cv_method: str = "kfold",
        cv_k: int = 5,
        parallel: bool = True,
    ) -> None: ...
    def fit(self, x: ArrayLike, y: ArrayLike) -> LowessResult: ...

class StreamingLowess:
    """Streaming LOWESS processor for incremental chunk-based smoothing."""

    def __init__(
        self,
        fraction: float = 0.3,
        chunk_size: int = 5000,
        overlap: int | None = None,
        iterations: int = 3,
        delta: float | None = None,
        weight_function: str = "tricube",
        robustness_method: str = "bisquare",
        scaling_method: str = "mad",
        boundary_policy: str = "extend",
        auto_converge: float | None = None,
        return_diagnostics: bool = False,
        return_residuals: bool = False,
        return_robustness_weights: bool = False,
        zero_weight_fallback: str = "use_local_mean",
        parallel: bool = True,
    ) -> None: ...
    def process_chunk(self, x: ArrayLike, y: ArrayLike) -> LowessResult: ...
    def finalize(self) -> LowessResult: ...

class OnlineLowess:
    """Online LOWESS processor for real-time data streams."""

    def __init__(
        self,
        fraction: float = 0.2,
        window_capacity: int = 100,
        min_points: int = 2,
        iterations: int = 3,
        delta: float | None = None,
        weight_function: str = "tricube",
        robustness_method: str = "bisquare",
        scaling_method: str = "mad",
        boundary_policy: str = "extend",
        update_mode: str = "full",
        auto_converge: float | None = None,
        return_robustness_weights: bool = False,
        zero_weight_fallback: str = "use_local_mean",
        parallel: bool = False,
    ) -> None: ...
    def add_point(self, x: float, y: float) -> float | None: ...
