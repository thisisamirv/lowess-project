"""Type stubs for fastlowess._core native extension module."""

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

def smooth(
    x: ArrayLike,
    y: ArrayLike,
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
) -> LowessResult:
    """LOWESS smoothing with the batch adapter.

    Parameters
    ----------
    x : array_like
        Independent variable values.
    y : array_like
        Dependent variable values.
    fraction : float, optional
        Smoothing fraction (default: 0.67).
    iterations : int, optional
        Number of robustness iterations (default: 3).
    delta : float, optional
        Interpolation optimization threshold.
    weight_function : str, optional
        Kernel function: "tricube", "epanechnikov", "gaussian", "uniform",
        "biweight", "triangle", "cosine".
    robustness_method : str, optional
        Robustness method: "bisquare", "huber", "talwar".
    scaling_method : str, optional
        Scaling method: "mad", "mar" (default: "mad").
    boundary_policy : str, optional
        Boundary policy: "extend", "reflect", "zero", "noboundary".
    confidence_intervals : float, optional
        Confidence level for confidence intervals (e.g., 0.95).
    prediction_intervals : float, optional
        Confidence level for prediction intervals (e.g., 0.95).
    return_diagnostics : bool, optional
        Whether to compute diagnostics (default: False).
    return_residuals : bool, optional
        Whether to include residuals (default: False).
    return_robustness_weights : bool, optional
        Whether to include robustness weights (default: False).
    zero_weight_fallback : str, optional
        Fallback when all weights are zero.
    auto_converge : float, optional
        Tolerance for auto-convergence.
    cv_fractions : sequence of float, optional
        Fractions to test for cross-validation.
    cv_method : str, optional
        CV method: "loocv" or "kfold" (default: "kfold").
    cv_k : int, optional
        Number of folds for k-fold CV (default: 5).
    parallel : bool, optional
        Enable parallel execution (default: True).

    Returns
    -------
    LowessResult
        Result object with smoothed values and optional outputs.
    """
    ...

def smooth_streaming(
    x: ArrayLike,
    y: ArrayLike,
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
) -> LowessResult:
    """Streaming LOWESS for large datasets.

    Parameters
    ----------
    x : array_like
        Independent variable values.
    y : array_like
        Dependent variable values.
    fraction : float, optional
        Smoothing fraction (default: 0.3).
    chunk_size : int, optional
        Size of each processing chunk (default: 5000).
    overlap : int, optional
        Overlap between chunks (default: 10% of chunk_size).
    iterations : int, optional
        Number of robustness iterations (default: 3).
    delta : float, optional
        Interpolation optimization threshold.
    weight_function : str, optional
        Kernel function.
    robustness_method : str, optional
        Robustness method.
    scaling_method : str, optional
        Scaling method.
    boundary_policy : str, optional
        Boundary policy.
    auto_converge : float, optional
        Tolerance for auto-convergence.
    return_diagnostics : bool, optional
        Whether to compute diagnostics.
    return_residuals : bool, optional
        Whether to include residuals.
    return_robustness_weights : bool, optional
        Whether to include robustness weights.
    zero_weight_fallback : str, optional
        Fallback when all weights are zero.
    parallel : bool, optional
        Enable parallel execution (default: True).

    Returns
    -------
    LowessResult
        Result object with smoothed values.
    """
    ...

def smooth_online(
    x: ArrayLike,
    y: ArrayLike,
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
) -> LowessResult:
    """Online LOWESS with sliding window.

    Parameters
    ----------
    x : array_like
        Independent variable values.
    y : array_like
        Dependent variable values.
    fraction : float, optional
        Smoothing fraction (default: 0.2).
    window_capacity : int, optional
        Maximum points to retain in window (default: 100).
    min_points : int, optional
        Minimum points before smoothing starts (default: 2).
    iterations : int, optional
        Number of robustness iterations (default: 3).
    delta : float, optional
        Interpolation optimization threshold.
    weight_function : str, optional
        Kernel function.
    robustness_method : str, optional
        Robustness method.
    scaling_method : str, optional
        Scaling method.
    boundary_policy : str, optional
        Boundary policy.
    update_mode : str, optional
        Update strategy: "full" or "incremental" (default: "full").
    auto_converge : float, optional
        Tolerance for auto-convergence.
    return_robustness_weights : bool, optional
        Whether to include robustness weights.
    zero_weight_fallback : str, optional
        Fallback when all weights are zero.
    parallel : bool, optional
        Enable parallel execution (default: False).

    Returns
    -------
    LowessResult
        Result object with smoothed values.
    """
    ...
