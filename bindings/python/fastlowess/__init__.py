"""fastlowess: High-performance LOWESS Smoothing for Python.

A high-performance LOWESS (Locally Weighted Scatterplot Smoothing) implementation
with parallel execution via Rayon and NumPy integration. Built on top of the
fastLowess Rust crate.

What is LOWESS?
---------------
LOWESS (Locally Weighted Scatterplot Smoothing) is a nonparametric regression
method that fits smooth curves through scatter plots. At each point, it fits
a weighted polynomial (typically linear) using nearby data points, with weights
decreasing smoothly with distance. This creates flexible, data-adaptive curves
without assuming a global functional form.

**Key advantages:**

- No parametric assumptions about the underlying relationship
- Automatic adaptation to local data structure
- Robust to outliers (with robustness iterations enabled)
- Provides uncertainty estimates via confidence/prediction intervals
- Handles irregular sampling and missing regions gracefully

**Common applications:**

- Exploratory data analysis and visualization
- Trend estimation in time series
- Baseline correction in spectroscopy and signal processing
- Quality control and process monitoring
- Genomic and epigenomic data smoothing
- Removing systematic effects in scientific measurements

Quick Start
-----------
**Basic Smoothing:**

>>> import numpy as np
>>> import fastlowess
>>>
>>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
>>> y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])
>>>
>>> # Simple smoothing
>>> result = fastlowess.smooth(x, y, fraction=0.5)
>>> print(result.y)
[2.   4.1  5.9  8.2  9.8]

**Full-Featured LOWESS:**

>>> result = fastlowess.smooth(
...     x, y,
...     fraction=0.5,
...     iterations=3,
...     confidence_intervals=0.95,
...     prediction_intervals=0.95,
...     return_diagnostics=True,
... )
>>> print(f"Smoothed: {result.y}")
>>> print(f"R²: {result.diagnostics.r_squared:.4f}")

Functions
---------

smooth(x, y, fraction=0.67, iterations=3, delta=None, weight_function="tricube",
    robustness_method="bisquare", scaling_method="mad", boundary_policy="extend",
    confidence_intervals=None, prediction_intervals=None,
    return_diagnostics=False, return_residuals=False,
    return_robustness_weights=False, zero_weight_fallback="use_local_mean",
    auto_converge=None, cv_fractions=None, cv_method="kfold", cv_k=5,
    parallel=True)
    LOWESS smoothing with the batch adapter.

    This is the primary interface for LOWESS smoothing. Processes the entire
    dataset in memory with optional parallel execution.

    **Parameters**

    x : array_like
        Independent variable values (1D array).
    y : array_like
        Dependent variable values (1D array, same length as x).
    fraction : float, optional
        Smoothing fraction - the proportion of data used for each local fit.
        Range: (0, 1]. Default: 0.67.

        - 0.1-0.3: Fine detail, may be noisy
        - 0.3-0.5: Moderate smoothing (good for most cases)
        - 0.5-0.7: Heavy smoothing, emphasizes trends
        - 0.7-1.0: Very smooth, may over-smooth

    iterations : int, optional
        Number of robustness iterations for outlier resistance.
        Default: 3.

        - 0: No robustness (fastest, sensitive to outliers)
        - 1-3: Light to moderate robustness (recommended)
        - 4-6: Strong robustness (for contaminated data)
        - 7+: Very strong (may over-smooth)

    delta : float, optional
        Interpolation optimization threshold. Points within delta distance
        reuse the previous fit. Larger values are faster but less accurate.
        Default: None (automatic: 1% of x-range for large datasets).
    weight_function : str, optional
        Kernel function for distance weighting. Options:

        - "tricube" (default): Classic LOWESS kernel
        - "epanechnikov": Optimal MSE kernel
        - "gaussian": Smooth, infinite support
        - "uniform": Equal weights within neighborhood
        - "biweight": Similar to tricube
        - "triangle": Linear decay
        - "cosine": Smooth cosine weighting

    robustness_method : str, optional
        Method for downweighting outliers. Options:

        - "bisquare" (default): Classic robust method
        - "huber": Less aggressive downweighting
        - "talwar": Hard rejection of outliers

    scaling_method : str, optional
        Method for scale estimation. Options:

        - "mad" (default): Median Absolute Deviation (robust to 50% outliers)
        - "mar": Median Absolute Residual (classic statsmodels behavior)

    boundary_policy : str, optional
        Handling of edge effects (default: "extend"). Options:

        - "extend": Extend boundary values (retains trend)
        - "reflect": Reflect values around boundary
        - "zero": Pad with zeros
        - "noboundary": No padding (may produce artifacts at edges)

    confidence_intervals : float, optional
        Confidence level for confidence intervals (e.g., 0.95 for 95%).
        If provided, populates result.confidence_lower and result.confidence_upper.
        Default: None (disabled).
    prediction_intervals : float, optional
        Confidence level for prediction intervals (e.g., 0.95 for 95%).
        If provided, populates result.prediction_lower and result.prediction_upper.
        Default: None (disabled).
    return_diagnostics : bool, optional
        Whether to compute diagnostic statistics (RMSE, MAE, R², etc.).
        If True, populates result.diagnostics. Default: False.
    return_residuals : bool, optional
        Whether to include residuals in output.
        If True, populates result.residuals. Default: False.
    return_robustness_weights : bool, optional
        Whether to include final robustness weights in output.
        If True, populates result.robustness_weights. Default: False.
    zero_weight_fallback : str, optional
        Behavior when all neighborhood weights are zero. Options:

        - "use_local_mean" (default): Use mean of neighborhood
        - "return_original": Return original y value
        - "return_none": Return NaN (for explicit handling)

    auto_converge : float, optional
        Tolerance for auto-convergence. When set, iterations stop when
        the maximum change between iterations is below this threshold.
        Default: None (disabled).
    cv_fractions : list of float, optional
        Fractions to test for cross-validation. When provided, enables
        cross-validation to select optimal fraction. The result's
        `fraction_used` will contain the selected fraction and `cv_scores`
        will contain the scores for each tested fraction.
        Default: None (disabled).
    cv_method : str, optional
        Cross-validation method: "loocv" (leave-one-out) or "kfold".
        Default: "kfold".
    cv_k : int, optional
        Number of folds for k-fold CV. Default: 5.
    parallel : bool, optional
        Enable parallel execution. Default: True.

    **Returns**

    LowessResult
        Result object with smoothed values and optional outputs.

    **Examples**
    >>> import numpy as np
    >>> import fastlowess
    >>> x = np.linspace(0, 10, 100)
    >>> y = 2 * x + np.random.normal(0, 1, 100)
    >>> result = fastlowess.smooth(x, y, fraction=0.3, confidence_intervals=0.95, parallel=True)
    >>> print(f"Smoothed {len(result.y)} points")

smooth_streaming(x, y, fraction=0.3, chunk_size=5000, overlap=None,
    iterations=3, delta=None, weight_function="tricube",
    robustness_method="bisquare", scaling_method="mad", boundary_policy="extend",
    auto_converge=None, return_diagnostics=False,
    return_residuals=False, return_robustness_weights=False,
    zero_weight_fallback="use_local_mean", parallel=True)
    Streaming LOWESS for large datasets.

    Processes data in chunks to maintain constant memory usage.
    Ideal for datasets that don't fit in memory or for batch pipelines.

    **Parameters**

    x : array_like
        Independent variable values.
    y : array_like
        Dependent variable values.
    fraction : float, optional
        Smoothing fraction. Default: 0.3.
    chunk_size : int, optional
        Size of each processing chunk. Default: 5000.
    overlap : int, optional
        Overlap between chunks for smooth transitions.
        Default: 10% of chunk_size.
    iterations : int, optional
        Number of robustness iterations. Default: 3.
    delta : float, optional
        Optimization threshold for point skipping. Default: None.
    weight_function : str, optional
        Kernel function: "tricube", "epanechnikov", "gaussian",
        "uniform", "biweight", "triangle". Default: "tricube".
    robustness_method : str, optional
        Robustness method: "bisquare", "huber", "talwar". Default: "bisquare".
    scaling_method : str, optional
        Scale estimation: "mad", "mar". Default: "mad".
    boundary_policy : str, optional
        Boundary handling: "extend", "reflect", "zero", "noboundary". Default: "extend".
    auto_converge : float, optional
        Adaptive convergence tolerance. Default: None.
    return_diagnostics : bool, optional
        Compute cumulative diagnostics across chunks. Default: False.
    return_residuals : bool, optional
        Whether to include residuals in output. Default: False.
    return_robustness_weights : bool, optional
        Include final robustness weights. Default: False.
    zero_weight_fallback : str, optional
        Fallback when all weights are zero: "use_local_mean",
        "return_original", "return_none". Default: "use_local_mean".
    parallel : bool, optional
        Enable parallel execution. Default: True.

    **Returns**

    LowessResult
        Result object with smoothed values.

    **Examples**
    >>> # Process 1 million points efficiently
    >>> x = np.arange(1_000_000, dtype=float)
    >>> y = np.sin(x * 0.001) + np.random.normal(0, 0.1, 1_000_000)
    >>> result = fastlowess.smooth_streaming(x, y, chunk_size=10000)


smooth_online(x, y, fraction=0.2, window_capacity=100, min_points=2,
    iterations=3, delta=None, weight_function="tricube",
    robustness_method="bisquare", scaling_method="mad", boundary_policy="extend",
    update_mode="full", auto_converge=None,
    return_robustness_weights=False,
    zero_weight_fallback="use_local_mean", parallel=False)
    Online LOWESS with sliding window.

    Maintains a sliding window for incremental updates. Ideal for
    real-time data streams or sensor data processing.

    **Parameters**

    x : array_like
        Independent variable values.
    y : array_like
        Dependent variable values.
    fraction : float, optional
        Smoothing fraction within the window. Default: 0.2.
    window_capacity : int, optional
        Maximum points to retain in the sliding window. Default: 100.
    min_points : int, optional
        Minimum points before smoothing starts. Default: 2.
    iterations : int, optional
        Number of robustness iterations. Default: 3.
    delta : float, optional
        Optimization threshold for point skipping. Default: None.
    weight_function : str, optional
        Kernel function. Default: "tricube".
    robustness_method : str, optional
        Robustness method. Default: "bisquare".
    scaling_method : str, optional
        Scale estimation. Default: "mad".
    boundary_policy : str, optional
        Boundary handling: "extend", "reflect", "zero", "noboundary". Default: "extend".
    update_mode : str, optional
        Update strategy: "full" or "incremental". Default: "full".
    auto_converge : float, optional
        Adaptive convergence tolerance. Default: None.
    return_robustness_weights : bool, optional
        Include final robustness weights. Default: False.
    zero_weight_fallback : str, optional
        Fallback when all weights are zero: "use_local_mean",
        "return_original", "return_none". Default: "use_local_mean".
    parallel : bool, optional
        Enable parallel execution. Default: False.

    **Returns**

    LowessResult
        Result object with smoothed values.

    **Examples**
    >>> # Simmons real-time sensor data
    >>> x = np.arange(1000, dtype=float)
    >>> y = 20.0 + 5.0 * np.sin(x * 0.1) + np.random.normal(0, 1, 1000)
    >>> result = fastlowess.smooth_online(x, y, window_capacity=50)


Classes
-------

LowessResult
    Result object containing smoothed values and optional outputs.

    **Attributes**

    x : numpy.ndarray
        Sorted x values.
    y : numpy.ndarray
        Smoothed y values (same length as x).
    fraction_used : float
        The smoothing fraction that was used.
    iterations_used : int or None
        Number of robustness iterations performed.
    standard_errors : numpy.ndarray or None
        Standard errors at each point (if intervals were requested).
    confidence_lower : numpy.ndarray or None
        Lower bound of confidence intervals (if confidence_intervals was set).
    confidence_upper : numpy.ndarray or None
        Upper bound of confidence intervals (if confidence_intervals was set).
    prediction_lower : numpy.ndarray or None
        Lower bound of prediction intervals (if prediction_intervals was set).
    prediction_upper : numpy.ndarray or None
        Upper bound of prediction intervals (if prediction_intervals was set).
    residuals : numpy.ndarray or None
        Residuals (y_original - y_smoothed) if return_residuals=True.
    robustness_weights : numpy.ndarray or None
        Final robustness weights if return_robustness_weights=True.
        Values near 0 indicate likely outliers; values near 1 are well-fitted.
    diagnostics : Diagnostics or None
        Diagnostic statistics if return_diagnostics=True.
    cv_scores : numpy.ndarray or None
        Cross-validation scores if cross-validation was performed.

    **Examples**
    >>> result = fastlowess.smooth(x, y, confidence_intervals=0.95)
    >>> print(f"Smoothed values: {result.y}")
    >>> print(f"Confidence interval: [{result.confidence_lower}, {result.confidence_upper}]")
    >>> print(f"Fraction used: {result.fraction_used}")


Diagnostics
    Diagnostic statistics for assessing fit quality.

    **Attributes**

    rmse : float
        Root Mean Squared Error - average prediction error magnitude.
    mae : float
        Mean Absolute Error - average absolute deviation.
    r_squared : float
        Coefficient of determination (R²) - proportion of variance explained.
        Range: [0, 1], with 1 indicating perfect fit.
    residual_sd : float
        Residual standard deviation.
    aic : float or None
        Akaike Information Criterion (when applicable).
    aicc : float or None
        Corrected AIC for small samples (when applicable).
    effective_df : float or None
        Effective degrees of freedom.

    **Examples**
    >>> result = fastlowess.smooth(x, y, return_diagnostics=True)
    >>> diag = result.diagnostics
    >>> print(f"RMSE: {diag.rmse:.4f}")
    >>> print(f"MAE: {diag.mae:.4f}")
    >>> print(f"R²: {diag.r_squared:.4f}")


Choosing the Right Function
---------------------------

+-------------------+----------------------------------------------------------+
| Function          | Use Case                                                 |
+===================+==========================================================+
| smooth()          | Primary interface - batch processing with all options    |
|                   | Includes cross-validation via cv_fractions parameter     |
+-------------------+----------------------------------------------------------+
| smooth_streaming()| Large datasets (>100K points), memory-constrained envs   |
+-------------------+----------------------------------------------------------+
| smooth_online()   | Real-time data streams, sliding window, sensor data      |
+-------------------+----------------------------------------------------------+


Parameter Guidelines
--------------------

**Fraction (smoothing span):**

- Larger fractions → smoother curves (more averaging)
- Smaller fractions → more detail (less averaging)
- Typical range: 0.2 to 0.7

**Iterations (robustness):**

- 0 iterations: No outlier handling (fastest)
- 1-3 iterations: Recommended for most data
- 4+ iterations: For heavily contaminated data

**Performance Tips:**

1. Use parallel=True for large datasets (>1,000 points)
2. Use delta optimization for very large datasets (>10,000 points)
3. Use streaming adapter for memory-constrained environments
4. Reduce iterations if speed is critical and data is clean


Version Information
-------------------
This package is built on top of the fastLowess Rust crate, providing
parallel LOWESS smoothing with NumPy integration.
"""

# Import from native extension
from .fastlowess import (
    smooth,
    smooth_streaming,
    smooth_online,
    LowessResult,
    Diagnostics,
)

__all__ = [
    "smooth",
    "smooth_streaming",
    "smooth_online",
    "LowessResult",
    "Diagnostics",
    "__version__",
]

from .__version__ import __version__
