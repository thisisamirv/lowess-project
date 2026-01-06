Parameter Reference
===================

Detailed documentation for all parameters available in the API functions.

.. _smooth_params:

smooth() Parameters
-------------------

Parameters for ``fastlowess.smooth()`` (Batch Mode).

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Description
   * - **x**
     - Array-like. Independent variable values (1D array). Must be numeric.
   * - **y**
     - Array-like. Dependent variable values (1D array). Must have same length as x.
   * - **fraction**
     - Float (0.0 - 1.0]. The smoothing span/bandwidth.
       
       * **0.1-0.3**: Fine detail, captures rapid changes but may be noisy.
       * **0.3-0.5**: Moderate smoothing (recommended starting point).
       * **0.5-0.7**: Heavy smoothing, emphasizes broad global trends.
       * **0.7-1.0**: Very heavy smoothing.
   * - **iterations**
     - Integer (>= 0). Number of robustifying iterations (IRLS).
       
       * **0**: No robustness (equivalent to standard weighted regression). Fastest.
       * **1-3**: Moderate robustness (recommended). Good trade-off.
       * **4+**: Strong robustness. Use for heavily contaminated data.
   * - **weight_function**
     - String. Kernel function for neighborhood weighting.
       
       * **"tricube"** (default): $$(1 - |u|^3)^3$$. Best all-rounder.
       * **"epanechnikov"**: Optimal for minimizing MSE.
       * **"gaussian"**: Gaussian/Normal distribution. Infinite support.
       * **"uniform"**: Boxcar/Moving average.
       * **"biweight"**, **"triangle"**, **"cosine"**.
   * - **robustness_method**
     - String. Method for downweighting outliers.
       
       * **"bisquare"** (default): Smooth descent to zero.
       * **"huber"**: Linear penalty tails.
       * **"talwar"**: Hard cut-off.
   * - **scaling_method**
     - String. Method for robustness scale estimation.
       
       * **"mad"** (default): Median Absolute Deviation.
       * **"mar"**: Median Absolute Residual.
   * - **boundary_policy**
     - String. Handling of edge effects.
       
       * **"extend"** (default): Replicates first/last values. Preserves trends.
       * **"reflect"**: Mirrors data.
       * **"zero"**: Zero padding.
       * **"noboundary"**: No padding.
   * - **delta**
     - Float or None. Interpolation threshold. Points within ``delta`` of the last computed point are interpolated.
       
       Defaults to ``None`` (automatic for large N). Set manually (e.g., 0.01 * range) for performance tuning.
   * - **parallel**
     - Boolean. Enable multi-threaded execution via Rayon. Default: ``True``.
   * - **return_diagnostics**
     - Boolean. Calculate R², RMSE, etc. Default: ``False``.
   * - **return_residuals**
     - Boolean. Include residuals (y - smoothed) in result. Default: ``False``.
   * - **return_robustness_weights**
     - Boolean. Include final robust weights in result. Default: ``False``.
   * - **zero_weight_fallback**
     - String. Behavior when a point has 0 total weight.
       
       * **"use_local_mean"** (default)
       * **"return_original"**
       * **"return_none"**
   * - **auto_converge**
     - Float or None. Stop iterations if max change < tolerance. Default: ``None``.
   * - **confidence_intervals**
     - Float (0.0 - 1.0) or None. Confidence level (e.g., 0.95). Default: ``None``.
   * - **prediction_intervals**
     - Float (0.0 - 1.0) or None. Prediction level (e.g., 0.95). Default: ``None``.
   * - **cv_fractions**
     - List of floats or None. Candidates for Cross-Validation. Default: ``None``.
   * - **cv_method**
     - String. "loocv" (Leave-One-Out) or "kfold". Default: "kfold".
   * - **cv_k**
     - Integer. Number of folds for k-fold CV. Default: 5.

.. _streaming_params:

smooth_streaming() Parameters
-----------------------------

Parameters for ``fastlowess.smooth_streaming()`` (Streaming Mode).

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Description
   * - **x, y**
     - Input arrays (see above).
   * - **fraction**
     - Smoothing fraction (see above).
   * - **chunk_size**
     - Integer. Number of points to process per batch. Affects memory usage. Default: 5000.
   * - **overlap**
     - Integer or None. Points of overlap between chunks to ensure smoothness. Default: 10% of chunk_size.
   * - **parallel**
     - Boolean. Enable multi-threaded execution. Default: ``True``.
   * - **iterations, delta**
     - See **smooth()** above.
   * - **weight_function**
     - See **smooth()** above.
   * - **robustness_method**
     - See **smooth()** above.
   * - **scaling_method**
     - See **smooth()** above.
   * - **boundary_policy**
     - See **smooth()** above.
   * - **return_diagnostics**
     - Boolean. Cumulative diagnostics over chunks. Default: ``False``.
   * - **return_residuals**
     - Boolean. Include residuals. Default: ``False``.
   * - **return_robustness_weights**
     - Boolean. Include robust weights. Default: ``False``.
   * - **zero_weight_fallback**
     - See **smooth()** above.
   * - **auto_converge**
     - See **smooth()** above.

.. _online_params:

smooth_online() Parameters
--------------------------

Parameters for ``fastlowess.smooth_online()`` (Online/Real-time Mode).

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Description
   * - **x, y**
     - Input arrays.
   * - **fraction**
     - Smoothing fraction within the sliding window.
   * - **window_capacity**
     - Integer. Maximum number of points to keep in history.
   * - **min_points**
     - Integer. Minimum points required before producing output. Default: 3.
   * - **update_mode**
     - String.
       
       * **"incremental"** (default): Optimized updates (O(1) amortized).
       * **"full"**: Full re-fit per point (O(N)).
   * - **parallel**
     - Boolean. Default: ``False`` (usually faster for small windows sequentially).
   * - **iterations, delta**
     - See **smooth()** above.
   * - **weight_function**
     - See **smooth()** above.
   * - **robustness_method**
     - See **smooth()** above.
   * - **scaling_method**
     - See **smooth()** above.
   * - **boundary_policy**
     - See **smooth()** above.
   * - **return_robustness_weights**
     - Boolean. Include robust weights.
   * - **zero_weight_fallback**
     - See **smooth()** above.
