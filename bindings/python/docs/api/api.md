<!-- markdownlint-disable MD046 -->
# fastLowess

The Python bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

> **StreamingLowess** and **OnlineLowess** are documented separately: [Streaming Adapter](api-streaming.md), [Online Adapter](api-online.md)

## When to Use Batch Adapter

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

## Classes

### `Lowess`

The `Lowess` class allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

:::{jupyter-execute}
import fastlowess as fl

model = fl.Lowess(fraction=0.5, iterations=3)
print(model)
:::

**Methods:**

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1

model = fl.Lowess(fraction=0.5)
result = model.fit(x, y)
print(result)
:::

- Fits the model to the provided `x` and `y` array-like objects.
- `custom_weights`: Optional array of per-observation weights. All values must be ≥ 0 and length must match `x`.
- Returns a `LowessResult` object containing the smoothed values and optional diagnostics.

## Options Structures

### `LowessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `int` | `3` | Number of robustifying iterations |
| `delta` | `float` | `None` | Interpolation distance (`None` auto-sets it to 1% of the x-range) |
| `weight_function` | `str` | `"tricube"` | Weight function name |
| `robustness_method` | `str` | `"bisquare"` | Robustness method name |
| `scaling_method` | `str` | `"mad"` | Residual scaling method |
| `boundary_policy` | `str` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `str` | `"use_local_mean"` | Zero-weight handling strategy |
| `auto_converge` | `float` | `None` | Auto-convergence tolerance |
| `confidence_intervals` | `float` | `None` | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `float` | `None` | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `bool` | `False` | Include diagnostics in result |
| `return_residuals` | `bool` | `False` | Include residuals in result |
| `return_robustness_weights` | `bool` | `False` | Include weights in result |
| `return_se` | `bool` | `False` | Return standard errors |
| `return_sorted` | `bool` | `False` | Return results sorted ascending by `x` instead of in original input order |
| `parallel` | `bool` | `True` | Enable parallel execution |
| `backend` | `str` | `"cpu"` | Execution backend (`"cpu"` or `"gpu"`); GPU requires the package to be built with the `gpu` Cargo feature |
| `cv_method` | `str` | `"kfold"` | CV method (`"kfold"` fast or `"loocv"` slow, exhaustive) |
| `cv_k` | `int` | `5` | Number of folds for k-fold CV |
| `cv_fractions` | `list[float]` | `None` | Fractions to test for cross-validation |
| `cv_seed` | `int` | `None` | Random seed for cross-validation shuffling |
| `custom_weights` | `list[float]` | `None` | Per-observation case weights — passed to `fit()`, not the constructor |

## Options

### fraction

`fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

### iterations

`iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

### delta

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `None` (default) auto-sets it to 1% of the x-range. Set it to `0.0` explicitly to disable interpolation and fit every point exactly.

### weight_function

*See: [Weight Functions](../weighting/kernels.md)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](../weighting/robustness.md)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### scaling_method

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### boundary_policy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### auto_converge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `None` (default) disables early stopping.

### confidence_intervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the confidence interval around the mean response (e.g. `0.95`). `None` (default) disables confidence intervals.

### prediction_intervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the prediction interval for new observations (e.g. `0.95`). `None` (default) disables prediction intervals.

### return_diagnostics

*See: [`Diagnostics`](#diagnostics)*

Include a `Diagnostics` object (RMSE, MAE, R², AIC/AICc, effective degrees of freedom) in the result. AIC/AICc/`effective_df` additionally require `return_se=True` (or confidence/prediction intervals) to be populated, since they depend on hat-matrix statistics.

- `False` (default) — leaves `result.diagnostics` as `None`
- `True` — populates `result.diagnostics`

### return_residuals

Include per-point residuals (`y - fitted`) in the result.

- `False` (default) — leaves `result.residuals` as `None`
- `True` — populates `result.residuals`

### return_robustness_weights

Include the final per-point robustness weights (from the last robustness iteration) in the result.

- `False` (default) — leaves `result.robustness_weights` as `None`
- `True` — populates `result.robustness_weights`

### return_se

*See: [Intervals](../guide/intervals.md#standard-errors)*

Computes hat-matrix statistics (effective degrees of freedom, leverage, delta1/delta2) in addition to standard errors.

### return_sorted

When set to `True`, it reorders every result field (residuals, intervals, etc.) by `x` in an ascending manner, instead of in original input order.
To get both orderings, sort the default result client-side (e.g. `np.argsort(result.x)`) instead of calling `fit()` twice.

### parallel

Enable multi-threaded execution via Rayon.

- `True` (default) — parallelizes the local regression fits across CPU cores
- `False` — forces single-threaded execution (useful for benchmarking or deterministic profiling)

### backend

*See: [GPU Backend](../advanced/gpu-backend.md)*

The batch `Lowess` class can optionally run on a GPU-accelerated backend powered by `wgpu`, for high-throughput processing of large datasets (10k+ points).

- `"cpu"` (default)
- `"gpu"` — requires the package to be built with the `gpu` Cargo feature

### CV Options

*See: [Cross-Validation](../guide/cross-validation.md)*

- `cv_method`: `"kfold"` (default) — fast, evaluates each candidate fraction over `cv_k` folds; `"loocv"` — slow, exhaustive leave-one-out cross-validation
- `cv_k`: Number of folds for k-fold CV. Ignored when `cv_method="loocv"`.
- `cv_fractions`: Candidate fractions to evaluate. Cross-validation is disabled unless this is set.
- `cv_seed`: Seed for reproducible k-fold shuffling. `None` (default) uses a random seed.

### custom_weights

*See: [Custom Weights](../weighting/custom-weights.md)*

Per-observation weights, passed to `fit()` rather than the constructor.

## Result Structure

### `LowessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `ndarray` | x values (same order as input) |
| `y` | `ndarray` | Smoothed y values |
| `fraction_used` | `float` | Fraction used (set or selected by CV) |
| `iterations_used` | `int \| None` | Robustness iterations actually performed |
| `standard_errors` | `ndarray \| None` | Per-point standard errors |
| `confidence_lower` | `ndarray \| None` | Lower confidence bounds |
| `confidence_upper` | `ndarray \| None` | Upper confidence bounds |
| `prediction_lower` | `ndarray \| None` | Lower prediction bounds |
| `prediction_upper` | `ndarray \| None` | Upper prediction bounds |
| `residuals` | `ndarray \| None` | Residuals (if `return_residuals`) |
| `robustness_weights` | `ndarray \| None` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `ndarray \| None` | CV score per tested fraction |
| `diagnostics` | `Diagnostics \| None` | Fit metrics (if `return_diagnostics`) |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `float` | Root Mean Squared Error |
| `mae` | `float` | Mean Absolute Error |
| `r_squared` | `float` | R-squared |
| `residual_sd` | `float` | Residual standard deviation |
| `effective_df` | `float \| None` | Effective degrees of freedom (`None` if not computed) |
| `aic` | `float \| None` | AIC (`None` if not computed) |
| `aicc` | `float \| None` | AICc (`None` if not computed) |

## Example

:::{jupyter-execute}
from fastlowess import Lowess
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1

## Configure model

model = Lowess(fraction=0.5)

## Fit data

result = model.fit(x, y)

print(result)
:::
