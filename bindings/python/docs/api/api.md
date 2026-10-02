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

#### `fit(x, y)`

Fits the model to the provided `x` and `y` array-like objects. `custom_weights` is an optional array of per-observation weights — all values must be ≥ 0 and length must match `x`. Returns a `LowessResult` object containing the smoothed values and optional diagnostics.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1

model = fl.Lowess(fraction=0.5)
result = model.fit(x, y)
print(result)
:::

## Options Structures

### `LowessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `int` | `3` | Number of robustifying iterations |
| `weight_function` | `str` | `"tricube"` | Weight function name |
| `robustness_method` | `str` | `"bisquare"` | Robustness method name |
| `delta` | `float` | `None` | Interpolation distance (`None` auto-sets it to 1% of the x-range) |
| `zero_weight_fallback` | `str` | `"use_local_mean"` | Zero-weight handling strategy |
| `boundary_policy` | `str` | `"extend"` | Boundary handling policy |
| `scaling_method` | `str` | `"mad"` | Residual scaling method |
| `auto_converge` | `float` | `None` | Auto-convergence tolerance |
| `missing` | `str` | `"error"` | Policy for non-finite (NaN/Inf) values in input data |
| `parallel` | `bool` | `True` | Enable parallel execution |
| `backend` | `str` | `"cpu"` | Execution backend (`"cpu"` or `"gpu"`); GPU requires the package to be built with the `gpu` Cargo feature |
| `outputs` | `Sequence[str]` | `[]` | Select `se`, `diagnostics`, `residuals`, `weights`, `derivative`, and/or `sorted` |
| `intervals` | `dict` | `None` | Grouped interval options: `confidence`, `prediction`, and `bootstrap` |
| `cv` | `dict` | `None` | Grouped CV options: `method`, `k`, and `fractions` |
| `seed` | `int` | `None` | Shared CV/bootstrap seed; `0` is a valid seed |
| `retain_model` | `bool` | `False` | Retain training data, enabling `predict()` on the result |
| `custom_weights` | `list[float]` | `None` | Per-observation case weights — passed to `fit()`, not the constructor |
| `return_derivative` | `bool` | `False` | Include the per-point local fit derivative (slope) in result |

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

### delta

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `None` (default) auto-sets it to 1% of the x-range. Set it to `0.0` explicitly to disable interpolation and fit every point exactly.

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### boundary_policy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### auto_converge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `None` (default) disables early stopping.

### missing

Policy for handling non-finite (NaN/Inf) values in `x`/`y` (and, `custom_weights`):

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Raise an error if any value is non-finite |
| `"drop"` | Silently remove observations where `x` or `y` is non-finite before fitting |

**Note:** A length mismatch between `x` and `y` always errors, even under `"drop"`.

### parallel

Enable multi-threaded execution via Rayon.

- `True` (default) — parallelizes the local regression fits across CPU cores
- `False` — forces single-threaded execution (useful for benchmarking or deterministic profiling)

### backend

*See: [GPU Backend](../advanced/gpu-backend.md)*

The batch `Lowess` class can optionally run on a GPU-accelerated backend powered by `wgpu`, for high-throughput processing of large datasets (10k+ points).

- `"cpu"` (default)
- `"gpu"` — requires the package to be built with the `gpu` Cargo feature

### outputs

*See: [`Diagnostics`](#diagnostics)*

Use `outputs=["diagnostics", "residuals", "weights", "derivative", "se", "sorted"]` to select optional result components. AIC/AICc/`effective_df` additionally require `"se"` (or confidence/prediction intervals) to be populated.

### outputs: se

*See: [Intervals](../guide/intervals.md#standard-errors)*

Computes hat-matrix statistics (effective degrees of freedom, leverage, delta1/delta2) in addition to standard errors.

The `"sorted"` output reorders every result field by ascending `x` instead of preserving input order.

### intervals

*See: [Intervals](../guide/intervals.md)*

A dict such as `{"confidence": 0.90, "prediction": 0.99, "bootstrap": 200}`. Confidence and prediction coverage levels are independent: `confidence` bounds the mean response and `prediction` bounds a new observation; omitted levels disable that bound. `bootstrap` (at least `2`) replaces analytic intervals with residual-bootstrap standard errors and percentile bounds. `seed` controls fit-time bootstrap draws but does not enable bootstrap by itself. Unknown keys raise `ValueError`.

### CV Options

*See: [Cross-Validation](../guide/cross-validation.md)*

A dict such as `{"method": "kfold", "k": 5, "fractions": [0.2, 0.3, 0.5]}`:

- `method`: `"kfold"` (default) — fast, evaluates each candidate fraction over `k` folds; `"loocv"` — slow, exhaustive leave-one-out cross-validation
- `k`: Number of folds for k-fold CV. Ignored when `method="loocv"`.
- `fractions`: Candidate fractions to evaluate. Required.

Seed k-fold shuffling with the outer `seed` argument, not inside `cv`. Unknown keys raise `ValueError`.

### seed

One seed shared by k-fold CV shuffling and fit-time residual bootstrap. It does not enable either feature by itself; `None` (default) uses each feature's default, and `0` is a valid seed.

### retain_model

*See: [Predict](../guide/predict.md)*

Retains the fitted model's training data, enabling `LowessResult.predict(new_x, ...)` to evaluate the fit at out-of-sample query points not in the training set. `False` (default) — no extra memory/copy cost unless requested.

### custom_weights

*See: [Custom Weights](../weighting/custom-weights.md)*

Per-observation weights, passed to `fit()` rather than the constructor.

### return_derivative

Each point's local WLS fit already computes a slope internally; this exposes that per-point slope (rate of change of the smoothed curve) in `LowessResult.derivative`, enabling turning-point/rate-of-change analysis at effectively no extra computation cost.

- `False` (default) — leaves `result.derivative` as `None`
- `True` — populates it

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
| `residuals` | `ndarray \| None` | Residuals (if `"residuals"` was requested) |
| `robustness_weights` | `ndarray \| None` | Robustness weights (if `"weights"` was requested) |
| `cv_scores` | `ndarray \| None` | CV score per tested fraction |
| `diagnostics` | `Diagnostics \| None` | Fit metrics (if `"diagnostics"` was requested) |
| `derivative` | `ndarray \| None` | Per-point local fit derivative/slope (if `return_derivative`) |

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

## Predict

### `LowessResult.predict(new_x, ...) -> PredictOutput`

Evaluates the fitted model at out-of-sample query points. Requires `retain_model=True` on the constructor before `fit()`, otherwise raises `LowessError`.

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
