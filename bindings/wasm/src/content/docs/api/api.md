---
title: API
---
The WebAssembly bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

> **StreamingLowess** and **OnlineLowess** are documented separately: [Streaming Adapter](api-streaming.md), [Online Adapter](api-online.md)

## When to Use Batch Adapter

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

## Classes

### `Lowess`

The `Lowess` class is the main entry point for batch smoothing.

**Constructor:**

```javascript
const { Lowess } = require('fastlowess-wasm');

const model = new Lowess({ fraction: 0.5, iterations: 3 });
console.log("typeof fit:", typeof model.fit);
```

```output
typeof fit: function
```

- `options`: An object containing `LowessOptions` fields.

#### `fit(x, y)`

Fits the model to the provided `x` (`Float64Array` of input x values) and `y` (`Float64Array` of input y values). Returns: A `LowessResult` object.

```javascript
const { Lowess } = require('fastlowess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const model = new Lowess({ fraction: 0.5 });
const result = model.fit(x, y);
console.log("Fraction used:", result.fraction_used);
```

```output
Fraction used: 0.5
```

## Options Structures

### `LowessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `number` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `number` | `3` | Number of robustifying iterations |
| `weight_function` | `string` | `"tricube"` | Weight function name |
| `robustness_method` | `string` | `"bisquare"` | Robustness method name |
| `delta` | `number` | `NaN` | Interpolation distance (`NaN` auto-sets it to 1% of the x-range) |
| `zero_weight_fallback` | `string` | `"use_local_mean"` | Zero-weight handling |
| `boundary_policy` | `string` | `"extend"` | Boundary handling policy |
| `scaling_method` | `string` | `"mad"` | Residual scaling method |
| `auto_converge` | `number` | `null` | Auto-convergence tolerance |
| `missing` | `string` | `"error"` | Policy for non-finite (NaN/Inf) values in input data |
| `parallel` | `boolean` | `true` | Enable parallel execution |
| `outputs` | `string[]` | `[]` | Select `se`, `diagnostics`, `residuals`, `weights`, `derivative`, and/or `sorted` |
| `intervals` | `object` | `null` | Grouped interval options: `confidence`, `prediction`, and `bootstrap` |
| `cv` | `object` | `null` | Grouped CV options: `method`, `k`, and `fractions` |
| `seed` | `number` | `null` | Shared CV/bootstrap seed; `0` is a valid seed |
| `retain_model` | `boolean` | `false` | Retain training data, enabling `result.predict()` |
| `custom_weights` | `Float64Array` | `null` | Per-observation case weights — passed to `fit()`, not the options object |
| `return_derivative` | `boolean` | `false` | Include the per-point local fit derivative (slope) in result |

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

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `NaN` (default) auto-sets it to 1% of the x-range. Set it to `0` explicitly to disable interpolation and fit every point exactly.

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

Convergence tolerance for early stopping of robustness iterations. `null` (default) disables early stopping.

### missing

Policy for handling non-finite (NaN/Inf) values in `x`/`y` (and `custom_weights`):

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Throw an error if any value is non-finite |
| `"drop"` | Silently remove observations where `x` or `y` is non-finite before fitting |

**Note:** A length mismatch between `x` and `y` always errors, even under `"drop"`.

### parallel

Enable multi-threaded execution via the Rayon-based web worker pool.

- `true` (default) — parallelizes the local regression fits
- `false` — forces single-threaded execution

### intervals

*See: [Intervals](../guide/intervals.md)*

An object such as `{ confidence: 0.95, prediction: 0.95, bootstrap: 200 }`. `confidence` bounds the mean response and `prediction` bounds a new observation; omitted levels disable that bound. `bootstrap` (at least `2`) replaces analytic intervals with residual-bootstrap standard errors and percentile bounds. `seed` controls fit-time bootstrap draws but does not enable bootstrap by itself.

### CV Options

*See: [Cross-Validation](../guide/cross-validation.md)*

An object such as `{ method: "kfold", k: 5, fractions: [0.2, 0.3, 0.5] }`:

- `method`: `"kfold"` (default) — fast, evaluates each candidate fraction over `k` folds; `"loocv"` — slow, exhaustive leave-one-out cross-validation
- `k`: Number of folds for k-fold CV. Ignored when `method: "loocv"`.
- `fractions`: Candidate fractions to evaluate. Required.

Seed k-fold shuffling with the outer `seed` option, not inside `cv`.

### seed

One seed shared by k-fold CV shuffling and fit-time residual bootstrap. It does not enable either feature by itself; `null` (default) uses each feature's default, and `0` is a valid seed. Negative values throw.

### retain_model

*See: [Predict](../guide/predict.md)*

Retains the fitted model's training data, enabling `result.predict(newX, options)` to evaluate the fit at out-of-sample query points not in the training set. `false` (default) — no extra memory/copy cost unless requested.

### custom_weights

*See: [Custom Weights](../weighting/custom-weights.md)*

Per-observation weights, passed to `fit()` rather than the options object.

### return_se

*See: [Intervals](../guide/intervals.md#standard-errors)*

Computes hat-matrix statistics (effective degrees of freedom, leverage, delta1/delta2) in addition to standard errors.

### return_diagnostics

*See: [`Diagnostics`](#diagnostics)*

Include a `Diagnostics` object (RMSE, MAE, R², AIC/AICc, effective degrees of freedom) in the result. AIC/AICc/`effective_df` additionally require `return_se: true` (or confidence/prediction intervals) to be populated, since they depend on hat-matrix statistics.

- `false` (default) — leaves `result.diagnostics` as `undefined`
- `true` — populates `result.diagnostics`

### return_residuals

Include per-point residuals (`y - fitted`) in the result.

- `false` (default) — leaves `result.residuals` as `undefined`
- `true` — populates `result.residuals`

### return_robustness_weights

Include the final per-point robustness weights (from the last robustness iteration) in the result.

- `false` (default) — leaves `result.robustness_weights` as `undefined`
- `true` — populates `result.robustness_weights`

### return_derivative

Each point's local WLS fit already computes a slope internally; this exposes that per-point slope (rate of change of the smoothed curve) in `LowessResult.derivative`, enabling turning-point/rate-of-change analysis at effectively no extra computation cost.

- `false` (default) — leaves `result.derivative` as `undefined`
- `true` — populates it

### return_sorted

When set to `true`, it reorders every result field (residuals, intervals, etc.) by `x` in an ascending manner, instead of in original input order.
To get both orderings, sort the default result client-side (e.g. by the returned `x` array's sort order) instead of calling `fit()` twice.

## Result Structure

### `LowessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Float64Array` | x values (same order as input) |
| `y` | `Float64Array` | Smoothed y values |
| `fraction_used` | `number` | Fraction used (set or selected by CV) |
| `iterations_used` | `number` \| `undefined` | Robustness iterations actually performed |
| `standard_errors` | `Float64Array` \| `undefined` | Per-point standard errors |
| `confidence_lower` | `Float64Array` \| `undefined` | Lower confidence bounds |
| `confidence_upper` | `Float64Array` \| `undefined` | Upper confidence bounds |
| `prediction_lower` | `Float64Array` \| `undefined` | Lower prediction bounds |
| `prediction_upper` | `Float64Array` \| `undefined` | Upper prediction bounds |
| `residuals` | `Float64Array` \| `undefined` | Residuals (if `return_residuals`) |
| `robustness_weights` | `Float64Array` \| `undefined` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `Float64Array` \| `undefined` | CV score per tested fraction |
| `diagnostics` | `Diagnostics` \| `undefined` | Fit metrics (if `return_diagnostics`) |
| `derivative` | `Float64Array` \| `undefined` | Per-point local fit derivative/slope (if `return_derivative`) |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `number` | Root Mean Squared Error |
| `mae` | `number` | Mean Absolute Error |
| `r_squared` | `number` | R-squared |
| `residual_sd` | `number` | Residual standard deviation |
| `effective_df` | `number` \| `undefined` | Effective degrees of freedom |
| `aic` | `number` \| `undefined` | AIC |
| `aicc` | `number` \| `undefined` | AICc |

## Predict

### `result.predict(newX, options) -> PredictOutput`

Evaluates the fitted model at out-of-sample query points. Requires `retain_model: true` on the constructor before `fit()`, otherwise throws.

## Example

```javascript
const { Lowess } = require('fastlowess-wasm');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

// Fit data
const model = new Lowess({ fraction: 0.5 });
const result = model.fit(x, y);

console.log("Smoothed Y:", result.y);
```

```output
Smoothed Y: Float64Array(5) [ 2.1, 4, 6.2, 8, 10.1 ]
```
