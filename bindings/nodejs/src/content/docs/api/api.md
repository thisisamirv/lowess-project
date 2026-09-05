---
title: API
---
The Node.js bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

> **StreamingLowess** and **OnlineLowess** are documented separately: [Streaming Adapter](api-streaming.md), [Online Adapter](api-online.md)

## When to Use Batch Adapter

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

## Classes

### `Lowess`

The `Lowess` class allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```javascript
const { Lowess } = require('fastlowess');

const model = new Lowess({ fraction: 0.5, iterations: 3 });
const result = model.fit(
    new Float64Array([0, 1, 2, 3, 4, 5]),
    new Float64Array([0.0, 1.1, 1.9, 3.1, 3.9, 5.0])
);
console.log("y[0]:", result.y[0].toFixed(4));
```

```output
y[0]: 0.0000
```

- `options`: An object containing `LowessOptions` fields.

**Methods:**

```javascript
const { Lowess } = require('fastlowess');

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

- Fits the model to the provided `x` and `y` typed arrays.
- `customWeights`: Optional `Float64Array` of per-observation weights. All values must be ≥ 0 and length must match `x`.
- Returns a `LowessResult` object containing the smoothed values and optional diagnostics.

## Options Structures

### `LowessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `number` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `number` | `3` | Number of robustifying iterations |
| `delta` | `number` | `NaN` | Interpolation distance (`NaN` auto-sets it to 1% of the x-range) |
| `weight_function` | `string` | `"tricube"` | Weight function name |
| `robustness_method` | `string` | `"bisquare"` | Robustness method name |
| `scaling_method` | `string` | `"mad"` | Residual scaling method |
| `boundary_policy` | `string` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `string` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge` | `number` | `null` | Auto-convergence tolerance |
| `confidence_intervals` | `number` | `null` | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `number` | `null` | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `boolean` | `false` | Include diagnostics in result |
| `return_residuals` | `boolean` | `false` | Include residuals in result |
| `return_robustness_weights` | `boolean` | `false` | Include weights in result |
| `return_se` | `boolean` | `false` | Return standard errors |
| `return_sorted` | `boolean` | `false` | Return results sorted ascending by `x` instead of in original input order |
| `parallel` | `boolean` | `true` | Enable parallel execution |
| `backend` | `string` | `"cpu"` | Execution backend (`"cpu"` or `"gpu"`); GPU requires the package to be built with the `gpu` Cargo feature |
| `cv_method` | `string` | `"kfold"` | CV method (`"kfold"` fast or `"loocv"` slow, exhaustive) |
| `cv_k` | `number` | `5` | Number of folds for k-fold CV |
| `cv_fractions` | `number[]` | `null` | Fractions to test for cross-validation |
| `cv_seed` | `number` | `null` | Random seed for cross-validation shuffling |
| `custom_weights` | `Float64Array` | `null` | Per-observation case weights — passed to `fit()`, not the options object |

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

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `NaN` (default) auto-sets it to 1% of the x-range. Set it to `0` explicitly to disable interpolation and fit every point exactly.

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

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### auto_converge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `null` (default) disables early stopping.

### confidence_intervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the confidence interval around the mean response (e.g. `0.95`). `null` (default) disables confidence intervals.

### prediction_intervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the prediction interval for new observations (e.g. `0.95`). `null` (default) disables prediction intervals.

### return_diagnostics

*See: [`Diagnostics`](#diagnostics)*

Include a `Diagnostics` object (RMSE, MAE, R², AIC/AICc, effective degrees of freedom) in the result. AIC/AICc/`effective_df` additionally require `return_se: true` (or confidence/prediction intervals) to be populated, since they depend on hat-matrix statistics.

- `false` (default) — leaves `result.diagnostics` as `null`
- `true` — populates `result.diagnostics`

### return_residuals

Include per-point residuals (`y - fitted`) in the result.

- `false` (default) — leaves `result.residuals` as `null`
- `true` — populates `result.residuals`

### return_robustness_weights

Include the final per-point robustness weights (from the last robustness iteration) in the result.

- `false` (default) — leaves `result.robustness_weights` as `null`
- `true` — populates `result.robustness_weights`

### return_se

*See: [Intervals](../guide/intervals.md#standard-errors)*

Computes hat-matrix statistics (effective degrees of freedom, leverage, delta1/delta2) in addition to standard errors.

### return_sorted

When set to `true`, it reorders every result field (residuals, intervals, etc.) by `x` in an ascending manner, instead of in original input order.
To get both orderings, sort the default result client-side (e.g. by the returned `x` array's sort order) instead of calling `fit()` twice.

### parallel

Enable multi-threaded execution via Rayon.

- `true` (default) — parallelizes the local regression fits across CPU cores
- `false` — forces single-threaded execution (useful for benchmarking or deterministic profiling)

### backend

*See: [GPU Backend](../advanced/gpu-backend.md)*

The batch `Lowess` class can optionally run on a GPU-accelerated backend powered by `wgpu`, for high-throughput processing of large datasets (10k+ points). GPU support applies to `Lowess` (batch) only — `StreamingLowess`/`OnlineLowess` remain CPU-only.

- `"cpu"` (default)
- `"gpu"` — requires the package to be built with the `gpu` Cargo feature

### CV Options

*See: [Cross-Validation](../guide/cross-validation.md)*

- `cv_method`: `"kfold"` (default) — fast, evaluates each candidate fraction over `cv_k` folds; `"loocv"` — slow, exhaustive leave-one-out cross-validation
- `cv_k`: Number of folds for k-fold CV. Ignored when `cv_method="loocv"`.
- `cv_fractions`: Candidate fractions to evaluate. Cross-validation is disabled unless this is set.
- `cv_seed`: Seed for reproducible k-fold shuffling. `null` (default) uses a random seed.

### custom_weights

*See: [Custom Weights](../weighting/custom-weights.md)*

Per-observation weights, passed to `fit()` rather than the options object.

## Result Structure

### `LowessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Float64Array` | x values (same order as input) |
| `y` | `Float64Array` | Smoothed y values |
| `fraction_used` | `number` | Fraction used (set or selected by CV) |
| `iterations_used` | `number \| null` | Robustness iterations actually performed |
| `standard_errors` | `Float64Array \| null` | Per-point standard errors |
| `confidence_lower` | `Float64Array \| null` | Lower confidence bounds |
| `confidence_upper` | `Float64Array \| null` | Upper confidence bounds |
| `prediction_lower` | `Float64Array \| null` | Lower prediction bounds |
| `prediction_upper` | `Float64Array \| null` | Upper prediction bounds |
| `residuals` | `Float64Array \| null` | Residuals (if `return_residuals`) |
| `robustness_weights` | `Float64Array \| null` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `Float64Array \| null` | CV score per tested fraction |
| `diagnostics` | `Diagnostics \| null` | Fit metrics (if `return_diagnostics`) |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `number` | Root Mean Squared Error |
| `mae` | `number` | Mean Absolute Error |
| `r_squared` | `number` | R-squared |
| `residual_sd` | `number` | Residual standard deviation |
| `effective_df` | `number` \| `null` | Effective degrees of freedom |
| `aic` | `number` \| `null` | AIC |
| `aicc` | `number` \| `null` | AICc |

## Example

```javascript
const { Lowess } = require('fastlowess');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

// Configure model
const model = new Lowess({ fraction: 0.5 });

// Fit data
const result = model.fit(x, y);

console.log("Smoothed Y:", result.y);
```

```output
Smoothed Y: Float64Array(5) [ 2.1, 4, 6.2, 8, 10.1 ]
```
