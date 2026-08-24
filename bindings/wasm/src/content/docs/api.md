---
title: fastLowess WebAssembly API Reference
---
The WebAssembly bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

> **StreamingLowess** and **OnlineLowess** are documented separately: [wasm-streaming.md](wasm-streaming.md), [wasm-online.md](wasm-online.md)

## Classes and Functions

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

* `options`: An object containing `LowessOptions` fields.

**Methods:**

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

* `x`: `Float64Array` of input x values.
* `y`: `Float64Array` of input y values.
* Returns: A `LowessResult` object.

See [wasm-streaming.md](wasm-streaming.md) for the `StreamingLowess` class.

See [wasm-online.md](wasm-online.md) for the `OnlineLowess` class.

## Options Structures

### `LowessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `number` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `number` | `3` | Number of robustifying iterations |
| `delta` | `number` | `NaN` | Interpolation distance (NaN for auto) |
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
| `parallel` | `boolean` | `true` | Enable parallel execution |
| `cv_method` | `string` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) (Batch only) |
| `cv_k` | `number` | `5` | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `number[]` | `null` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `number` | `null` | Random seed for cross-validation shuffling (Batch only) |
| `custom_weights` | `Float64Array` | `null` | Per-observation case weights — passed to `fit()`, not the options object (Batch only) |

See [wasm-streaming.md](wasm-streaming.md) for `StreamingOptions`.

See [wasm-online.md](wasm-online.md) for `OnlineOptions`.

## Result Structure

See [wasm-online.md](wasm-online.md) for `OnlineOutput`.

### `LowessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Float64Array` | Sorted x values |
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

## Options

### weight_function

*See: [Weight Functions](../user-guide/kernels.md)*

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"` (alias: `"boxcar"`)
* `"biweight"` (alias: `"bisquare"`)
* `"triangle"` (alias: `"triangular"`)
* `"cosine"`

### robustness_method

*See: [Robustness](../user-guide/robustness.md)*

* `"bisquare"` (default; alias: `"biweight"`)
* `"huber"`
* `"talwar"`

### boundary_policy

*See: [Boundary Handling](../user-guide/boundary.md)*

* `"extend"` (default; alias: `"pad"`)
* `"reflect"` (alias: `"mirror"`)
* `"zero"`
* `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](../user-guide/scaling.md)*

* `"mad"` (default; alias: `"median_absolute_deviation"`)
* `"mar"` (alias: `"median_absolute_residual"`)
* `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

*See: [Parameters](../user-guide/parameters.md)*

* `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`)
* `"return_original"` (alias: `"original"`)
* `"return_none"` (alias: `"none"`)

### merge_strategy

See [wasm-streaming.md](wasm-streaming.md).

### update_mode

See [wasm-online.md](wasm-online.md).

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
