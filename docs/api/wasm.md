# fastLowess WebAssembly API Reference

The WebAssembly bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

## Classes and Functions

### `Lowess`

The `Lowess` class is the main entry point for batch smoothing.

**Constructor:**

```javascript
const { Lowess } = require('fastlowess');

const model = new Lowess({ fraction: 0.5, iterations: 3 });
```

* `options`: An object containing `LowessOptions` fields.

**Methods:**

```javascript
const { Lowess } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const model = new Lowess({ fraction: 0.5 });
const result = model.fit(x, y);
```

* `x`: `Float64Array` of input x values.
* `y`: `Float64Array` of input y values.
* Returns: A `LowessResult` object.

### `StreamingLowess`

The `StreamingLowess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```javascript
const { StreamingLowess } = require('fastlowess');

const stream = new StreamingLowess({ fraction: 0.3 }, { chunk_size: 50, overlap: 10 });
```

* `options`: An object containing `LowessOptions` fields.
* `streamingOptions`: An object containing `StreamingOptions` fields.

**Methods:**

```javascript
const { StreamingLowess } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const stream = new StreamingLowess({ fraction: 0.3 }, { chunk_size: 50, overlap: 10 });
const partialResult = stream.process_chunk(x.slice(0, 50), y.slice(0, 50));
```

* Processes a chunk of data. Returns partial results.

```javascript
const { StreamingLowess } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const stream = new StreamingLowess({ fraction: 0.3 }, { chunk_size: 50, overlap: 10 });
stream.process_chunk(x, y);
const finalResult = stream.finalize();
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `OnlineLowess`

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

```javascript
const { OnlineLowess } = require('fastlowess');

const online = new OnlineLowess({ fraction: 0.3 }, { window_capacity: 50, min_points: 5 });
```

* `options`: An object containing `LowessOptions` fields.
* `onlineOptions`: An object containing `OnlineOptions` fields.

**Methods:**

```javascript
const { OnlineLowess } = require('fastlowess');

const online = new OnlineLowess({ fraction: 0.3 }, { window_capacity: 50, min_points: 5 });
const result = online.add_point(1.0, 2.0);  // returns OnlineOutput | null
```

* Adds new points to the model and returns the smoothed values (retrospective or prospective depending on mode).

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
| `custom_weights` | `Float64Array` | `null` | Per-observation case weights — passed to `fit()`/`process_chunk()`, not the options object (Batch only) |

### `StreamingOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `number` | `5000` | Data chunk size |
| `overlap` | `number` | `500` | Overlap between chunks |
| `merge_strategy` | `string` | `"weighted_average"` | Strategy for blending overlap regions |

### `OnlineOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `number` | `1000` | Max points in sliding window |
| `min_points` | `number` | `3` | Min points before smoothing starts |
| `update_mode` | `string` | `"full"` | Update mode (`"full"` or `"incremental"`) |
| `parallel` | `boolean` | `false` | Enable parallel execution (off by default; online LOWESS fits one point at a time) |

## Result Structure

### `OnlineOutput`

Returned by `add_point()` once the window has enough points (`undefined` until then).

| Field | Type | Description |
| --- | --- | --- |
| `smoothed` | `number` | Smoothed value for the latest point |
| `std_error` | `number \| undefined` | Standard error (if requested) |
| `residual` | `number \| undefined` | Residual y − smoothed (if requested) |
| `robustness_weight` | `number \| undefined` | Robustness weight (if requested) |
| `iterations_used` | `number \| undefined` | Robustness iterations performed |

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

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"` (alias: `"boxcar"`)
* `"biweight"` (alias: `"bisquare"`)
* `"triangle"` (alias: `"triangular"`)
* `"cosine"`

### robustness_method

* `"bisquare"` (default; alias: `"biweight"`)
* `"huber"`
* `"talwar"`

### boundary_policy

* `"extend"` (default; alias: `"pad"`)
* `"reflect"` (alias: `"mirror"`)
* `"zero"`
* `"noboundary"` (alias: `"none"`)

### scaling_method

* `"mad"` (default; alias: `"median_absolute_deviation"`)
* `"mar"` (alias: `"median_absolute_residual"`)
* `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

* `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`)
* `"return_original"` (alias: `"original"`)
* `"return_none"` (alias: `"none"`)

### merge_strategy

* `"weighted_average"` (default; alias: `"weighted"`)
* `"average"` (alias: `"mean"`)
* `"take_first"` (alias: `"first"`)
* `"take_last"` (alias: `"last"`)

### update_mode

* `"full"` (default; alias: `"resmooth"`)
* `"incremental"` (alias: `"single"`)

## Example

```javascript
const { Lowess } = require('./fastlowess_wasm.js');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

// Fit data
const model = new Lowess({ fraction: 0.5 });
const result = model.fit(x, y);

console.log("Smoothed Y:", result.y);
```
