---
title: Out-of-Sample Prediction
---
Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

> Out-of-sample prediction is available in **Batch** mode only. Streaming and Online modes do not support it.

`result.predict(newX, options)` evaluates the fit at arbitrary query points, like R's `predict(model, newdata)`.

It reuses `fit()`'s own (possibly `delta`-interpolated) smoothed curve for its `y` output — so predicting at a training `x` always exactly reproduces that point's `fit()` output, regardless of `delta`. A fresh local fit is only run when `return_derivative`, `return_se` (or an interval level), or `max_neighbor_distance` needs the actual regression slope or standard error.

Requires `retain_model: true` on the constructor before `fit()`, otherwise `predict()` throws.

---

## Options

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `return_se` | `boolean` | `false` | Include standard errors in the output |
| `confidence_level` | `number` | disabled | Confidence interval coverage level (e.g. `0.95`) |
| `prediction_level` | `number` | disabled | Prediction interval coverage level (e.g. `0.95`) |
| `return_derivative` | `boolean` | `false` | Include the local fit's derivative (slope) at each query point |
| `extrapolation` | `string` | `"clamp"` | Behavior for query points outside the training `x`-range |
| `max_extrapolation_distance` | `number` | disabled | Under `"linear"` extrapolation, the max allowed distance beyond the training boundary before erroring |
| `max_neighbor_distance` | `number` | disabled | Max allowed distance to the farthest training point in a query's local window before erroring |

### return_se

Computes standard errors for each query point, using the retained model's residual scale and per-point leverage. Required for `confidence_level`/`prediction_level` to be populated. `false` by default.

### confidence_level

Confidence level for the confidence interval around the mean response at each query point (e.g. `0.95`). Uses the same z-score convention as `fit()`'s own confidence intervals. Disabled by default.

### prediction_level

Confidence level for the prediction interval for a new observation at each query point (e.g. `0.95`). Widens using the same MAD-based residual scale `fit()` uses for its own intervals. Disabled by default.

### return_derivative

Includes the local fit's derivative (slope) at each query point in the output. `false` by default.

### extrapolation

Behavior for query points outside `[min(x_train), max(x_train)]`:

| Policy | Behavior |
| --- | --- |
| `"clamp"` (default) | Clamps the query to the nearest boundary window |
| `"linear"` | Linearly extrapolates from the nearest boundary point's local fit and slope |
| `"error"` | Fails the whole call with an error |

### max_extrapolation_distance

Under `"linear"` extrapolation, the maximum allowed distance beyond the training boundary before `predict()` throws, instead of returning an unbounded value. Disabled by default (uncapped).

### max_neighbor_distance

Maximum allowed distance to the farthest training point in a query's local window before `predict()` throws. Guards against an "empty range" blind spot: a query point can fall within `[min(x_train), max(x_train)]` yet still be far from any real training point (e.g. training `x` in `[0,10]` and `[90,100]`, query at `x=50`). Disabled by default (uncapped); applies regardless of `extrapolation`.

## Example

### Basic Usage

```javascript
const { Lowess } = require('fastlowess-wasm');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

const model = new Lowess({ fraction: 0.7, retain_model: true });
const result = model.fit(x, y);

const prediction = result.predict(new Float64Array([1.5, 4.5]));
console.log("Predicted y:", prediction.y);
```

```output
Predicted y: Float64Array(2) [ 3.05, 9.05 ]
```

### Standard Errors and Derivative

```javascript
const { Lowess } = require('fastlowess-wasm');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

const model = new Lowess({ fraction: 0.7, retain_model: true });
const result = model.fit(x, y);

const prediction = result.predict(new Float64Array([2.5]), {
    return_se: true,
    return_derivative: true,
});
console.log(prediction.y, prediction.standard_errors, prediction.derivative);
```

```output
Float64Array(1) [ 5.1 ] Float64Array(1) [ 0 ] Float64Array(1) [ 2.2 ]
```

### Linear Extrapolation

```javascript
const { Lowess } = require('fastlowess-wasm');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

const model = new Lowess({ fraction: 0.7, retain_model: true });
const result = model.fit(x, y);

const prediction = result.predict(new Float64Array([10.0]), {
    extrapolation: "linear",
});
console.log("Extrapolated y:", prediction.y);
```

```output
Extrapolated y: Float64Array(1) [ 10.1 ]
```
