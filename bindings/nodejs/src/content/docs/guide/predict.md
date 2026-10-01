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
| `outputs` | `string[]` | `[]` | Select `"se"` and/or `"derivative"` |
| `intervals` | `object` | `null` | Grouped `confidence`, `prediction`, and optional `bootstrap` options |
| `seed` | `number` | `null` | Reproducible prediction-time bootstrap draws; `0` is valid |
| `extrapolation` | `string` | `"clamp"` | Behavior for query points outside the training `x`-range |
| `max_extrapolation_distance` | `number` | disabled | Under `"linear"` extrapolation, the max allowed distance beyond the training boundary before erroring |
| `max_neighbor_distance` | `number` | disabled | Max allowed distance to the farthest training point in a query's local window before erroring |

### outputs

Request `"se"` for standard errors (from the retained model's residual scale and per-point leverage) and `"derivative"` for the local slope at each query point. Analytic intervals compute their required standard errors even without an explicit `"se"` output.

### intervals

An object such as `{ confidence: 0.95, prediction: 0.95, bootstrap: 200 }`. `confidence` bounds the mean response and `prediction` bounds a new observation at each query point. Without bootstrap, these use normal-theory standard errors and the retained residual scale.

Set `bootstrap` to at least `2` to resample the retained Batch residuals, refit the model, and calculate query-point percentile intervals and standard errors.

### seed

Seeds prediction-time bootstrap draws, independently of the fit/CV `seed` on `Lowess`. It does not enable bootstrap by itself.

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
const { Lowess } = require('fastlowess');

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
const { Lowess } = require('fastlowess');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

const model = new Lowess({ fraction: 0.7, retain_model: true });
const result = model.fit(x, y);

const prediction = result.predict(new Float64Array([2.5]), {
    outputs: ["se", "derivative"],
});
console.log(prediction.y, prediction.standard_errors, prediction.derivative);
```

```output
Float64Array(1) [ 5.1 ] Float64Array(1) [ 0 ] Float64Array(1) [ 2.2 ]
```

### Bootstrap Intervals

```javascript
const { Lowess } = require('fastlowess');

const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1, 11.8, 14.2, 16.1, 17.9, 20.2]);

const model = new Lowess({ fraction: 0.7, retain_model: true });
const result = model.fit(x, y);

const prediction = result.predict(new Float64Array([2.5, 4.5]), {
    intervals: { confidence: 0.95, prediction: 0.95, bootstrap: 200 },
    seed: 7,
});
console.log("CI present:", prediction.confidence_lower !== null);
```

```output
CI present: true
```

### Linear Extrapolation

```javascript
const { Lowess } = require('fastlowess');

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
