# Out-of-Sample Prediction

Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

Out-of-sample prediction is available in **Batch** mode only. Streaming and Online modes do not support it.

`predict(model::PredictModel, new_x; kwargs...)` evaluates the fit at arbitrary query points, like R's `predict(model, newdata)`.

It reuses `fit`'s own (possibly `delta`-interpolated) smoothed curve for its `y` output — so predicting at a training `x` always exactly reproduces that point's `fit` output, regardless of `delta`. A fresh local fit is only run when `return_derivative`, `return_se` (or an interval level), or `max_neighbor_distance` needs the actual regression slope or standard error.

`model` comes from `result.predict_model`, populated only when `retain_model=true` was passed to `Lowess`.

---

## Options

| Keyword Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `return_se` | `Bool` | `false` | Include standard errors in the output |
| `confidence_level` | `Union{Float64, Nothing}` | `nothing` | Confidence interval coverage level (e.g. `0.95`) |
| `prediction_level` | `Union{Float64, Nothing}` | `nothing` | Prediction interval coverage level (e.g. `0.95`) |
| `return_derivative` | `Bool` | `false` | Include the local fit's derivative (slope) at each query point |
| `extrapolation` | `String` | `"clamp"` | Behavior for query points outside the training `x`-range |
| `max_extrapolation_distance` | `Union{Float64, Nothing}` | `nothing` | Under `"linear"` extrapolation, the max allowed distance beyond the training boundary before erroring |
| `max_neighbor_distance` | `Union{Float64, Nothing}` | `nothing` | Max allowed distance to the farthest training point in a query's local window before erroring |

### return_se

Computes standard errors for each query point, using the retained model's residual scale and per-point leverage. Required for `confidence_level`/`prediction_level` to be populated. `false` by default.

### confidence_level

Confidence level for the confidence interval around the mean response at each query point (e.g. `0.95`). Uses the same z-score convention as `fit`'s own confidence intervals. `nothing` (default) disables it.

### prediction_level

Confidence level for the prediction interval for a new observation at each query point (e.g. `0.95`). Widens using the same MAD-based residual scale `fit` uses for its own intervals. `nothing` (default) disables it.

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

Under `"linear"` extrapolation, the maximum allowed distance beyond the training boundary before `predict` errors, instead of returning an unbounded value. `nothing` (default, uncapped).

### max_neighbor_distance

Maximum allowed distance to the farthest training point in a query's local window before `predict` errors. Guards against an "empty range" blind spot: a query point can fall within `[min(x_train), max(x_train)]` yet still be far from any real training point (e.g. training `x` in `[0,10]` and `[90,100]`, query at `x=50`). `nothing` (default, uncapped); applies regardless of `extrapolation`.

## Example

### Basic Usage

```@example predict
using FastLOWESS

x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.1, 4.0, 6.2, 8.0, 10.1]

model = Lowess(; fraction=0.7, retain_model=true)
result = fit(model, x, y)

prediction = predict(result.predict_model, [1.5, 4.5])
println("Predicted y: ", prediction.y)
```

### Standard Errors and Derivative

```@example predict
prediction = predict(result.predict_model, [2.5]; return_se=true, return_derivative=true)
println("y: ", prediction.y)
println("SE: ", prediction.standard_errors)
println("Derivative: ", prediction.derivative)
```

### Linear Extrapolation

```@example predict
prediction = predict(result.predict_model, [10.0]; extrapolation="linear")
println("Extrapolated y: ", prediction.y)
```
