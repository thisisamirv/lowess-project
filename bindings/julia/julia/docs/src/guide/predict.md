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
| `outputs` | `Vector{String}` | `String[]` | Select `"se"` and/or `"derivative"` |
| `intervals` | `NamedTuple` | `nothing` | Grouped `confidence`, `prediction`, and optional `bootstrap` options |
| `seed` | `Union{Integer, Nothing}` | `nothing` | Reproducible prediction-time bootstrap draws; `0` is valid |
| `extrapolation` | `String` | `"clamp"` | Behavior for query points outside the training `x`-range |
| `max_extrapolation_distance` | `Union{Float64, Nothing}` | `nothing` | Under `"linear"` extrapolation, the max allowed distance beyond the training boundary before erroring |
| `max_neighbor_distance` | `Union{Float64, Nothing}` | `nothing` | Max allowed distance to the farthest training point in a query's local window before erroring |

### outputs

Request `"se"` for standard errors (from the retained model's residual scale and per-point leverage) and `"derivative"` for the local slope at each query point. Analytic intervals compute their required standard errors even without an explicit `"se"` output.

### intervals

A `NamedTuple` such as `(confidence=0.95, prediction=0.95, bootstrap=200)`. `confidence` bounds the mean response and `prediction` bounds a new observation at each query point. Without bootstrap, these use normal-theory standard errors and the retained residual scale.

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
prediction = predict(result.predict_model, [2.5]; outputs=["se", "derivative"])
println("y: ", prediction.y)
println("SE: ", prediction.standard_errors)
println("Derivative: ", prediction.derivative)
```

### Bootstrap Intervals

```@example predict
prediction = predict(
    result.predict_model,
    [2.5, 4.5];
    intervals=(confidence=0.95, prediction=0.95, bootstrap=200),
    seed=7,
)
println("CI lower: ", prediction.confidence_lower)
```

### Linear Extrapolation

```@example predict
prediction = predict(result.predict_model, [10.0]; extrapolation="linear")
println("Extrapolated y: ", prediction.y)
```
