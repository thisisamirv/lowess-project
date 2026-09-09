---
title: "Out-of-Sample Prediction"
weight: 40
---

Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

> Out-of-sample prediction is available in **Batch** mode only. Streaming and Online modes do not support it.

`(*PredictModel) Predict(newX, opts)` evaluates the fit at arbitrary query points, like R's `predict(model, newdata)`.

It reuses `Fit`'s own (possibly `Delta`-interpolated) smoothed curve for its `Y` output — so predicting at a training `x` always exactly reproduces that point's `Fit` output, regardless of `Delta`. A fresh local fit is only run when `ReturnDerivative`, `ReturnSE` (or an interval level), or `MaxNeighborDistance` needs the actual regression slope or standard error.

`Result.PredictModel` is non-nil only when `Options.RetainModel` was set to `true`. Call `Close()` on it when done (or let its finalizer run).

---

## Options

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `ReturnSE` | `bool` | `false` | Include standard errors in the output |
| `ConfidenceLevel` | `*float64` | `nil` | Confidence interval coverage level (e.g. `0.95`) |
| `PredictionLevel` | `*float64` | `nil` | Prediction interval coverage level (e.g. `0.95`) |
| `ReturnDerivative` | `bool` | `false` | Include the local fit's derivative (slope) at each query point |
| `Extrapolation` | `string` | `"clamp"` | Behavior for query points outside the training `x`-range |
| `MaxExtrapolationDistance` | `*float64` | `nil` | Under `"linear"` extrapolation, the max allowed distance beyond the training boundary before erroring |
| `MaxNeighborDistance` | `*float64` | `nil` | Max allowed distance to the farthest training point in a query's local window before erroring |

### ReturnSE

Computes standard errors for each query point, using the retained model's residual scale and per-point leverage. Required for `ConfidenceLevel`/`PredictionLevel` to be populated. `false` by default.

### ConfidenceLevel

Confidence level for the confidence interval around the mean response at each query point (e.g. `0.95`). Uses the same z-score convention as `Fit`'s own confidence intervals. `nil` (default) disables it.

### PredictionLevel

Confidence level for the prediction interval for a new observation at each query point (e.g. `0.95`). Widens using the same MAD-based residual scale `Fit` uses for its own intervals. `nil` (default) disables it.

### ReturnDerivative

Includes the local fit's derivative (slope) at each query point in the output. `false` by default.

### Extrapolation

Behavior for query points outside `[min(x_train), max(x_train)]`:

| Policy | Behavior |
| --- | --- |
| `"clamp"` (default) | Clamps the query to the nearest boundary window |
| `"linear"` | Linearly extrapolates from the nearest boundary point's local fit and slope |
| `"error"` | Fails the whole call with an error |

### MaxExtrapolationDistance

Under `"linear"` extrapolation, the maximum allowed distance beyond the training boundary before `Predict` errors, instead of returning an unbounded value. `nil` (default, uncapped).

### MaxNeighborDistance

Maximum allowed distance to the farthest training point in a query's local window before `Predict` errors. Guards against an "empty range" blind spot: a query point can fall within `[min(x_train), max(x_train)]` yet still be far from any real training point (e.g. training `x` in `[0,10]` and `[90,100]`, query at `x=50`). `nil` (default, uncapped); applies regardless of `Extrapolation`.

## Example

### Basic Usage

```go
opts := fastlowess.DefaultOptions()
opts.Fraction = 0.7
opts.RetainModel = true

model, _ := fastlowess.NewLowess(opts)
defer model.Close()

x := []float64{1, 2, 3, 4, 5}
y := []float64{2.1, 4.0, 6.2, 8.0, 10.1}
result, _ := model.Fit(x, y)
defer result.PredictModel.Close()

prediction, _ := result.PredictModel.Predict([]float64{1.5, 4.5}, fastlowess.PredictOptions{})
fmt.Println(prediction.Y)
```

```output
[3.05 9.05]
```

### Standard Errors and Derivative

```go
prediction, _ := result.PredictModel.Predict([]float64{2.5}, fastlowess.PredictOptions{
 ReturnSE:         true,
 ReturnDerivative: true,
})
fmt.Println(prediction.Y, prediction.StandardErrors, prediction.Derivative)
```

```output
[5.1] [0] [2.2]
```

### Linear Extrapolation

```go
prediction, _ := result.PredictModel.Predict([]float64{10.0}, fastlowess.PredictOptions{
 Extrapolation: "linear",
})
fmt.Println(prediction.Y)
```

```output
[10.1]
```
