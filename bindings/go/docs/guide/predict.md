---
title: "Out-of-Sample Prediction"
weight: 40
---

Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

> Out-of-sample prediction is available in **Batch** mode only. Streaming and Online modes do not support it.

`(*PredictModel) Predict(newX, opts)` evaluates the fit at arbitrary query points, like R's `predict(model, newdata)`.

It reuses `Fit`'s own (possibly `Delta`-interpolated) smoothed curve for its `Y` output — so predicting at a training `x` always exactly reproduces that point's `Fit` output, regardless of `Delta`. A fresh local fit is only run when `"derivative"`, `"se"` (or an interval level), or `MaxNeighborDistance` needs the actual regression slope or standard error.

`Result.PredictModel` is non-nil only when `Options.RetainModel` was set to `true`. Call `Close()` on it when done (or let its finalizer run).

---

## Options

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `Outputs` | `[]string` | `nil` | Select `se` and/or `derivative` |
| `Intervals` | `*IntervalsOptions` | `nil` | Confidence/prediction levels and optional bootstrap refits |
| `Seed` | `*uint64` | `nil` | Reproducible prediction-time bootstrap draws; zero is valid |
| `Extrapolation` | `string` | `"clamp"` | Behavior for query points outside the training `x`-range |
| `MaxExtrapolationDistance` | `*float64` | `nil` | Under `"linear"` extrapolation, the max allowed distance beyond the training boundary before erroring |
| `MaxNeighborDistance` | `*float64` | `nil` | Max allowed distance to the farthest training point in a query's local window before erroring |

### Outputs

Request `"se"` for standard errors and `"derivative"` for the local slope at each query point. Analytic intervals compute their required standard errors even without an explicit `"se"` output.

### Intervals

Set `Intervals.Confidence` for bounds around the mean response and `Intervals.Prediction` for bounds of a new observation (e.g. `0.95`). With no bootstrap, these use normal-theory standard errors and the retained residual scale.

Set `Intervals.Bootstrap` to at least 2 to resample the retained Batch residuals, refit the model, and calculate query-point percentile intervals and standard errors. `Seed` on `PredictOptions` controls these draws independently of the fit/CV seed. Prediction bootstrap requires `Options.RetainModel = true`.

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
 Outputs: []string{"se", "derivative"},
})
fmt.Println(prediction.Y, prediction.StandardErrors, prediction.Derivative)
```

```output
[5.1] [0] [2.2]
```

### Bootstrap Intervals

```go
package main

import (
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess/v4"
)

func main() {
 x, y := make([]float64, 30), make([]float64, 30)
 for i := range x {
  x[i] = float64(i) * 0.1
  y[i] = math.Sin(x[i]) + 0.1*math.Cos(7*x[i])
 }
 opts := fastlowess.DefaultOptions()
 opts.RetainModel = true
 model, err := fastlowess.NewLowess(opts)
 if err != nil { panic(err) }
 defer model.Close()
 result, err := model.Fit(x, y)
 if err != nil { panic(err) }
 defer result.PredictModel.Close()

 level, seed := 0.95, uint64(42)
 predictOpts := fastlowess.PredictOptions{
  Outputs: []string{"se", "derivative"},
  Intervals: &fastlowess.IntervalsOptions{Confidence: &level, Prediction: &level, Bootstrap: 40},
  Seed: &seed,
 }
 predicted, err := result.PredictModel.Predict([]float64{0.75, 1.25}, predictOpts)
 if err != nil || len(predicted.ConfidenceLower) != 2 { panic("bootstrap Predict failed") }
}
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
