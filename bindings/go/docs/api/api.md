---
title: "API"
weight: 30
---

Batch `Lowess` reference. Best suited when the dataset fits in memory and you need intervals, cross-validation, or diagnostics.

> **StreamingLowess** and **OnlineLowess** are documented separately: [api-streaming.md](api-streaming.md), [api-online.md](api-online.md)

## `fastlowess.DefaultOptions() Options`

Returns recommended defaults. Start from this and override only the fields you need:

```go
opts := fastlowess.DefaultOptions()
opts.Fraction = 0.3
opts.ReturnDiagnostics = true
```

## `Options` fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `Fraction` | `float64` | `0.67` | Smoothing fraction, in (0, 1]. |
| `Iterations` | `int` | `3` | Robustness iterations, in [0, 1000]. |
| `Delta` | `*float64` | `nil` (auto) | Interpolation distance threshold, as a fraction of the x range. |
| `WeightFunction` | `string` | `"tricube"` | Kernel: `tricube`, `gaussian`, `uniform`, `cosine`, `epanechnikov`, `biweight`, `triangle`. |
| `RobustnessMethod` | `string` | `"bisquare"` | Outlier downweighting: `bisquare`, `huber`, `talwar`. |
| `ScalingMethod` | `string` | `"mad"` | Residual scale estimator: `mad`, `mar`, `mean`. |
| `BoundaryPolicy` | `string` | `"extend"` | Boundary handling: `extend`, `reflect`, `zero`, `noboundary`. |
| `ZeroWeightFallback` | `string` | `"use_local_mean"` | Fallback when all robustness weights hit zero: `use_local_mean`, `return_original`, `return_none`. |
| `ConfidenceIntervals` | `*float64` | `nil` (disabled) | Confidence level in (0, 1), e.g. `0.95`. |
| `PredictionIntervals` | `*float64` | `nil` (disabled) | Confidence level in (0, 1), e.g. `0.95`. |
| `AutoConverge` | `*float64` | `nil` (disabled) | Convergence tolerance for early stopping. |
| `ReturnDiagnostics` | `bool` | `false` | Populate `Result.Diagnostics`. |
| `ReturnResiduals` | `bool` | `false` | Populate `Result.Residuals`. |
| `ReturnRobustnessWeights` | `bool` | `false` | Populate `Result.RobustnessWeights`. |
| `ReturnSE` | `bool` | `false` | Populate `Result.StandardErrors` (hat-matrix statistics). |
| `CVFractions` | `[]float64` | `nil` (disabled) | Candidate fractions for cross-validation. |
| `CVMethod` | `string` | `"kfold"` | `kfold` or `loocv`. |
| `CVK` | `int` | `5` | Number of folds for k-fold CV. |
| `CVSeed` | `*uint64` | `nil` (random) | RNG seed for reproducible k-fold splits. |
| `Parallel` | `bool` | `true` | Enable parallel processing. |
| `Backend` | `string` | `"cpu"` | `cpu` or `gpu` (requires a `gpu`-feature build of the native library). |

`Fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

`Iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

## `fastlowess.NewLowess(opts Options) (*Lowess, error)`

Creates a new batch model. Returns an error if any option is out of range (some validation is eager at construction, e.g. `Iterations`; some is deferred to `Fit`, e.g. `Fraction`).

## `(*Lowess) Fit(x, y []float64, customWeights ...[]float64) (Result, error)`

Smooths `y` as a function of `x`. `x` and `y` must be non-empty and the same length. An optional `customWeights` slice (same length) applies per-observation case weights.

## `(*Lowess) Close() error`

Releases native resources. Safe to call multiple times. A finalizer is registered as a safety net, but call `Close` explicitly (e.g. via `defer`) rather than relying on the garbage collector.

## `Result` fields

| Field | Type | Populated when |
| --- | --- | --- |
| `X`, `Y` | `[]float64` | Always. |
| `StandardErrors` | `[]float64` | `ReturnSE` |
| `ConfidenceLower`, `ConfidenceUpper` | `[]float64` | `ConfidenceIntervals` set |
| `PredictionLower`, `PredictionUpper` | `[]float64` | `PredictionIntervals` set |
| `Residuals` | `[]float64` | `ReturnResiduals` |
| `RobustnessWeights` | `[]float64` | `ReturnRobustnessWeights` |
| `CVScores` | `[]float64` | `CVFractions` set |
| `FractionUsed` | `float64` | Always. |
| `IterationsUsed` | `int` | Always (`-1` if not available). |
| `Diagnostics` | `*Diagnostics` | `ReturnDiagnostics` |

`Diagnostics` holds `RMSE`, `MAE`, `RSquared`, `AIC`, `AICc`, `EffectiveDF`, `ResidualSD`.

## Options

### WeightFunction

*See: [Weight Functions](../weighting/kernels.md)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### RobustnessMethod

*See: [Robustness](../weighting/robustness.md)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### BoundaryPolicy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### ScalingMethod

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### ZeroWeightFallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

## Custom weights

```go
weights := make([]float64, len(x))
for i := range weights {
    weights[i] = 1.0
}
weights[0] = 5.0 // trust the first observation more

result, err := model.Fit(x, y, weights)
```

## Cross-validation

```go
opts := fastlowess.DefaultOptions()
opts.CVFractions = []float64{0.1, 0.2, 0.3, 0.5}
opts.CVMethod = "kfold"
opts.CVK = 5
seed := uint64(42)
opts.CVSeed = &seed

model, _ := fastlowess.NewLowess(opts)
defer model.Close()
result, _ := model.Fit(x, y)
fmt.Println(result.CVScores, result.FractionUsed) // FractionUsed = the CV-selected fraction
```

## Example

```go
package main

import (
 "fmt"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func main() {
 x := []float64{1, 2, 3, 4, 5}
 y := []float64{2.1, 4.0, 6.2, 8.0, 10.1}

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.5

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  panic(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  panic(err)
 }
 fmt.Println(result.Y)
}
```

```output
[2.1 4 6.2 8 10.1]
```
