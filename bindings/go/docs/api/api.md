---
title: "API"
weight: 30
---

Batch `Lowess` reference. Best suited when the dataset fits in memory and you need intervals, cross-validation, or diagnostics.

> **StreamingLowess** and **OnlineLowess** are documented separately: [api-streaming.md](api-streaming.md), [api-online.md](api-online.md)

## When to Use Batch Adapter

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

## Classes

### `Lowess`

The `Lowess` type allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```go
opts := fastlowess.DefaultOptions()
opts.Fraction = 0.3
opts.ReturnDiagnostics = true

model, err := fastlowess.NewLowess(opts)
if err != nil {
 panic(err)
}
defer model.Close()
```

- `fastlowess.NewLowess(opts Options) (*Lowess, error)` creates a new batch model. Returns an error if any option is out of range (some validation is eager at construction, e.g. `Iterations`; some is deferred to `Fit`, e.g. `Fraction`).
- `opts`: An `Options` struct. Use `fastlowess.DefaultOptions()` and override only the fields you need.

**Methods:**

```go
result, err := model.Fit(x, y)
```

- `(*Lowess) Fit(x, y []float64, customWeights ...[]float64) (Result, error)` smooths `y` as a function of `x`. `x` and `y` must be non-empty and the same length. An optional `customWeights` slice (same length) applies per-observation case weights.
- `(*Lowess) Close() error` releases native resources. Safe to call multiple times. A finalizer is registered as a safety net, but call `Close` explicitly (e.g. via `defer`) rather than relying on the garbage collector.
- Returns a `Result` containing the smoothed values and optional diagnostics.

## Options Structures

### `Options`

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
| `ReturnSorted` | `bool` | `false` | Return results sorted ascending by `X` instead of in original input order. |
| `Parallel` | `bool` | `true` | Enable parallel processing. |
| `Backend` | `string` | `"cpu"` | `cpu` or `gpu` (requires a `gpu`-feature build of the native library). |
| `CVMethod` | `string` | `"kfold"` | `kfold` or `loocv`. |
| `CVK` | `int` | `5` | Number of folds for k-fold CV. |
| `CVFractions` | `[]float64` | `nil` (disabled) | Candidate fractions for cross-validation. |
| `CVSeed` | `*uint64` | `nil` (random) | RNG seed for reproducible k-fold splits. |

Use `fastlowess.DefaultOptions()` and override only the fields you need:

```go
opts := fastlowess.DefaultOptions()
opts.Fraction = 0.3
opts.ReturnDiagnostics = true
```

## Options

### Fraction

`Fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

### Iterations

`Iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

### Delta

Points within `Delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `nil` (default) auto-sets it to 1% of the x-range. Set it to `0.0` explicitly to disable interpolation and fit every point exactly.

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

### ScalingMethod

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### BoundaryPolicy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### ZeroWeightFallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### AutoConverge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `nil` (default) disables early stopping.

### ConfidenceIntervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the confidence interval around the mean response (e.g. `0.95`). `nil` (default) disables confidence intervals.

### PredictionIntervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the prediction interval for new observations (e.g. `0.95`). `nil` (default) disables prediction intervals.

### ReturnDiagnostics

*See: [`Diagnostics`](#diagnostics)*

Populate `Result.Diagnostics` (RMSE, MAE, R², AIC/AICc, effective degrees of freedom). AIC/AICc/`EffectiveDF` additionally require `ReturnSE: true` (or confidence/prediction intervals) to be populated, since they depend on hat-matrix statistics.

- `false` (default) — leaves `Result.Diagnostics` as `nil`
- `true` — populates `Result.Diagnostics`

### ReturnResiduals

Populate `Result.Residuals` (`Y - fitted`).

- `false` (default) — leaves `Result.Residuals` as `nil`
- `true` — populates `Result.Residuals`

### ReturnRobustnessWeights

Populate `Result.RobustnessWeights` (from the last robustness iteration).

- `false` (default) — leaves `Result.RobustnessWeights` as `nil`
- `true` — populates `Result.RobustnessWeights`

### ReturnSE

*See: [Intervals](../guide/intervals.md#standard-errors)*

Computes hat-matrix statistics (effective degrees of freedom, leverage, delta1/delta2) in addition to standard errors.

### ReturnSorted

When set to `true`, it reorders every result field (residuals, intervals, etc.) by `X` in an ascending manner, instead of in original input order.
To get both orderings, sort the default result client-side (e.g. via `sort.Slice`) instead of calling `Fit` twice.

### Parallel

Enable multi-threaded execution via Rayon.

- `true` (default) — parallelizes the local regression fits across CPU cores
- `false` — forces single-threaded execution (useful for benchmarking or deterministic profiling)

### Backend

*See: [GPU Backend](../advanced/gpu-backend.md)*

The batch `Lowess` type can optionally run on a GPU-accelerated backend powered by `wgpu`, for high-throughput processing of large datasets (10k+ points).

- `"cpu"` (default)
- `"gpu"` — requires a `gpu`-feature build of the native library

### CV Options

*See: [Cross-Validation](../guide/cross-validation.md)*

- `CVMethod`: `"kfold"` (default) — fast, evaluates each candidate fraction over `CVK` folds; `"loocv"` — slow, exhaustive leave-one-out cross-validation
- `CVK`: Number of folds for k-fold CV. Ignored when `CVMethod="loocv"`.
- `CVFractions`: Candidate fractions to evaluate. Cross-validation is disabled unless this is set.
- `CVSeed`: Seed for reproducible k-fold shuffling. `nil` (default) uses a random seed.

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

### CustomWeights

*See: [Custom Weights](../weighting/custom-weights.md)*

Per-observation weights, passed to `Fit` rather than the constructor.

```go
weights := make([]float64, len(x))
for i := range weights {
 weights[i] = 1.0
}
weights[0] = 5.0 // trust the first observation more

result, err := model.Fit(x, y, weights)
```

## Result Structure

### `Result`

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

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `RMSE` | `float64` | Root Mean Squared Error |
| `MAE` | `float64` | Mean Absolute Error |
| `RSquared` | `float64` | R-squared |
| `ResidualSD` | `float64` | Residual standard deviation |
| `EffectiveDF` | `float64` | Effective degrees of freedom |
| `AIC` | `float64` | AIC |
| `AICc` | `float64` | AICc |

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
