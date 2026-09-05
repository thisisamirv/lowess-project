---
title: "OnlineLowess API"
weight: 34
---

For real-time data: processes one `(x, y)` point at a time and returns a smoothed value immediately once enough points have been seen.

See also: [API](api.md)

## When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

![Online Adapter](../assets/diagrams/online_comparison.svg)

## Class

### `OnlineLowess`

The `OnlineLowess` type updates the model incrementally with new data points.

**Constructor:**

```go
opts := fastlowess.DefaultOnlineOptions()
opts.WindowCapacity = 200
opts.MinPoints = 10

model, err := fastlowess.NewOnlineLowess(opts)
if err != nil {
 panic(err)
}
defer model.Close()
```

- `fastlowess.NewOnlineLowess(opts OnlineOptions) (*OnlineLowess, error)` creates a new online model with the given options.
- `opts`: An `OnlineOptions` struct.

**Methods:**

#### `AddPoint(x, y float64) (res PointResult, ok bool, err error)`

Adds a single point to the sliding window and returns the smoothed value for that point; `ok` is `false` while the window is still filling up (fewer than `MinPoints` seen so far). Once the window reaches `WindowCapacity`, each new point evicts the oldest one, so memory stays bounded regardless of how much history has passed through. `UpdateMode` controls how much work each call does: `"incremental"` re-fits only the newest point, while `"full"` re-smooths the entire window for a more accurate but slower result.

```go
res, ok, err := model.AddPoint(x, y)
if err != nil {
 panic(err)
}
if ok {
 fmt.Println(res.Y)
}
```

- `(*OnlineLowess) Close() error` releases native resources. Safe to call multiple times.

## Options Structure

### `OnlineOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `Fraction` | `float64` | `0.67` | Smoothing fraction (bandwidth) |
| `Iterations` | `int` | `3` | Number of robustifying iterations |
| `Delta` | `*float64` | `nil` | Interpolation distance (`nil` auto-sets it to 0.0 in Online, i.e. interpolation disabled) |
| `WeightFunction` | `string` | `"tricube"` | Weight function name |
| `RobustnessMethod` | `string` | `"bisquare"` | Robustness method name |
| `ScalingMethod` | `string` | `"mad"` | Residual scaling method |
| `BoundaryPolicy` | `string` | `"extend"` | Boundary handling policy |
| `ZeroWeightFallback` | `string` | `"use_local_mean"` | Zero-weight handling |
| `Missing` | `string` | `"error"` | Policy for non-finite (NaN/Inf) values in each point |
| `AutoConverge` | `*float64` | `nil` | Auto-convergence tolerance |
| `ReturnRobustnessWeights` | `bool` | `false` | Include `RobustnessWeight` in result |
| `WindowCapacity` | `int` | `1000` | Maximum number of recent points retained |
| `MinPoints` | `int` | `2` | Minimum points required before output starts |
| `UpdateMode` | `string` | `"incremental"` | How the window is updated as new points arrive |

Confidence/prediction intervals, standard errors, cross-validation, GPU `Backend`, `CustomWeights`, `ReturnSorted`, `ReturnDiagnostics`, `ReturnResiduals`, and `Parallel` are Batch-only (or Batch/Streaming-only) and not available here; see [API](api.md) for those.

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

Points within `Delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `nil` (default) auto-sets it to `0` in Online mode, i.e. interpolation is disabled and every point is fit exactly.

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

### Missing

Policy for handling a non-finite (NaN/Inf) `x` or `y` value passed to `AddPoint`:

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Return an error |
| `"drop"` | Silently ignore the point — `AddPoint` returns `ok=false` instead of adding it to the window |

### AutoConverge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `nil` (default) disables early stopping.

### ReturnRobustnessWeights

Include the robustness weight for the latest point (from the last robustness iteration) in the result.

- `false` (default) — leaves `PointResult.RobustnessWeight` as `NaN`
- `true` — populates `PointResult.RobustnessWeight`

### WindowCapacity

Maximum number of most recent points kept in the sliding window; older points are discarded as new ones arrive. Each `AddPoint` call costs O(`WindowCapacity`) rather than growing with total history.

### MinPoints

Minimum number of points required before smoothing starts. `AddPoint` returns `ok == false` until the window reaches this size.

### UpdateMode

*See: [Execution Modes](../guide/adapter-choice.md)*

| Mode | Alias | Behavior | Speed |
| --- | --- | --- | --- |
| `"incremental"` (default) | `"single"` | Update only affected fits | Faster |
| `"full"` | `"resmooth"` | Recompute entire window | More accurate |

## Result Structure

### `PointResult`

Returned by `AddPoint` once the window has enough points (`ok == false` until then).

| Field | Type | Notes |
| --- | --- | --- |
| `Y` | `float64` | Smoothed value for the latest point. |
| `StandardError` | `float64` | Always `NaN` — standard errors require `ReturnSE`/confidence intervals, which are Batch-only. |
| `Residual` | `float64` | Residual y − smoothed; always populated (there is no `ReturnResiduals` option for Online). |
| `RobustnessWeight` | `float64` | Robustness weight, if `ReturnRobustnessWeights` was set. |
| `IterationsUsed` | `int` | Robustness iterations performed (`-1` if not applicable). |

There is no diagnostics structure or `ReturnDiagnostics` option for `OnlineLowess`: `PointResult` carries no diagnostics field, since diagnostics like RMSE/R² need more than one point's worth of history to be meaningful.

## Example

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func main() {
 const n = 100
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(x[i]) + 0.1
 }

 opts := fastlowess.DefaultOnlineOptions()
 opts.Fraction = 0.5
 opts.WindowCapacity = 50
 opts.MinPoints = 3

 model, err := fastlowess.NewOnlineLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 _, ok1, err := model.AddPoint(x[0], y[0])
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println(ok1)

 _, ok2, err := model.AddPoint(x[1], y[1])
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println(ok2)

 res, ok3, err := model.AddPoint(x[2], y[2])
 if err != nil {
  log.Fatal(err)
 }
 if ok3 {
  fmt.Println(res.Y)
 }
}
```

```output
false
false
0.22659245357374927
```
