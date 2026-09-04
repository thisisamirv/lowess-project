---
title: "StreamingLowess API"
weight: 32
---

For datasets that don't fit in memory or arrive in chunks. Processes data incrementally, merging overlapping regions between chunks.

See also: [API](api.md)

## When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

## Class

### `StreamingLowess`

The `StreamingLowess` type processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```go
opts := fastlowess.DefaultStreamingOptions()
opts.ChunkSize = 2000
opts.Overlap = 200

model, err := fastlowess.NewStreamingLowess(opts)
if err != nil {
 panic(err)
}
defer model.Close()
```

- `fastlowess.NewStreamingLowess(opts StreamingOptions) (*StreamingLowess, error)` creates a new streaming model with the given options.
- `opts`: A `StreamingOptions` struct (embeds `Options`).

**Methods:**

#### `ProcessChunk(x, y []float64) (Result, error)`

Feeds one chunk of data into the model. Each chunk is fit together with the trailing `Overlap` points buffered from the previous call, then only the points that are fully resolved are returned — the tail of the chunk (the next `Overlap` points) is held back internally, since it will be refit once the following chunk arrives and its estimate reconciled via `MergeStrategy`. This is what lets the adapter process a dataset far larger than memory allows, one bounded-size chunk at a time, without ever materializing the whole dataset at once.

```go
result, err := model.ProcessChunk(x, y)
```

#### `Finalize() (Result, error)`

Flushes the overlap points still buffered from the last `ProcessChunk` call. Because each call withholds its tail until the next chunk arrives to resolve it, the final chunk's tail would never be emitted otherwise — always call `Finalize` once after the last chunk to retrieve it.

```go
result, err := model.Finalize()
```

- `(*StreamingLowess) Close() error` releases native resources. Safe to call multiple times.

## Options Structure

### `StreamingOptions` (embeds `Options`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `Fraction` | `float64` | `0.67` | Smoothing fraction (bandwidth) |
| `Iterations` | `int` | `3` | Number of robustifying iterations |
| `Delta` | `*float64` | `nil` | Interpolation distance (`nil` auto-sets it to 0.0 in Streaming, i.e. interpolation disabled) |
| `WeightFunction` | `string` | `"tricube"` | Weight function name |
| `RobustnessMethod` | `string` | `"bisquare"` | Robustness method name |
| `ScalingMethod` | `string` | `"mad"` | Residual scaling method |
| `BoundaryPolicy` | `string` | `"extend"` | Boundary handling policy |
| `ZeroWeightFallback` | `string` | `"use_local_mean"` | Zero-weight handling |
| `AutoConverge` | `*float64` | `nil` | Auto-convergence tolerance |
| `ReturnDiagnostics` | `bool` | `false` | Include diagnostics in result |
| `ReturnResiduals` | `bool` | `false` | Include residuals in result |
| `ReturnRobustnessWeights` | `bool` | `false` | Include weights in result |
| `Parallel` | `bool` | `true` | Enable parallel execution |
| `ChunkSize` | `int` | `5000` | Data chunk size |
| `Overlap` | `int` | library default | Overlap between chunks (negative means "use the library default") |
| `MergeStrategy` | `string` | `"weighted_average"` | Strategy for blending overlap regions |

Confidence/prediction intervals, standard errors, cross-validation, GPU `Backend`, `CustomWeights`, and `ReturnSorted` are Batch-only and not available here; see [API](api.md) for those.

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

Points within `Delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `nil` (default) auto-sets it to `0` in Streaming mode, i.e. interpolation is disabled and every point is fit exactly.

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

### ChunkSize

Number of points processed per chunk. Larger chunks reduce per-chunk overhead and give each local fit more surrounding context, at the cost of higher peak memory; smaller chunks bound memory tightly but increase the fraction of points that fall in overlap regions. A good starting point is balancing available memory against how much processing overhead per chunk is acceptable — match it to your file-read buffer or message-batch size to avoid unnecessary copying.

### Overlap

Number of points retained from the previous chunk as context, so the neighbourhood at chunk boundaries isn't artificially truncated. Points inside the overlap zone are fitted twice (once by each chunk) and reconciled via `MergeStrategy`. A good starting point is 10–20% of `ChunkSize`: too little overlap causes visible boundary artefacts, while too much wastes computation refitting the same points twice.

### MergeStrategy

*See: [Merge Strategies](../advanced/merge.md)*

| Strategy | Behavior |
| --- | --- |
| `"weighted_average"` (default) | Distance-weighted blend |
| `"average"` | Average overlapping values |
| `"take_first"` | Keep left chunk values |
| `"take_last"` | Keep right chunk values |

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

---

> **Always call Finalize():** The streaming adapter buffers overlap data. Call `Finalize` after the last chunk to retrieve the buffered tail.

## Result Structure

### `Result`

Returned by `ProcessChunk` and `Finalize`.

| Field | Type | Populated when |
| --- | --- | --- |
| `X`, `Y` | `[]float64` | Always. |
| `StandardErrors` | `[]float64` | Always empty (Batch only) |
| `ConfidenceLower`, `ConfidenceUpper` | `[]float64` | Always empty (Batch only) |
| `PredictionLower`, `PredictionUpper` | `[]float64` | Always empty (Batch only) |
| `Residuals` | `[]float64` | `ReturnResiduals` |
| `RobustnessWeights` | `[]float64` | `ReturnRobustnessWeights` |
| `CVScores` | `[]float64` | Always empty (Batch only) |
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
| `EffectiveDF` | `float64` | Always `NaN` (requires standard errors, Batch only) |
| `AIC` | `float64` | Always `NaN` (requires `EffectiveDF`, Batch only) |
| `AICc` | `float64` | Always `NaN` (requires `EffectiveDF`, Batch only) |

## Example

```go
package main

import (
 "fmt"
 "log"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func main() {
 const n = 20
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i)
  y[i] = float64(i) + 0.1
 }

 opts := fastlowess.DefaultStreamingOptions()
 opts.ChunkSize = 10
 opts.Overlap = 2

 model, err := fastlowess.NewStreamingLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 if _, err := model.ProcessChunk(x[:10], y[:10]); err != nil {
  log.Fatal(err)
 }
 if _, err := model.ProcessChunk(x[10:], y[10:]); err != nil {
  log.Fatal(err)
 }

 result, err := model.Finalize()
 if err != nil {
  log.Fatal(err)
 }
 fmt.Printf("y[0]: %.4f\n", result.Y[0])
}
```

```output
y[0]: 17.6391
```
