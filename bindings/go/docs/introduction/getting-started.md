---
title: "Getting Started"
weight: 10
---

## Installation

Add the module to your `go.mod`:

```sh
go get github.com/thisisamirv/lowess-project/bindings/go/fastlowess
```

This package uses `cgo`, so `CGO_ENABLED=1` and a working C compiler (GCC or Clang; on Windows, a MinGW-w64 toolchain) are required at build time. See [installation.md](installation.md) for details on providing the native `fastlowess_go` library.

## A first fit

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func main() {
 n := 100
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i) / 10
  y[i] = math.Sin(x[i]) + 0.1*math.Sin(float64(i))
 }

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.2 // smaller fraction = less smoothing, more local detail

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }

 for i := 0; i < 5; i++ {
  fmt.Printf("x=%.2f  y=%.3f  smoothed=%.3f\n", result.X[i], y[i], result.Y[i])
 }
}
```

```output
x=0.00  y=0.000  smoothed=0.166
x=0.10  y=0.184  smoothed=0.215
x=0.20  y=0.290  smoothed=0.270
x=0.30  y=0.310  smoothed=0.331
x=0.40  y=0.314  smoothed=0.395
```

`fraction` is the most important tuning parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

## Choosing a model

- **`Lowess`** (batch): the whole dataset fits in memory and you want every feature (intervals, cross-validation, GPU). Start here.
- **`StreamingLowess`**: the dataset doesn't fit in memory or arrives in chunks.
- **`OnlineLowess`**: you need a smoothed value immediately as each point arrives (real-time).

See [api.md](../api/api.md), [api-streaming.md](../api/api-streaming.md), and [api-online.md](../api/api-online.md) for the full reference of each.

## Handling outliers

LOWESS can robustly handle outliers through iterative reweighting:

```go
package main

import (
 "fmt"
 "log"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func main() {
 x := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
 yWithOutlier := []float64{2.0, 4.0, 6.0, 50.0, 10.0, 12.0} // 50.0 is an outlier

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.7
 opts.Iterations = 3                 // more iterations for outliers
 opts.RobustnessMethod = "bisquare"  // default, smooth downweighting
 opts.ReturnRobustnessWeights = true // see which points were downweighted

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, yWithOutlier)
 if err != nil {
  log.Fatal(err)
 }

 // Outliers will have low robustness weights
 for i, w := range result.RobustnessWeights {
  if w < 0.5 {
   fmt.Printf("Point %d is likely an outlier (weight: %.3f)\n", i, w)
  }
 }
}
```

```output
Point 3 is likely an outlier (weight: 0.000)
```

## Streaming mode

For datasets too large to fit in memory, stream them in fixed-size chunks with overlap:

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func main() {
 n := 5000
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i) * 10.0 * math.Pi / float64(n-1)
  y[i] = math.Sin(x[i]/math.Pi) * math.Exp(-x[i]/30.0)
 }

 opts := fastlowess.DefaultStreamingOptions()
 opts.Fraction = 0.2
 opts.ChunkSize = 1000
 opts.Overlap = 100

 model, err := fastlowess.NewStreamingLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 for i := 0; i < n; i += opts.ChunkSize {
  end := i + opts.ChunkSize
  if end > n {
   end = n
  }
  if _, err := model.ProcessChunk(x[i:end], y[i:end]); err != nil {
   log.Fatal(err)
  }
 }

 result, err := model.Finalize()
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("Smoothed", len(result.Y), "points in the final chunk")
}
```

```output
Smoothed 100 points in the final chunk
```

## Next steps

| Topic | Link |
| --- | --- |
| How LOWESS works | [Concepts](concepts.md) |
| All parameters explained | [API Reference](../api/api.md) |
| Batch vs Streaming vs Online | [Execution Modes](../guide/adapter-choice.md) |
| Edge handling | [Boundary Handling](../advanced/boundary.md) |
| Outlier handling in depth | [Robustness](../weighting/robustness.md) |
| Kernel functions | [Weight Functions](../weighting/kernels.md) |
| Residual scale estimation | [Scaling Methods](../weighting/scaling.md) |
| Automated parameter selection | [Cross-Validation](../guide/cross-validation.md) |
| Per-observation weights | [Custom Weights](../weighting/custom-weights.md) |
| Reconciling streaming chunk overlaps | [Merge Strategies](../advanced/merge.md) |
| Running on the GPU | [GPU Backend](../advanced/gpu-backend.md) |
