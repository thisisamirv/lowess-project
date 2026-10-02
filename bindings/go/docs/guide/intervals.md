---
title: "Intervals"
weight: 60
---

Confidence and prediction intervals for uncertainty quantification.

## Overview

![Confidence and Prediction Intervals](../assets/diagrams/intervals_comparison.svg)

> **Adapter support:** Confidence and prediction intervals are available in **Batch** mode, **Streaming** mode (computed per chunk and merged across overlap boundaries via `MergeStrategy`, like `Y`/`Derivative`), and **Online** mode when `UpdateMode = "full"` is set (errors if combined with the default `"incremental"` mode).

Confidence and prediction coverage levels are independent; for example, a 90% confidence interval can be paired with a 99% prediction interval.

| Type | Represents | Width | Use |
| --- | --- | --- | --- |
| **Confidence** | Uncertainty in mean curve | Narrow | Where is the true trend? |
| **Prediction** | Uncertainty for new points | Wide | Where will new data fall? |

---

## Confidence Intervals

Estimate uncertainty in the smoothed curve itself.

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess/v4"
)

func main() {
 n := 100
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(x[i]) + 0.1
 }

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.5
 ci := 0.95 // 95% CI
 opts.Intervals = &fastlowess.IntervalsOptions{Confidence: &ci}

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }

 for i := 0; i < 3; i++ {
  fmt.Printf("x=%.2f: y=%.2f [%.2f, %.2f]\n",
   result.X[i], result.Y[i], result.ConfidenceLower[i], result.ConfidenceUpper[i])
 }
}
```

```output
x=0.00: y=0.33 [0.29, 0.37]
x=0.06: y=0.36 [0.32, 0.40]
x=0.13: y=0.39 [0.35, 0.43]
```

---

## Prediction Intervals

Estimate where new observations might fall.

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess/v4"
)

func main() {
 n := 100
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(x[i]) + 0.1
 }

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.5
 pi := 0.95 // 95% PI
 opts.Intervals = &fastlowess.IntervalsOptions{Prediction: &pi}

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Printf("Prediction bounds: [%.2f, %.2f]\n", result.PredictionLower[0], result.PredictionUpper[0])
}
```

```output
Prediction bounds: [-0.04, 0.71]
```

---

## Both Intervals

Request both types simultaneously:

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess/v4"
)

func main() {
 n := 100
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(x[i]) + 0.1
 }

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.5
 ci := 0.95
 pi := 0.95
 opts.Intervals = &fastlowess.IntervalsOptions{Confidence: &ci, Prediction: &pi}

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Printf("First point 95%% CI: [%v, %v]\n", result.ConfidenceLower[0], result.ConfidenceUpper[0])
}
```

```output
First point 95% CI: [0.2929290861928646, 0.3721536723881128]
```

---

## Confidence Levels

Common levels and their z-values:

| Level | z-value | Interpretation |
| --- | --- | --- |
| 0.90 | 1.645 | 90% of intervals contain true value |
| 0.95 | 1.960 | 95% of intervals contain true value |
| 0.99 | 2.576 | 99% of intervals contain true value |

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess/v4"
)

func main() {
 n := 100
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(x[i]) + 0.1
 }

 // 99% confidence interval
 opts := fastlowess.DefaultOptions()
 ci := 0.99
 opts.Intervals = &fastlowess.IntervalsOptions{Confidence: &ci}

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First lower CI bound (99%):", result.ConfidenceLower[0])
}
```

```output
First lower CI bound (99%): 0.31746448671917393
```

---

## Residual Bootstrap

Set `Intervals.Bootstrap` to at least 2 to replace analytic uncertainty with residual-bootstrap refits. Batch shares its outer `Seed` with CV; Streaming restarts the seed per combined chunk, and Online per full-update window. Online bootstrap requires `UpdateMode = "full"`.

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
 level, seed := 0.95, uint64(42)
 intervals := &fastlowess.IntervalsOptions{Confidence: &level, Prediction: &level, Bootstrap: 20}

 batchOpts := fastlowess.DefaultOptions()
 batchOpts.Intervals, batchOpts.Seed = intervals, &seed
 batch, err := fastlowess.NewLowess(batchOpts)
 if err != nil { panic(err) }
 defer batch.Close()
 fitted, err := batch.Fit(x, y)
 if err != nil || len(fitted.StandardErrors) != len(x) { panic("batch bootstrap failed") }

 streamOpts := fastlowess.DefaultStreamingOptions()
 streamOpts.ChunkSize = len(x)
 streamOpts.Intervals, streamOpts.Seed = intervals, &seed
 stream, err := fastlowess.NewStreamingLowess(streamOpts)
 if err != nil { panic(err) }
 defer stream.Close()
 chunk, err := stream.ProcessChunk(x, y)
 if err != nil || len(chunk.ConfidenceLower) == 0 { panic("streaming bootstrap failed") }

 onlineOpts := fastlowess.DefaultOnlineOptions()
 onlineOpts.MinPoints = 5
 onlineOpts.UpdateMode = "full"
 onlineOpts.Intervals, onlineOpts.Seed = intervals, &seed
 online, err := fastlowess.NewOnlineLowess(onlineOpts)
 if err != nil { panic(err) }
 defer online.Close()
 ready := false
 for i := range x[:12] {
  point, ok, err := online.AddPoint(x[i], y[i])
  if err != nil { panic(err) }
  if ok { ready = !math.IsNaN(point.PredictionLower) }
 }
 if !ready { panic("online bootstrap failed") }
}
```

---

## Standard Errors

Access standard errors directly (available when intervals are computed):

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess/v4"
)

func main() {
 n := 100
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(x[i]) + 0.1
 }

 opts := fastlowess.DefaultOptions()
 opts.Outputs = []string{"se"}
 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 for i := 0; i < 3; i++ {
  fmt.Printf("Point %d: SE = %.4f\n", i, result.StandardErrors[i])
 }
}
```

```output
Point 0: SE = 0.0246
Point 1: SE = 0.0252
Point 2: SE = 0.0258
```

---

## Availability

> **Supported In All Three Adapters:** Confidence and prediction intervals are available in **Batch**, **Streaming**, and **Online** mode (`UpdateMode = "full"` only).

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✓ | ✓ (`UpdateMode = "full"` only) |
| Prediction intervals | ✓ | ✓ | ✓ (`UpdateMode = "full"` only) |
| Standard errors | ✓ | ✓ | ✓ (`UpdateMode = "full"` only) |
