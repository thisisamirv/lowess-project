---
title: "Execution Modes"
weight: 25
---
<!-- markdownlint-disable MD024 -->
Choose the right adapter for your use case.

## Overview

Choose the first row below whose condition applies:

| Condition | Adapter |
| --- | --- |
| Data too large to fit in memory | `StreamingLowess` |
| Fits in memory, need real-time/incremental updates | `OnlineLowess` |
| Fits in memory, no real-time requirement | `Lowess` |

| Mode | Use Case | Memory | Features |
| --- | --- | --- | --- |
| **Batch** (`Lowess`) | Complete datasets | Full | All features |
| **Streaming** (`StreamingLowess`) | Large files (>100K) | Chunked | Residuals, robustness |
| **Online** (`OnlineLowess`) | Real-time sensors | Fixed window | Incremental updates |

![Adapter Comparison](../assets/diagrams/adapter_comparison.svg)

---

## Batch Adapter

Standard mode for complete datasets. **Supports all features.**

### When to Use

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

### Example

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
  x[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(x[i]) + 0.1
 }

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.5
 opts.Iterations = 3
 ci := 0.95
 opts.ConfidenceIntervals = &ci
 pi := 0.95
 opts.PredictionIntervals = &pi
 opts.ReturnDiagnostics = true
 opts.Parallel = true

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 mid := n / 2
 fmt.Printf("95%% CI at midpoint: [%.4f, %.4f]\n", result.ConfidenceLower[mid], result.ConfidenceUpper[mid])
 fmt.Printf("R2: %.4f\n", result.Diagnostics.RSquared)
}
```

---

## Streaming Adapter

Process large datasets in chunks with configurable overlap.

### When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `chunk_size` | 5000 | Points per chunk |
| `overlap` | `chunk_size / 10` | Overlap between chunks |
| `merge_strategy` | `"weighted_average"` | How to merge overlaps |

### Merge Strategies

| Strategy | Behavior |
| --- | --- |
| `"average"` | Average overlapping values |
| `"weighted_average"` | Distance-weighted blend |
| `"take_first"` | Keep left chunk values |
| `"take_last"` | Keep right chunk values |

### Example

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
  x[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(x[i]) + 0.1
 }

 opts := fastlowess.DefaultStreamingOptions()
 opts.Fraction = 0.3
 opts.Iterations = 2
 opts.ChunkSize = 5000
 opts.Overlap = 500
 opts.MergeStrategy = "average"

 model, err := fastlowess.NewStreamingLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 if _, err := model.ProcessChunk(x, y); err != nil {
  log.Fatal(err)
 }
 result, err := model.Finalize()
 if err != nil {
  log.Fatal(err)
 }
 fmt.Printf("Smoothed y[0]: %.4f\n", result.Y[0])
}
```

---

> **Always call Finalize():** Always call `model.Finalize()` after processing all chunks to retrieve buffered overlap data.

## Online Adapter

Incremental updates with a sliding window for real-time data.

### When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `window_capacity` | 1000 | Max points in window |
| `min_points` | 2 | Points before output starts |
| `update_mode` | `"incremental"` | Update strategy |

### Update Modes

| Mode | Behavior | Speed |
| --- | --- | --- |
| `"incremental"` | Update only affected fits | Faster |
| `"full"` | Recompute entire window | More accurate |

### Example

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
  x[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(x[i]) + 0.1
 }

 opts := fastlowess.DefaultOnlineOptions()
 opts.Fraction = 0.2
 opts.Iterations = 1
 opts.WindowCapacity = 100
 opts.MinPoints = 5
 opts.UpdateMode = "incremental"

 model, err := fastlowess.NewOnlineLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 shown := 0
 for i := 0; i < len(x) && shown < 5; i++ {
  res, ok, err := model.AddPoint(x[i], y[i])
  if err != nil {
   log.Fatal(err)
  }
  if ok {
   fmt.Println(res.Y)
   shown++
  }
 }
}
```

---

## Feature Comparison

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✗ | ✗ |
| Prediction intervals | ✓ | ✗ | ✗ |
| Cross-validation | ✓ | ✗ | ✗ |
| Diagnostics | ✓ | ✓ | ✗ |
| Residuals | ✓ | ✓ | ✓ |
| Robustness weights | ✓ | ✓ | ✓ |
| Parallel execution | ✓ | ✓ | ✗ |

---

## Next Steps

- [API Reference](../api/api.md) — All configuration options
- [Streaming API](../api/api-streaming.md) · [Online API](../api/api-online.md)
