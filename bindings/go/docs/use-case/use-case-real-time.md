---
title: "Real-Time Processing"
weight: 85
---

Streaming and online LOWESS for live data.

## Overview

When data arrives continuously—from sensors, logs, or streaming pipelines—you need incremental smoothing that doesn't require reprocessing the entire dataset.

---

## Online Mode: Point-by-Point

For true real-time applications where each point must be processed immediately.

`WindowCapacity = 25` limits the internal buffer to the 25 most recent observations; each `AddPoint` call costs O(window) rather than growing with total history. `MinPoints = 5` suppresses output until the window holds enough points for a stable fit — calls made before that threshold return `ok == false`. `UpdateMode = "incremental"` re-fits only the most recent point rather than the full window, halving typical latency at a modest accuracy cost.

### Sensor Data Example

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func main() {
 opts := fastlowess.DefaultOnlineOptions()
 opts.Fraction = 0.3
 opts.Iterations = 1
 opts.WindowCapacity = 25
 opts.MinPoints = 5
 opts.UpdateMode = "incremental"

 model, err := fastlowess.NewOnlineLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 count := 0
 for i := 0; i < 100; i++ {
  xi := float64(i)
  yi := 20.0 + 5.0*math.Sin(xi/10.0) + math.Sin(xi*1.7)*0.5
  res, ok, err := model.AddPoint(xi, yi)
  if err != nil {
   log.Fatal(err)
  }
  if ok {
   if count < 5 {
    fmt.Printf("Time %v: smoothed = %.4f\n", xi, res.Y)
   }
   count++
  }
 }
 fmt.Printf("... (%d more)\n", count-5)
}
```

```output
Time 4: smoothed = 22.1941
Time 5: smoothed = 22.7964
Time 6: smoothed = 22.4733
Time 7: smoothed = 22.9120
Time 8: smoothed = 24.0164
... (91 more)
```

---

## Streaming Mode: Chunk Processing

For large datasets that arrive in batches or files.

`ChunkSize` controls how many data points are processed in one pass; matching it to your file-read buffer or message-batch size avoids unnecessary copying. `Overlap` retains that many points from the previous chunk as context so the neighbourhood at chunk boundaries is not artificially truncated. `MergeStrategy = "weighted_average"` blends the overlapping region smoothly; use `"last"` if chunk boundaries are guaranteed to be well separated and no blending is needed.

> **Always call Finalize():** The streaming adapter buffers overlap data. Call `Finalize()` after the last chunk to retrieve the buffered tail.

### Log File Processing

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func main() {
 chunk1X := make([]float64, 50)
 chunk1Y := make([]float64, 50)
 chunk2X := make([]float64, 50)
 chunk2Y := make([]float64, 50)
 for i := 0; i < 50; i++ {
  chunk1X[i] = float64(i)
  chunk1Y[i] = math.Sin(chunk1X[i]) + 0.1
  chunk2X[i] = float64(i + 50)
  chunk2Y[i] = math.Sin(chunk2X[i]) + 0.1
 }

 opts := fastlowess.DefaultStreamingOptions()
 opts.Fraction = 0.1
 opts.Iterations = 2
 opts.ChunkSize = 50
 opts.Overlap = 10
 opts.MergeStrategy = "weighted_average"

 model, err := fastlowess.NewStreamingLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 if _, err := model.ProcessChunk(chunk1X, chunk1Y); err != nil {
  log.Fatal(err)
 }
 if _, err := model.ProcessChunk(chunk2X, chunk2Y); err != nil {
  log.Fatal(err)
 }
 result, err := model.Finalize()
 if err != nil {
  log.Fatal(err)
 }
 fmt.Printf("y[0]: %.6f\n", result.Y[0])
}
```

```output
y[0]: 0.516484
```

---

## Real-Time Dashboard Example

The dashboard pattern uses a plain LOWESS fit on a manually managed sliding window rather than `OnlineLowess`. This is the simplest approach when your UI framework already owns the data buffer and you only need the most recent smoothed value per frame. The trade-off is a full O(window^2) refit on every tick; for high-frequency streams prefer `OnlineLowess` with `UpdateMode = "incremental"` to bound per-frame cost.

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

 var windowX, windowY []float64
 latest := 0.0

 for i := 0; i < n; i++ {
  windowX = append(windowX, x[i])
  windowY = append(windowY, y[i])
  if len(windowX) > 50 {
   windowX = windowX[1:]
   windowY = windowY[1:]
  }
  if len(windowX) < 2 {
   continue
  }

  opts := fastlowess.DefaultOptions()
  opts.Fraction = 0.4
  model, err := fastlowess.NewLowess(opts)
  if err != nil {
   log.Fatal(err)
  }
  result, err := model.Fit(windowX, windowY)
  model.Close()
  if err != nil {
   log.Fatal(err)
  }
  latest = result.Y[len(result.Y)-1]
 }

 fmt.Printf("Smoothed (dashboard, latest tick): %v\n", latest)
}
```

```output
Smoothed (dashboard, latest tick): -0.06634730089857399
```

---

## Choosing Parameters

### Online Mode

| Parameter | Guidance |
| --- | --- |
| `WindowCapacity` | Enough history for `Fraction` to work |
| `MinPoints` | 2–5 typically; higher for stability |
| `UpdateMode` | `"incremental"` for speed, `"full"` for accuracy |

### Streaming Mode

| Parameter | Guidance |
| --- | --- |
| `ChunkSize` | Balance memory vs. processing overhead |
| `Overlap` | 10–20% of ChunkSize for smooth transitions |
| `MergeStrategy` | `"weighted_average"` for best quality, `"average"` for simplicity |

---

## Performance Considerations

| Mode | Memory | Latency | Use Case |
| --- | --- | --- | --- |
| **Online** | Fixed (window) | ~1ms/point | Sensors, dashboards |
| **Streaming** | ~ChunkSize | ~100ms/chunk | Large files, ETL |
| **Batch** | Full dataset | N/A | Analysis, reports |

---

## See Also

- [Execution Modes](../guide/adapter-choice.md) — Detailed mode comparison
- [Merge Strategies](../advanced/merge.md) — Chunk reconciliation in depth
- [Scaling Methods](../weighting/scaling.md) — Robustness scale estimation
- [Time Series](use-case-time-series.md) — General time series analysis
