---
title: "Genomic Data Smoothing"
weight: 80
---

LOWESS for methylation profiles, ChIP-seq signals, and other genomic data.

## Overview

Genomic data often contains noise from sequencing depth variation, PCR artifacts, or biological heterogeneity. LOWESS smoothing helps reveal underlying patterns.

---

## Methylation Profile Smoothing

### The Challenge

DNA methylation data (from bisulfite sequencing or arrays) shows position-dependent patterns that can be obscured by measurement noise.

### Solution

A small `Fraction = 0.1` lets LOWESS follow fine-scale spatial structure without smearing the transitions between methylated and unmethylated regions. `ConfidenceIntervals = 0.95` produces uncertainty bands that naturally widen at positions with sparser CpG coverage, making low-confidence segments immediately apparent in the plot.

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
 positions := make([]float64, n)
 observed := make([]float64, n)
 for i := 0; i < n; i++ {
  positions[i] = float64(i) * 1000.0
  observed[i] = 50.0 + math.Sin(positions[i]/1000.0)*20.0 + 5.0
 }

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.1
 opts.Iterations = 3
 ci := 0.95
 opts.ConfidenceIntervals = &ci

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(positions, observed)
 if err != nil {
  log.Fatal(err)
 }

 fmt.Printf("95%% CI: [%v, %v]\n", result.ConfidenceLower[0], result.ConfidenceUpper[0])
}
```

```output
95% CI: [51.67728847225473, 68.73718630709102]
```

---

## ChIP-seq Signal Smoothing

### Application

ChIP-seq experiments produce sparse, noisy coverage data. LOWESS can help identify binding regions.

`Fraction = 0.05` provides high spatial resolution — important for resolving narrow binding peaks that would otherwise be smeared into the background. The larger `Iterations = 5` is deliberate: Poisson-distributed read counts produce tall, isolated spikes, and extra robustness iterations progressively down-weight them so the estimated background level is not inflated by a handful of extreme counts.

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
 positions := make([]float64, n)
 observed := make([]float64, n)
 for i := 0; i < n; i++ {
  positions[i] = float64(i) * 1000.0
  observed[i] = 50.0 + math.Sin(positions[i]/1000.0)*20.0 + 5.0
 }

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.05
 opts.Iterations = 5

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(positions, observed)
 if err != nil {
  log.Fatal(err)
 }

 // Find peaks above threshold
 peakCount := 0
 for _, y := range result.Y {
  if y > 65.0 {
   peakCount++
  }
 }

 fmt.Println("y[0]:", result.Y[0])
 fmt.Println("Peak count:", peakCount)
}
```

```output
y[0]: 59.951990929979125
Peak count: 26
```

---

## Large Genome Coverage (Streaming)

For whole-genome data that doesn't fit in memory:

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func main() {
 n := 1001
 xChunk := make([]float64, n)
 yChunk := make([]float64, n)
 for i := 0; i < n; i++ {
  xChunk[i] = float64(i) * 10.0
  yChunk[i] = 50.0 + math.Sin(xChunk[i]/100.0)*20.0 + 5.0
 }

 opts := fastlowess.DefaultStreamingOptions()
 opts.Fraction = 0.05
 opts.Iterations = 3
 opts.ChunkSize = 50
 opts.Overlap = 10
 opts.MergeStrategy = "weighted_average"

 model, err := fastlowess.NewStreamingLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 if _, err := model.ProcessChunk(xChunk, yChunk); err != nil {
  log.Fatal(err)
 }
 result, err := model.Finalize()
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("y[0]:", result.Y[0])
}
```

```output
y[0]: 41.29765849569398
```

---

## Best Practices for Genomic Data

| Consideration | Recommendation |
| --- | --- |
| **Fraction** | 0.05–0.15 (preserve local features) |
| **Iterations** | 3–5 (handle sequencing outliers) |
| **Large data** | Use streaming mode |
| **Sparse regions** | Use `BoundaryPolicy = "extend"` |
| **Multiple chromosomes** | Process separately or ensure sorted |

---

## See Also

- [Concepts](../introduction/concepts.md) — How LOWESS works
- [API Reference](../api/api.md) — All options
- [Robustness](../weighting/robustness.md) — Outlier downweighting in depth
- [Merge Strategies](../advanced/merge.md) — Streaming chunk reconciliation
- [Boundary Handling](../advanced/boundary.md) — Edge handling for sparse regions
- [Real-Time Processing](use-case-real-time.md) — For sequencing runs
