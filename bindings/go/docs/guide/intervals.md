---
title: "Intervals"
weight: 60
---

Confidence and prediction intervals for uncertainty quantification.

## Overview

![Confidence and Prediction Intervals](../assets/diagrams/intervals_comparison.svg)

> **Adapter support:** Confidence and prediction intervals are available in **Batch** mode only. Streaming and Online modes do not support intervals.

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
 ci := 0.95 // 95% CI
 opts.ConfidenceIntervals = &ci

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
x=0.13: y=0.39 [0.34, 0.43]
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
 pi := 0.95 // 95% PI
 opts.PredictionIntervals = &pi

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
 ci := 0.95
 pi := 0.95
 opts.ConfidenceIntervals = &ci
 opts.PredictionIntervals = &pi

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
First point 95% CI: [0.29412988645250654, 0.3746108343833248]
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

 // 99% confidence interval
 opts := fastlowess.DefaultOptions()
 ci := 0.99
 opts.ConfidenceIntervals = &ci

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
First lower CI bound (99%): 0.3171427263910015
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
 opts.ReturnSE = true
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
Point 0: SE = 0.0254
Point 1: SE = 0.0269
Point 2: SE = 0.0283
```

---

## Availability

> **Batch Mode Only:** Confidence and prediction intervals are only available in **Batch** mode. `StreamingLowess` and `OnlineLowess` do not support intervals.

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✗ | ✗ |
| Prediction intervals | ✓ | ✗ | ✗ |
| Standard errors | ✓ | ✗ | ✗ |
