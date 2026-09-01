---
title: "Time Series Analysis"
weight: 90
---

LOWESS for trend extraction and temporal smoothing.

## Overview

Time series data often contains noise, seasonality, and trends. LOWESS provides flexible trend extraction without parametric assumptions.

---

## Basic Trend Extraction

`Fraction = 0.1` sizes the neighbourhood as 10% of the data at each evaluation point — narrow enough to follow a slowly varying trend without smearing periodic variation. Three robustness `Iterations` down-weight noise spikes so they cannot bias the fitted curve.

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func main() {
 n := 500
 t := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  t[i] = float64(i) * 100.0 / float64(n-1)
  y[i] = 10.0 + 0.5*t[i] + 3.0*math.Sin(t[i]/10.0) + (math.Mod(float64(i*7+3), 1.7)-0.85)*3.0
 }

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.1
 opts.Iterations = 3

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(t, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("y[0]:", result.Y[0])
}
```

```output
y[0]: 11.321590922416165
```

---

## Detrending

Remove trend to analyze residual patterns.

Setting `ReturnResiduals = true` stores `observed − smoothed` alongside the smooth. A slightly wider `Fraction = 0.3` produces a smoother baseline trend, so short-duration oscillations end up in the residuals rather than being absorbed into the trend component.

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
 t := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  t[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(t[i]) + 0.1
 }

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.3
 opts.Iterations = 3
 opts.ReturnResiduals = true

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(t, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("residuals[0]:", result.Residuals[0])
}
```

```output
residuals[0]: -0.15824361645514948
```

---

## Forecasting with Prediction Intervals

Prediction intervals widen the uncertainty band to include both the uncertainty in the fitted curve (confidence interval) and the expected scatter of new observations around it. `Fraction = 0.2` offers a balance between local detail and stable interval width.

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
 t := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  t[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(t[i]) + 0.1
 }

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.2
 opts.Iterations = 3
 ci := 0.95
 pi := 0.95
 opts.ConfidenceIntervals = &ci
 opts.PredictionIntervals = &pi

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(t, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Printf("95%% PI: [%v, %v]\n", result.PredictionLower[0], result.PredictionUpper[0])
}
```

```output
95% PI: [0.15801046224996285, 0.2908827214492591]
```

---

## Handling Missing Data

LOWESS naturally handles irregular time sampling:

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
 tIrregular := make([]float64, n)
 yIrregular := make([]float64, n)
 for i := 0; i < n; i++ {
  tIrregular[i] = float64(i)*1.0 + float64((i*31)%10)*0.1
  yIrregular[i] = 10.0 + 0.3*tIrregular[i] + 2.0*math.Sin(tIrregular[i]*0.1)
 }

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.2

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(tIrregular, yIrregular)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("y[0]:", result.Y[0])
}
```

```output
y[0]: 11.327309510260003
```

---

## Multi-Scale Analysis

Use different fractions to extract features at different scales:

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
 t := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  t[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(t[i]) + 0.1
 }

 for _, f := range []float64{0.05, 0.2, 0.5} {
  opts := fastlowess.DefaultOptions()
  opts.Fraction = f
  model, err := fastlowess.NewLowess(opts)
  if err != nil {
   log.Fatal(err)
  }
  result, err := model.Fit(t, y)
  model.Close()
  if err != nil {
   log.Fatal(err)
  }
  fmt.Printf("fraction=%v: y[0] = %v\n", f, result.Y[0])
 }
}
```

```output
fraction=0.05: y[0] = 0.13171195982828227
fraction=0.2: y[0] = 0.22444659184961097
fraction=0.5: y[0] = 0.33437036041791557
```

---

## Gene Expression Time Course

Biological application:

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func main() {
 n := 49
 hours := make([]float64, n)
 expression := make([]float64, n)
 for i := 0; i < n; i++ {
  hours[i] = float64(i) * 0.5
  expression[i] = 100.0*(1.0+0.5*math.Sin(hours[i]*math.Pi/12.0)) + (math.Mod(float64(i*7+3), 1.7)-0.85)*10.0
 }

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.3
 opts.Iterations = 3
 opts.ReturnDiagnostics = true

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(hours, expression)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Printf("R2: %.3f\n", result.Diagnostics.RSquared)
}
```

```output
R2: 0.973
```

---

## Choosing Fraction for Time Series

| Data Type | Recommended Fraction | Rationale |
| --- | --- | --- |
| Daily data (years) | 0.3–0.5 | Capture annual trends |
| Hourly data (days) | 0.1–0.2 | Capture daily patterns |
| Sensor data (minutes) | 0.05–0.1 | Preserve short-term features |
| Noisy data | Higher | Reduce noise impact |
| Clean data | Lower | Preserve detail |

---

## See Also

- [Real-Time Processing](use-case-real-time.md) — For streaming time series
- [Cross-Validation](../guide/cross-validation.md) — Optimal fraction selection
- [Boundary Handling](../advanced/boundary.md) — Edge bias in trend extraction
- [API Reference](../api/api.md) — Full parameter reference
