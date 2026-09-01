---
title: "Robustness"
weight: 45
---

Outlier handling through iterative reweighting.

## How Robustness Works

Standard LOWESS can be biased by outliers. Robustness iterations downweight points with large residuals:

1. Fit initial LOWESS
2. Compute residuals
3. Assign robustness weights (large residuals → low weight)
4. Refit using combined distance × robustness weights
5. Repeat steps 2–4

![Robustness Methods](../assets/diagrams/robust_method_comparison.svg)

![Robustness Iterations](../assets/diagrams/robust_iter_comparison.svg)

---

## Robustness Methods

### Bisquare (Default)

Smooth downweighting. Points transition gradually from full weight to zero.

$$w(u) = \begin{cases} (1 - u^2)^2 & |u| < 1 \\ 0 & |u| \geq 1 \end{cases}$$

**Use when**: General purpose, balanced approach.

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
 opts.Iterations = 3
 opts.RobustnessMethod = "bisquare"
 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (bisquare robustness):", result.Y[0])
}
```

```output
First smoothed value (bisquare robustness): 0.38260776436644134
```

---

### Huber

Linear penalty beyond threshold. Less aggressive than Bisquare.

$$w(u) = \begin{cases} 1 & |u| \leq k \\ k/|u| & |u| > k \end{cases}$$

**Use when**: Moderate outliers, want to retain some influence.

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
 opts.Iterations = 3
 opts.RobustnessMethod = "huber"
 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (huber robustness):", result.Y[0])
}
```

```output
First smoothed value (huber robustness): 0.38170495877281807
```

---

### Talwar

Hard threshold. Points are either fully weighted or completely excluded.

$$w(u) = \begin{cases} 1 & |u| \leq k \\ 0 & |u| > k \end{cases}$$

**Use when**: Extreme outliers, want binary exclusion.

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
 opts.Iterations = 3
 opts.RobustnessMethod = "talwar"
 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (talwar robustness):", result.Y[0])
}
```

```output
First smoothed value (talwar robustness): 0.37877937542419027
```

---

## Comparison

| Method | Transition | Aggressiveness | Use Case |
| --- | --- | --- | --- |
| **Bisquare** | Smooth | Moderate | General purpose |
| **Huber** | Gradual | Mild | Preserve influence |
| **Talwar** | Hard | Strong | Extreme contamination |

---

## Detecting Outliers

Use robustness weights to identify potential outliers:

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
 for _, i := range []int{20, 50, 80} {
  y[i] += 5.0 // inject outliers
 }

 opts := fastlowess.DefaultOptions()
 opts.Iterations = 5
 opts.ReturnRobustnessWeights = true
 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }

 count := 0
 for i, w := range result.RobustnessWeights {
  if w < 0.5 {
   if count < 5 {
    fmt.Printf("Potential outlier at index %d: weight = %.3f\n", i, w)
   }
   count++
  }
 }
}
```

```output
Potential outlier at index 20: weight = 0.000
Potential outlier at index 50: weight = 0.000
Potential outlier at index 80: weight = 0.000
```

---

## Scale Estimation

Residuals are scaled before computing robustness weights. Two methods:

| Method | Formula | Robustness |
| --- | --- | --- |
| **MAD** | `median(\|r − median(r)\|)` | Very robust (default) |
| **MAR** | `median(\|r\|)` | Robust, uncentered |
| **Mean** | `mean(\|r\|)` | Less robust, fastest |

![Scaling Methods Comparison](../assets/diagrams/scaling_comparison.svg)

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
 opts.Iterations = 3
 opts.ScalingMethod = "mad" // Default
 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (mad scaling):", result.Y[0])
}
```

```output
First smoothed value (mad scaling): 0.38260776436644134
```

See [Scaling Methods](scaling.md) for a full breakdown.

---

## Auto-Convergence

Stop iterations early when weights stabilize:

> **Performance:** Auto-convergence can significantly reduce computation when weights stabilize before reaching max iterations.

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
 opts.Iterations = 10 // Maximum iterations
 converge := 1e-6      // Stop when change < 1e-6
 opts.AutoConverge = &converge
 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (auto-converge):", result.Y[0])
}
```

```output
First smoothed value (auto-converge): 0.3826022035717678
```
