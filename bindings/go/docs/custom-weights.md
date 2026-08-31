# Custom Weights

Per-observation weights that encode data quality directly into the LOWESS fit.

## How Custom Weights Work

Standard LOWESS assigns equal prior trust to all observations. Custom weights let you override this assumption point by point — before any distance or robustness weighting is applied.

The effective weight of observation $j$ in a local fit centred at $x_i$ is:

$$w_{ij} = \text{customWeights}_j \times K\!\left(\frac{d_{ij}}{h_i}\right) \times r_j$$

where $K$ is the distance kernel, $h_i$ is the local bandwidth, and $r_j$ is the robustness weight from the current iteration.

> **Batch adapter only:** `customWeights` applies in **Batch** (`Lowess`) mode. It is silently ignored in `StreamingLowess` and `OnlineLowess`.

---

## When to Use Custom Weights

| Situation | Recommended weight |
| --- | --- |
| Point known to be erroneous | `0.0` — fully excluded |
| Unreliable sensor / low precision | `0.1 – 0.5` |
| Standard observation | `1.0` (default) |
| Carefully calibrated measurement | `> 1.0` |
| Measurement uncertainty $\sigma_i$ | $1 / \sigma_i^2$ |

### Custom Weights vs. Robustness Iterations

Both mechanisms handle unreliable data, but they serve different purposes:

| | Custom Weights | Robustness Iterations |
| --- | --- | --- |
| **When known** | Before fitting | Computed from residuals |
| **Knowledge required** | Prior knowledge of quality | None — data-driven |
| **Effect** | Fixed throughout fit | Adapts each iteration |
| **Use case** | Known bad sensors, calibration | Unknown outlier contamination |

They compose: you can use both simultaneously. Custom weights suppress *a priori* bad points; robustness iterations then handle any *residual* outliers that remain.

---

## Basic Usage

### Suppress a Known Outlier

Set the weight to `0` at the bad point — it is excluded from every local fit that would otherwise include it.

```go
package main

import (
 "fmt"
 "log"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func main() {
 x := make([]float64, 10)
 y := make([]float64, 10)
 for i := range x {
  x[i] = float64(i)
  y[i] = float64(i) * 2.0
 }
 y[5] = 100.0 // spike

 weights := make([]float64, 10)
 for i := range weights {
  weights[i] = 1.0
 }
 weights[5] = 0.0 // exclude the spike

 opts := fastlowess.DefaultOptions()
 opts.Fraction = 0.5
 opts.Iterations = 0

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y, weights)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (custom weights):", result.Y[0])
}
```

```output
First smoothed value (custom weights): 0.5726210350584308
```

---

### Emphasize Important Points

Assign high weights to measurements you trust most — calibration standards, reference instruments, or low-noise observations.

```go
weights := make([]float64, len(x))
for i := range weights {
 weights[i] = 1.0
}
for _, i := range []int{5, 20, 40, 60, 80} {
 weights[i] = 10.0 // trust calibration 10x more
}

opts := fastlowess.DefaultOptions()
opts.Fraction = 0.5
model, err := fastlowess.NewLowess(opts)
// ...
result, err := model.Fit(x, y, weights)
```

```output
First smoothed value (custom weights): 0.3237419725380614
```

---

### Propagate Measurement Uncertainty

If each observation has a known standard deviation $\sigma_i$, set $w_i = 1 / \sigma_i^2$ to give the fit information-theoretically optimal weighting.

```go
sigma := make([]float64, len(x))
weights := make([]float64, len(x))
for i := range sigma {
 sigma[i] = 0.1 + float64(i%4)*0.1
 weights[i] = 1.0 / (sigma[i] * sigma[i])
}

opts := fastlowess.DefaultOptions()
opts.Fraction = 0.5
model, err := fastlowess.NewLowess(opts)
// ...
result, err := model.Fit(x, y, weights)
```

```output
First smoothed value (custom weights): 0.1522219803222457
```

---

## Combined with Robustness Iterations

Custom weights and robustness iterations compose naturally: use custom weights for *known* bad points and robustness for *unknown* contamination.

```go
x := make([]float64, 20)
y := make([]float64, 20)
for i := range x {
 x[i] = float64(i)
 y[i] = float64(i) * 1.5
}
y[3] = -50.0 // known bad
y[12] = 80.0 // unknown outlier

weights := make([]float64, 20)
for i := range weights {
 weights[i] = 1.0
}
weights[3] = 0.0

opts := fastlowess.DefaultOptions()
opts.Fraction = 0.4
opts.Iterations = 3
model, err := fastlowess.NewLowess(opts)
// ...
result, err := model.Fit(x, y, weights)
```

```output
First smoothed value (custom weights): 1.7983966900445223
```

---

## Validation Rules

| Rule | Effect |
| --- | --- |
| Length must equal `n` | Error at `Fit` time if mismatched |
| All values must be ≥ 0 | Negative weights are rejected |
| All-zero weight vector | Error: no points remain for any local fit |
| Uniform weights (`1.0` everywhere) | Identical result to omitting weights |

> **Zero-weight windows:** If a local neighbourhood contains only zero-weight points, the fit at that centre point falls back to the behaviour specified by `ZeroWeightFallback` (default: `"use_local_mean"`).

---

## See Also

- [Robustness](robustness.md) — adaptive outlier downweighting via IRLS
- [API Reference](api.md) — full parameter reference
