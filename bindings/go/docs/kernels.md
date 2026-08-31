# Weight Functions

Kernel functions for distance weighting.

## Overview

Weight functions (kernels) determine how neighboring points contribute to each local fit. Points closer to the target receive higher weights.

![Weight Functions](../assets/diagrams/kernel_comparison.svg)

## Available Kernels

| Kernel | Efficiency | Smoothness | Support |
| --- | --- | --- | --- |
| **Tricube** | 0.998 | Very smooth | Compact |
| **Epanechnikov** | 1.000 | Smooth | Compact |
| **Gaussian** | 0.961 | Infinite | Unbounded |
| **Biweight** | 0.995 | Very smooth | Compact |
| **Cosine** | 0.999 | Smooth | Compact |
| **Triangle** | 0.989 | Moderate | Compact |
| **Uniform** | 0.943 | None | Compact |

**Efficiency** = AMISE relative to Epanechnikov (1.0 = optimal)

---

## Tricube (Default)

Cleveland's original choice. Best all-around performance.

$$w(u) = (1 - |u|^3)^3$$

**Use when**: Default choice for most applications.

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
 opts.WeightFunction = "tricube"
 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (tricube kernel):", result.Y[0])
}
```

```output
First smoothed value (tricube kernel): 0.38260776436644134
```

---

## Epanechnikov

Theoretically optimal for kernel density estimation.

$$w(u) = \frac{3}{4}(1 - u^2)$$

**Use when**: Optimal MSE properties desired.

```go
opts.WeightFunction = "epanechnikov"
```

```output
First smoothed value (epanechnikov kernel): 0.40672777844316566
```

---

## Gaussian

Infinitely smooth. No boundary effects.

$$w(u) = \exp(-u^2/2)$$

**Use when**: Maximum smoothness needed, computational cost acceptable.

```go
opts.WeightFunction = "gaussian"
```

```output
First smoothed value (gaussian kernel): 0.43576701891170694
```

---

## Biweight

Good balance of efficiency and smoothness.

$$w(u) = (1 - u^2)^2$$

**Use when**: Alternative to Tricube with slightly different properties.

```go
opts.WeightFunction = "biweight"
```

```output
First smoothed value (biweight kernel): 0.37590304570290123
```

---

## Cosine

Smooth and computationally efficient.

$$w(u) = \cos(\pi u / 2)$$

**Use when**: Want smooth kernel with simple form.

```go
opts.WeightFunction = "cosine"
```

```output
First smoothed value (cosine kernel): 0.40082995540195804
```

---

## Triangle

Simple linear taper.

$$w(u) = 1 - |u|$$

**Use when**: Simple, interpretable weights.

```go
opts.WeightFunction = "triangle"
```

```output
First smoothed value (triangle kernel): 0.3816572331939024
```

---

## Uniform

Equal weights within window. Fastest but least smooth.

$$w(u) = 1$$

**Use when**: Speed is critical, smoothness less important.

```go
opts.WeightFunction = "uniform"
```

```output
First smoothed value (uniform kernel): 0.45083991221911446
```

---

## Choosing a Kernel

Choose the first row below whose condition applies:

| Condition | Kernel |
| --- | --- |
| Need maximum smoothness | `"gaussian"` |
| Default is acceptable | `"tricube"` |
| Need optimal asymptotic MSE | `"epanechnikov"` |
| Speed is critical | `"uniform"` |
| None of the above | `"biweight"` |

> **Recommendation:** Stick with **Tricube** (default) unless you have specific requirements. The differences between kernels are usually small in practice.
