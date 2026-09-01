---
title: "Cross-Validation"
weight: 65
---

Automated parameter selection via cross-validation.

## Overview

Cross-validation helps select optimal parameters (especially `Fraction`) by evaluating performance on held-out data.

![Cross-Validation](../assets/diagrams/cv_comparison.svg)

---

## K-Fold Cross-Validation

Split data into K folds, train on K-1, validate on 1.

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
 opts.CVMethod = "kfold"
 opts.CVK = 5
 opts.CVFractions = []float64{0.2, 0.3, 0.5, 0.7}

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }

 // The best fraction was automatically selected
 fmt.Println("Selected fraction:", result.FractionUsed)
 fmt.Println("CV scores:", result.CVScores)
}
```

```output
Selected fraction: 0.2
CV scores: [0.2660065925434152 0.26663733476135054 0.36243048571062464 0.4466813477111355]
```

---

## Leave-One-Out (LOOCV)

Each point is held out once. Most thorough but slowest.

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
 opts.CVMethod = "loocv"
 opts.CVFractions = []float64{0.2, 0.3, 0.5, 0.7}

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("Selected fraction (CV):", result.FractionUsed)
}
```

```output
Selected fraction (CV): 0.2
```

---

## Seeded Randomization

Set a seed for reproducible fold assignments:

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
 opts.CVMethod = "kfold"
 opts.CVK = 5
 opts.CVFractions = []float64{0.3, 0.5, 0.7}
 seed := uint64(42)
 opts.CVSeed = &seed

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("Selected fraction (CV):", result.FractionUsed)
}
```

```output
Selected fraction (CV): 0.7
```

---

## Comparison

| Method | Folds | Speed | Variance | Bias |
| --- | --- | --- | --- | --- |
| **KFold(5)** | 5 | Fast | Moderate | Low |
| **KFold(10)** | 10 | Medium | Lower | Lower |
| **LOOCV** | N | Slow | Lowest | Lowest |

> **Recommendation:** Use **5-fold** or **10-fold** CV for most applications. LOOCV is only worth it for small datasets (N < 100).

---

## CV Metrics

Cross-validation uses MSE (Mean Squared Error) by default:

```text
MSE = mean((y_true - y_pred)²)
```

Lower MSE indicates better fit on held-out data.

---

## Interpreting Results

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
 opts.CVMethod = "kfold"
 opts.CVK = 5
 opts.CVFractions = []float64{0.1, 0.3, 0.5, 0.7}

 // Fraction  | CV Score (MSE)
 // 0.1       | 0.0542  <- Undersmoothed
 // 0.3       | 0.0231  <- Best
 // 0.5       | 0.0298
 // 0.7       | 0.0412  <- Oversmoothed

 model, err := fastlowess.NewLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("Selected fraction (CV):", result.FractionUsed)
}
```

```output
Selected fraction (CV): 0.3
```

The fraction with **lowest CV score** is automatically selected.

---

## Availability

> **Batch Mode Only:** Cross-validation is only available in **Batch** mode.

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| K-Fold CV | ✓ | ✗ | ✗ |
| LOOCV | ✓ | ✗ | ✗ |

---

## Best Practices

1. **Test a range**: Include fractions from 0.1 to 0.9
2. **Use enough folds**: 5-10 folds balance speed and accuracy
3. **Set a seed**: For reproducible results
4. **Check the curve**: CV optimizes MSE, but visual inspection matters
