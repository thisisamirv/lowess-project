<!-- markdownlint-disable MD024 MD033 MD046 -->
# Cross-Validation

Automated parameter selection via cross-validation.

## Overview

Cross-validation helps select optimal parameters (especially `fraction`) by evaluating performance on held-out data.

![Cross-Validation](https://raw.githubusercontent.com/thisisamirv/lowess-project/main/crates/lowess/assets/diagrams/cv_comparison.svg)

---

## K-Fold Cross-Validation

Split data into K folds, train on K-1, validate on 1.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();


    let model = Lowess::new()
        .cv_method("kfold")
        .cv_k(5)
        .cv_fractions(vec![0.2, 0.3, 0.5, 0.7])
        .build()?;

    let result = model.fit(&x, &y)?;

    // The best fraction was automatically selected
    println!("Selected fraction: {}", result.fraction_used);

    if let Some(scores) = &result.cv_scores {
        println!("CV scores: {:?}", scores);
    }

    Ok(())
}
```

```output
Selected fraction: 0.2
CV scores: [0.266006592543415, 0.26663733476135043, 0.36243048571062464, 0.4466813477111355]
```

---

## Leave-One-Out (LOOCV)

Each point is held out once. Most thorough but slowest.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .cv_method("loocv")
        .cv_fractions(vec![0.2, 0.3, 0.5, 0.7])
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("Selected fraction (CV): {}", result.fraction_used);
    Ok(())
}
```

```output
Selected fraction (CV): 0.2
```

---

## Seeded Randomization

Set a seed for reproducible fold assignments:

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .cv_method("kfold")
        .cv_k(5)
        .cv_fractions(vec![0.3, 0.5, 0.7])
        .cv_seed(42)
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("Selected fraction (CV): {}", result.fraction_used);
    Ok(())
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

!!! tip "Recommendation"
    Use **5-fold** or **10-fold** CV for most applications. LOOCV is only worth it for small datasets (N < 100).

---

## CV Metrics

Cross-validation uses MSE (Mean Squared Error) by default:

```text
MSE = mean((y_true - y_pred)^2)
```

Lower MSE indicates better fit on held-out data.

---

## Interpreting Results

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    // Example output
    let model = Lowess::new()
        .cv_method("kfold")
        .cv_k(5)
        .cv_fractions(vec![0.1, 0.3, 0.5, 0.7])
        .build()?;

    let result = model.fit(&x, &y)?;

    // Fraction  | CV Score (MSE)
    // 0.1       | 0.0542  ← Undersmoothed
    // 0.3       | 0.0231  ← Best
    // 0.5       | 0.0298
    // 0.7       | 0.0412  ← Oversmoothed

    println!("Selected fraction (CV): {}", result.fraction_used);
    Ok(())
}
```

```output
Selected fraction (CV): 0.3
```

The fraction with **lowest CV score** is automatically selected.

---

## Availability

!!! warning "Batch Mode Only"
    Cross-validation is only available in **Batch** mode.

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
