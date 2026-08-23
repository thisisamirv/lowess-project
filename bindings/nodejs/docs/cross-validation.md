<!-- markdownlint-disable MD024 MD033 MD046 -->
# Cross-Validation

Automated parameter selection via cross-validation.

## Overview

Cross-validation helps select optimal parameters (especially `fraction`) by evaluating performance on held-out data.

![Cross-Validation](../assets/diagrams/cv_comparison.svg)

---

## K-Fold Cross-Validation

Split data into K folds, train on K-1, validate on 1.

```javascript
const { Lowess } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const model = new Lowess({
    cv_method: "kfold",
    cv_k: 5,
    cv_fractions: [0.2, 0.3, 0.5, 0.7]
});
const result = model.fit(x, y);

console.log("Selected fraction:", result.fraction_used);
console.log("CV scores:", result.cv_scores);
```

---

## Leave-One-Out (LOOCV)

Each point is held out once. Most thorough but slowest.

```javascript
const { Lowess } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const model = new Lowess({
    cv_method: "loocv",
    cv_fractions: [0.2, 0.3, 0.5, 0.7]
});
const result = model.fit(x, y);
```

---

## Seeded Randomization

Set a seed for reproducible fold assignments:

```javascript
const fl = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);

const model = new fl.Lowess({
    cv_method: "kfold",
    cv_k: 5,
    cv_fractions: [0.3, 0.5, 0.7],
    cv_seed: 42
});
const result = model.fit(x, y);
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
MSE = mean((y_true - y_pred)²)
```

Lower MSE indicates better fit on held-out data.

---

## Interpreting Results

```javascript
const { Lowess } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

// Example output
const model = new Lowess({
    cv_method: "kfold",
    cv_k: 5,
    cv_fractions: [0.1, 0.3, 0.5, 0.7]
});
const result = model.fit(x, y);

// Fraction  | CV Score (MSE)
// 0.1       | 0.0542  ← Undersmoothed
// 0.3       | 0.0231  ← Best
// 0.5       | 0.0298
// 0.7       | 0.0412  ← Oversmoothed
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
