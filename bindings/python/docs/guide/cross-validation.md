# Cross-Validation

Automated parameter selection via cross-validation.

## Overview

Cross-validation helps select optimal parameters (especially `fraction`) by evaluating performance on held-out data.

![Cross-Validation](../assets/diagrams/cv_comparison.svg)

---

## K-Fold Cross-Validation

Split data into K folds, train on K-1, validate on 1.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

model = fl.Lowess(cv_method="kfold",
    cv_k=5,
    cv_fractions=[0.2, 0.3, 0.5, 0.7]
)
result = model.fit(x, y)

print(f"Selected fraction: {result.fraction_used}")
print(f"CV scores: {result.cv_scores}")
:::

---

## Leave-One-Out (LOOCV)

Each point is held out once. Most thorough but slowest.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

model = fl.Lowess(cv_method="loocv",
    cv_fractions=[0.2, 0.3, 0.5, 0.7]
)
result = model.fit(x, y)
print(f"Selected fraction (CV): {result.fraction_used}")
:::

---

## Seeded Randomization

Set a seed for reproducible fold assignments:

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

model = fl.Lowess(cv_method="kfold",
    cv_k=5,
    cv_fractions=[0.3, 0.5, 0.7],
    cv_seed=42
)
result = model.fit(x, y)
print(f"Selected fraction (CV): {result.fraction_used}")
:::

---

## Comparison

| Method | Folds | Speed | Variance | Bias |
| --- | --- | --- | --- | --- |
| **KFold(5)** | 5 | Fast | Moderate | Low |
| **KFold(10)** | 10 | Medium | Lower | Lower |
| **LOOCV** | N | Slow | Lowest | Lowest |

:::{tip} Recommendation
Use **5-fold** or **10-fold** CV for most applications. LOOCV is only worth it for small datasets (N < 100).
:::

---

## CV Metrics

Cross-validation uses MSE (Mean Squared Error) by default:

```text
MSE = mean((y_true - y_pred)^2)
```

Lower MSE indicates better fit on held-out data.

---

## Interpreting Results

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

model = fl.Lowess(cv_method="kfold", cv_k=5,
                   cv_fractions=[0.1, 0.3, 0.5, 0.7])
result = model.fit(x, y)

print(f"Selected fraction (CV): {result.fraction_used}")
:::

The fraction with **lowest CV score** is automatically selected.

---

## Availability

:::{warning} Batch Mode Only
Cross-validation is only available in **Batch** mode.
:::

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
