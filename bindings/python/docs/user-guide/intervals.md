# Intervals

Confidence and prediction intervals for uncertainty quantification.

## Overview

![Confidence and Prediction Intervals](../assets/diagrams/intervals_comparison.svg)

:::{note} Adapter support
Confidence and prediction intervals are available in **Batch** mode only. Streaming and Online modes do not support intervals.
:::

| Type | Represents | Width | Use |
| --- | --- | --- | --- |
| **Confidence** | Uncertainty in mean curve | Narrow | Where is the true trend? |
| **Prediction** | Uncertainty for new points | Wide | Where will new data fall? |

---

## Confidence Intervals

Estimate uncertainty in the smoothed curve itself.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

model = fl.Lowess(fraction=0.5, confidence_intervals=0.95)
result = model.fit(x, y)

print("Smoothed:", result.y)
print("CI Lower:", result.confidence_lower)
print("CI Upper:", result.confidence_upper)
:::

---

## Prediction Intervals

Estimate where new observations might fall.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

model = fl.Lowess(fraction=0.5, prediction_intervals=0.95)
result = model.fit(x, y)

print("PI Lower:", result.prediction_lower)
print("PI Upper:", result.prediction_upper)
:::

---

## Both Intervals

Request both types simultaneously:

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

model = fl.Lowess(
    fraction=0.5,
    confidence_intervals=0.95,
    prediction_intervals=0.95
)
result = model.fit(x, y)
print(f"95% CI at midpoint: [{result.confidence_lower[50]:.4f}, {result.confidence_upper[50]:.4f}]")
:::

---

## Confidence Levels

Common levels and their z-values:

| Level | z-value | Interpretation |
| --- | --- | --- |
| 0.90 | 1.645 | 90% of intervals contain true value |
| 0.95 | 1.960 | 95% of intervals contain true value |
| 0.99 | 2.576 | 99% of intervals contain true value |

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

### 90% confidence interval (narrower)

model = fl.Lowess(confidence_intervals=0.90)
result = model.fit(x, y)
print(f"95% CI at midpoint: [{result.confidence_lower[50]:.4f}, {result.confidence_upper[50]:.4f}]")
:::

---

## Standard Errors

Access standard errors directly (available when intervals are computed):

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

model = fl.Lowess(confidence_intervals=0.95)
result = model.fit(x, y)
print("Standard errors:", result.standard_errors)
:::

---

## Availability

:::{warning} Batch Mode Only
Confidence and prediction intervals are only available in **Batch** mode. Streaming and Online modes do not support intervals.
:::

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✗ | ✗ |
| Prediction intervals | ✓ | ✗ | ✗ |
| Standard errors | ✓ | ✗ | ✗ |
