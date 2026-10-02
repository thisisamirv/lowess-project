# Intervals

Confidence and prediction intervals for uncertainty quantification.

## Overview

![Confidence and Prediction Intervals](../assets/diagrams/intervals_comparison.svg)

:::{note} Adapter support
Confidence and prediction intervals are available in **Batch** mode, **Streaming** mode (computed per chunk and merged across overlap boundaries via `merge_strategy`, like `y`/`derivative`), and **Online** mode when `update_mode="full"` is set (raises if combined with the default `"incremental"` mode).
:::

Confidence and prediction coverage levels are independent; for example, a 90% confidence interval can be paired with a 99% prediction interval.

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

model = fl.Lowess(fraction=0.5, intervals={"confidence": 0.95})
result = model.fit(x, y)

print("Smoothed (first 5):", result.y[:5])
print("CI Lower (first 5):", result.confidence_lower[:5])
print("CI Upper (first 5):", result.confidence_upper[:5])
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

model = fl.Lowess(fraction=0.5, intervals={"prediction": 0.95})
result = model.fit(x, y)

print("PI Lower (first 5):", result.prediction_lower[:5])
print("PI Upper (first 5):", result.prediction_upper[:5])
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
    intervals={"confidence": 0.95, "prediction": 0.95},
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

### 99% confidence interval (wider)

model = fl.Lowess(intervals={"confidence": 0.99})
result = model.fit(x, y)
print(f"99% CI at midpoint: [{result.confidence_lower[50]:.4f}, {result.confidence_upper[50]:.4f}]")
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

model = fl.Lowess(intervals={"confidence": 0.95})
result = model.fit(x, y)
print("Standard errors (first 5):", result.standard_errors[:5])
:::

---

## Residual Bootstrap

Set `bootstrap` to at least `2` to replace analytic uncertainty with residual-bootstrap refits. Batch shares its outer `seed` with CV; Streaming restarts the seed per combined chunk, and Online per full-update window. Online bootstrap requires `update_mode="full"`.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

x = np.arange(30) *0.1
y = np.sin(x) + 0.1* np.cos(7 * x)
intervals = {"confidence": 0.95, "prediction": 0.95, "bootstrap": 20}

batch = fl.Lowess(intervals=intervals, seed=42).fit(x, y)
print("Batch SEs:", len(batch.standard_errors))

stream = fl.StreamingLowess(chunk_size=len(x), intervals=intervals, seed=42)
print("Streaming CI present:", stream.process_chunk(x, y).confidence_lower is not None)

online = fl.OnlineLowess(min_points=5, update_mode="full", intervals=intervals, seed=42)
last = None
for xi, yi in zip(x[:12], y[:12]):
    last = online.add_point(xi, yi) or last
print("Online PI present:", last.prediction_lower is not None)
:::

---

## Availability

:::{note} Supported In All Three Adapters
Confidence and prediction intervals are available in **Batch**, **Streaming**, and **Online** mode (`update_mode="full"` only).
:::

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✓ | ✓ (`update_mode="full"` only) |
| Prediction intervals | ✓ | ✓ | ✓ (`update_mode="full"` only) |
| Standard errors | ✓ | ✓ | ✓ (`update_mode="full"` only) |
| Residual bootstrap | ✓ | ✓ | ✓ (`update_mode="full"` only) |
