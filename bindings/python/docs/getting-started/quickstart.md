# Quick Start

Get up and running with LOWESS in minutes.

## Basic Smoothing

Smooth a noisy sine wave — the kind of signal where LOWESS shines. Each example recovers the underlying trend from 100 points of Gaussian noise.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

## 100-point noisy sine wave

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

model = fl.Lowess(fraction=0.3, iterations=3)
result = model.fit(x, y)

print(f"First smoothed value: {result.y[0]:.4f}  (true: {np.sin(x[0]):.4f})")
:::

---

## With Confidence Intervals

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

model = fl.Lowess(
    fraction=0.5,
    iterations=3,
    confidence_intervals=0.95,
    prediction_intervals=0.95,
    return_diagnostics=True
)
result = model.fit(x, y)

print("Smoothed:", result.y)
print("CI Lower:", result.confidence_lower)
print("CI Upper:", result.confidence_upper)
print("R²:", result.diagnostics.r_squared)
:::

---

## Handling Outliers

LOWESS can robustly handle outliers through iterative reweighting:

:::{jupyter-execute}
import fastlowess as fl
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
t = np.linspace(0, 100, 500)
trend_true = 10 + 0.5 *t + 3* np.sin(t / 10)
y = trend_true + np.random.normal(0, 3, len(t))

x_out = np.linspace(1, 6, 6)
y_with_outlier = np.array([2.0, 4.0, 6.0, 50.0, 10.0, 12.0])

model = fl.Lowess(
    fraction=0.5,
    iterations=5,
    robustness_method="bisquare",
    return_robustness_weights=True
)
result = model.fit(x_out, y_with_outlier)

## Check which points were downweighted

for i, w in enumerate(result.robustness_weights):
    if w < 0.5:
        print(f"Point {i} is likely an outlier (weight: {w:.3f})")
:::

---

## Streaming Mode

For datasets too large to fit in memory, stream them in fixed-size chunks with overlap.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 10 *np.pi, 5000)
y = np.sin(x / np.pi)* np.exp(-x / 30) + rng.normal(0, 0.15, 5000)

model = fl.StreamingLowess(
    fraction=0.2,
    chunk_size=1000,
    overlap=100,
    merge_strategy="weighted_average",
)

chunk_size = 1000
for start in range(0, 4001, chunk_size):
    end = min(start + chunk_size, len(x))
    model.process_chunk(x[start:end], y[start:end])
result = model.finalize()
print(f"Smoothed {len(result.y)} points in streaming mode")
:::

---

## Next Steps

| Topic | Link |
| --- | --- |
| How LOWESS works | [Concepts](concepts.md) |
| All parameters explained | [Parameters](../user-guide/parameters.md) |
| Batch vs Streaming vs Online | [Execution Modes](../user-guide/adapters.md) |
| Edge handling | [Boundary](../user-guide/boundary.md) |
| Outlier handling in depth | [Robustness](../user-guide/robustness.md) |
| Full API per language | [API Reference](../api/index.md) |
