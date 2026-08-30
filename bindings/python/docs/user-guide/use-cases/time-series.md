# Time Series Analysis

LOWESS for trend extraction and temporal smoothing.

## Overview

Time series data often contains noise, seasonality, and trends. LOWESS provides flexible trend extraction without parametric assumptions.

---

## Basic Trend Extraction

`fraction = 0.1` sizes the neighbourhood as 10% of the data at each evaluation point — narrow enough to follow a slowly varying trend without smearing periodic variation. Three robustness `iterations` down-weight noise spikes so they cannot bias the fitted curve; this is especially important when the signal-to-noise ratio is low or when occasional outliers are expected.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np
import matplotlib.pyplot as plt

## Simulate noisy time series with trend

np.random.seed(42)
t = np.linspace(0, 100, 500)
trend = 10 + 0.5 *t + 3* np.sin(t / 10)
noise = np.random.normal(0, 3, len(t))
y = trend + noise

## Extract trend with LOWESS

model = fl.Lowess(fraction=0.1, iterations=3)
result = model.fit(t, y)

## Plot

plt.figure(figsize=(12, 5))
plt.plot(t, y, "gray", alpha=0.5, label="Observed")
plt.plot(t, result.y, "b-", linewidth=2, label="Trend (LOWESS)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.title("Trend Extraction")
plt.show()
:::

---

## Detrending

Remove trend to analyze residual patterns.

Setting `return_residuals = True` stores `observed − smoothed` alongside the smooth. A slightly wider `fraction = 0.3` produces a smoother baseline trend, so short-duration oscillations end up in the residuals rather than being absorbed into the trend component. The residual series is then ready for spectral analysis, seasonality detection, or change-point methods.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
t = np.linspace(0, 100, 500)
trend_true = 10 + 0.5 *t + 3* np.sin(t / 10)
y = trend_true + np.random.normal(0, 3, len(t))

## Smooth to get trend

model = fl.Lowess(fraction=0.3, iterations=3, return_residuals=True)
result = model.fit(t, y)

trend = result.y
detrended = result.residuals

## Analyze residuals for seasonality, etc

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t, trend)
plt.title("Extracted Trend")

plt.subplot(1, 2, 2)
plt.plot(t, detrended)
plt.title("Detrended (Residuals)")
plt.tight_layout()
:::

---

## Forecasting with Prediction Intervals

Prediction intervals widen the uncertainty band to include both the uncertainty in the fitted curve (confidence interval) and the expected scatter of new observations around it. `fraction = 0.2` offers a balance between local detail and stable interval width — too small a fraction produces jagged interval edges; too large a fraction underestimates local variance near turning points.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
t = np.linspace(0, 100, 500)
trend_true = 10 + 0.5 *t + 3* np.sin(t / 10)
y = trend_true + np.random.normal(0, 3, len(t))

model = fl.Lowess(
    fraction=0.2,
    iterations=3,
    confidence_intervals=0.95,
    prediction_intervals=0.95
)
result = model.fit(t, y)

## Plot with uncertainty bands

plt.figure(figsize=(12, 5))
plt.plot(t, y, "gray", alpha=0.3)
plt.plot(t, result.y, "b-", linewidth=2, label="Trend")
plt.fill_between(
    t,
    result.prediction_lower,
    result.prediction_upper,
    alpha=0.2, color="blue", label="95% Prediction"
)
plt.legend()
:::

---

## Handling Missing Data

LOWESS naturally handles irregular time sampling:

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

## Irregular time points (gaps in data)

t_irregular = np.sort(np.random.uniform(0, 100, 200))
y_irregular = 10 + t_irregular * 0.3 + np.random.normal(0, 2, 200)

## LOWESS handles this seamlessly

model = fl.Lowess(fraction=0.2)
result = model.fit(t_irregular, y_irregular)
print(f"Smoothed y[0]: {result.y[0]:.4f}")
:::

---

## Multi-Scale Analysis

Use different fractions to extract features at different scales:

:::{jupyter-execute}
import fastlowess as fl
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
t = np.linspace(0, 100, 500)
trend_true = 10 + 0.5 *t + 3* np.sin(t / 10)
y = trend_true + np.random.normal(0, 3, len(t))

## Multiple smoothing scales

fractions = [0.05, 0.2, 0.5]

plt.figure(figsize=(12, 5))
plt.plot(t, y, "gray", alpha=0.3, label="Data")

for f in fractions:
    model = fl.Lowess(fraction=f)
    result = model.fit(t, y)
    plt.plot(t, result.y, label=f"fraction={f}")

plt.legend()
plt.title("Multi-Scale LOWESS")
:::

---

## Gene Expression Time Course

Biological application:

:::{jupyter-execute}
import numpy as np
import fastlowess as fl

## Gene expression over 24 hours

hours = np.arange(0, 24.5, 0.5)
expression = 100 *(1 + 0.5* np.sin(hours * np.pi / 12)) + np.random.normal(0, 10, len(hours))

model = fl.Lowess(
    fraction=0.3,
    iterations=3,
    confidence_intervals=0.95,
    return_diagnostics=True
)
result = model.fit(hours, expression)

print(f"R²: {result.diagnostics.r_squared:.3f}")
:::

---

## Choosing Fraction for Time Series

| Data Type | Recommended Fraction | Rationale |
| --- | --- | --- |
| Daily data (years) | 0.3–0.5 | Capture annual trends |
| Hourly data (days) | 0.1–0.2 | Capture daily patterns |
| Sensor data (minutes) | 0.05–0.1 | Preserve short-term features |
| Noisy data | Higher | Reduce noise impact |
| Clean data | Lower | Preserve detail |

---

## See Also

- [Real-Time Processing](real-time.md) — For streaming time series
- [Cross-Validation](../cross-validation.md) — Optimal fraction selection
- [Boundary Handling](../boundary.md) — Edge bias in trend extraction
- [API Reference](../../api/python.md) — Full parameter reference
