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

## Deterministic trend + seasonal time series

n = 500
t = np.linspace(0, 100, n)
i = np.arange(n)
y = 10 + 0.5 *t + 3* np.sin(t / 10) + (np.mod(i *7 + 3, 1.7) - 0.85)* 3

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

print(f"y[0]: {result.y[0]:.4f}")
:::

---

## Detrending

Remove trend to analyze residual patterns.

Setting `return_residuals = True` stores `observed − smoothed` alongside the smooth. A slightly wider `fraction = 0.3` produces a smoother baseline trend, so short-duration oscillations end up in the residuals rather than being absorbed into the trend component. The residual series is then ready for spectral analysis, seasonality detection, or change-point methods.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np
import matplotlib.pyplot as plt

n = 100
t = np.linspace(0, 2 * np.pi, n)
y = np.sin(t) + 0.1

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

print(f"residuals[0]: {detrended[0]:.4f}")
:::

---

## Forecasting with Prediction Intervals

Prediction intervals widen the uncertainty band to include both the uncertainty in the fitted curve (confidence interval) and the expected scatter of new observations around it. `fraction = 0.2` offers a balance between local detail and stable interval width — too small a fraction produces jagged interval edges; too large a fraction underestimates local variance near turning points.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np
import matplotlib.pyplot as plt

n = 100
t = np.linspace(0, 2 * np.pi, n)
y = np.sin(t) + 0.1

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

print(f"95% PI: [{result.prediction_lower[0]:.4f}, {result.prediction_upper[0]:.4f}]")
:::

---

## Handling Missing Data

LOWESS naturally handles irregular time sampling:

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

t_irregular = np.array([i *1.0 + (i* 31 % 10) *0.1 for i in range(100)])
y_irregular = 10 + t_irregular* 0.3 + 2.0 *np.sin(t_irregular* 0.1)

## LOWESS handles this seamlessly

model = fl.Lowess(fraction=0.2)
result = model.fit(t_irregular, y_irregular)
print(f"y[0]: {result.y[0]:.4f}")
:::

---

## Multi-Scale Analysis

Use different fractions to extract features at different scales:

:::{jupyter-execute}
import fastlowess as fl
import numpy as np
import matplotlib.pyplot as plt

n = 100
t = np.linspace(0, 2 * np.pi, n)
y = np.sin(t) + 0.1

## Multiple smoothing scales

fractions = [0.05, 0.2, 0.5]

plt.figure(figsize=(12, 5))
plt.plot(t, y, "gray", alpha=0.3, label="Data")

for f in fractions:
    model = fl.Lowess(fraction=f)
    result = model.fit(t, y)
    plt.plot(t, result.y, label=f"fraction={f}")
    print(f"fraction={f}: y[0] = {result.y[0]:.4f}")

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
i = np.arange(len(hours))
expression = 100 *(1 + 0.5* np.sin(hours *np.pi / 12)) + (np.mod(i* 7 + 3, 1.7) - 0.85) * 10

model = fl.Lowess(
    fraction=0.3,
    iterations=3,
    confidence_intervals=0.95,
    return_diagnostics=True
)
result = model.fit(hours, expression)

print(f"R2: {result.diagnostics.r_squared:.3f}")
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

- [Real-Time Processing](use-case-real-time.md) — For streaming time series
- [Cross-Validation](../guide/cross-validation.md) — Optimal fraction selection
- [Boundary Handling](../advanced/boundary.md) — Edge bias in trend extraction
- [API Reference](../api/api.md) — Full parameter reference
