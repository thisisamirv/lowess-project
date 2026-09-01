# Time Series Analysis

LOWESS for trend extraction and temporal smoothing.

## Overview

Time series data often contains noise, seasonality, and trends. LOWESS provides flexible trend extraction without parametric assumptions.

---

## Basic Trend Extraction

`fraction = 0.1` sizes the neighbourhood as 10% of the data at each evaluation point — narrow enough to follow a slowly varying trend without smearing periodic variation. Three robustness `iterations` down-weight noise spikes so they cannot bias the fitted curve; this is especially important when the signal-to-noise ratio is low or when occasional outliers are expected.

```@example use-case-time-series
using FastLOWESS

n = 500
t = collect(range(0, 100, length=n))
y = [10.0 + 0.5 * t[i] + 3.0 * sin(t[i] / 10.0) + (mod((i - 1) * 7 + 3, 1.7) - 0.85) * 3.0 for i in 1:n]

# Extract trend
model = Lowess(; fraction=0.1, iterations=3)
result = fit(model, t, y)

println("y[0]: ", result.y[1])
```

---

## Detrending

Remove trend to analyze residual patterns.

Setting `return_residuals = True` stores `observed − smoothed` alongside the smooth. A slightly wider `fraction = 0.3` produces a smoother baseline trend, so short-duration oscillations end up in the residuals rather than being absorbed into the trend component. The residual series is then ready for spectral analysis, seasonality detection, or change-point methods.

```@example use-case-time-series
using FastLOWESS

n = 100
t = collect(range(0, 2π, length=n))
y = sin.(t) .+ 0.1

# Smooth to get trend and residuals
model = Lowess(; fraction=0.3, iterations=3, return_residuals=true)
result = fit(model, t, y)

trend = result.y
detrended = result.residuals

println("residuals[0]: ", detrended[1])
```

---

## Forecasting with Prediction Intervals

Prediction intervals widen the uncertainty band to include both the uncertainty in the fitted curve (confidence interval) and the expected scatter of new observations around it. `fraction = 0.2` offers a balance between local detail and stable interval width — too small a fraction produces jagged interval edges; too large a fraction underestimates local variance near turning points.

```@example use-case-time-series
using FastLOWESS

n = 100
t = collect(range(0, 2π, length=n))
y = sin.(t) .+ 0.1

model = Lowess(;
    fraction=0.2,
    iterations=3,
    confidence_intervals=0.95,
    prediction_intervals=0.95
)
result = fit(model, t, y)

# Intervals are available in result.prediction_lower/upper
println("95% PI: [$(result.prediction_lower[1]), $(result.prediction_upper[1])]")
```

---

## Handling Missing Data

LOWESS naturally handles irregular time sampling:

```@example use-case-time-series
using FastLOWESS

t_irregular = [i * 1.0 + mod(i * 31, 10) * 0.1 for i in 0:99]
y_irregular = [10.0 + 0.3 * t + 2.0 * sin(t * 0.1) for t in t_irregular]

# LOWESS handles this seamlessly
model = Lowess(; fraction=0.2)
result = fit(model, t_irregular, y_irregular)
println("y[0]: ", result.y[1])
```

---

## Multi-Scale Analysis

Use different fractions to extract features at different scales:

```@example use-case-time-series
using FastLOWESS

n = 100
t = collect(range(0, 2π, length=n))
y = sin.(t) .+ 0.1

fractions = [0.05, 0.2, 0.5]

results = map(fractions) do f
    model = Lowess(; fraction=f)
    result = fit(model, t, y)
end
for (f, res) in zip(fractions, results)
    println("fraction=$(f): y[0] = $(round(res.y[1]; digits=4))")
end
```

---

## Gene Expression Time Course

Biological application:

```@example use-case-time-series
using FastLOWESS

hours = collect(0.0:0.5:24.0)[1:49]
expression = [100.0 * (1.0 + 0.5 * sin(hours[i] * pi / 12.0)) + (mod((i - 1) * 7 + 3, 1.7) - 0.85) * 10.0 for i in 1:49]

model = Lowess(;
    fraction=0.3,
    iterations=3,
    confidence_intervals=0.95,
    return_diagnostics=true
)
result = fit(model, hours, expression)

println("R2: ", result.diagnostics.r_squared)
```

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
- [API Reference](../api.md) — Full parameter reference
