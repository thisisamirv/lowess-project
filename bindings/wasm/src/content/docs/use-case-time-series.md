---
title: Time Series Analysis
---
<!-- markdownlint-disable MD024 MD046 MD033 MD037 -->
LOWESS for trend extraction and temporal smoothing.

## Overview

Time series data often contains noise, seasonality, and trends. LOWESS provides flexible trend extraction without parametric assumptions.

---

## Basic Trend Extraction

`fraction = 0.1` sizes the neighbourhood as 10% of the data at each evaluation point — narrow enough to follow a slowly varying trend without smearing periodic variation. Three robustness `iterations` down-weight noise spikes so they cannot bias the fitted curve; this is especially important when the signal-to-noise ratio is low or when occasional outliers are expected.

```javascript
const { Lowess } = require('fastlowess-wasm');

const n = 500;
const t = Float64Array.from({ length: n }, (_, i) => i * 100.0 / (n - 1));
const y = Float64Array.from(t, (ti, i) => 10.0 + 0.5 * ti + 3.0 * Math.sin(ti / 10.0) + (((i * 7 + 3) % 1.7) - 0.85) * 3.0);

const model = new Lowess({ 
    fraction: 0.1, 
    iterations: 3 
});
const result = model.fit(t, y);
console.log("y[0]:", result.y[0].toFixed(4));
```

```output
y[0]: 11.3216
```

---

## Detrending

Remove trend to analyze residual patterns.

Setting `return_residuals = True` stores `observed − smoothed` alongside the smooth. A slightly wider `fraction = 0.3` produces a smoother baseline trend, so short-duration oscillations end up in the residuals rather than being absorbed into the trend component. The residual series is then ready for spectral analysis, seasonality detection, or change-point methods.

```javascript
const { Lowess } = require('fastlowess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const model = new Lowess({ 
    fraction: 0.3, 
    iterations: 3, 
    return_residuals: true 
});
const result = model.fit(x, y);
console.log("y[0]:", result.y[0].toFixed(4), "residual[0]:", result.residuals[0].toFixed(4));
```

```output
y[0]: 0.2582 residual[0]: -0.1582
```

---

## Forecasting with Prediction Intervals

Prediction intervals widen the uncertainty band to include both the uncertainty in the fitted curve (confidence interval) and the expected scatter of new observations around it. `fraction = 0.2` offers a balance between local detail and stable interval width — too small a fraction produces jagged interval edges; too large a fraction underestimates local variance near turning points.

```javascript
const { Lowess } = require('fastlowess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const model = new Lowess({
    fraction: 0.2,
    iterations: 3,
    prediction_intervals: 0.95
});
const result = model.fit(x, y);
console.log("Prediction lower[0]:", result.prediction_lower[0].toFixed(4));
```

```output
Prediction lower[0]: 0.1580
```

---

## Handling Missing Data

LOWESS naturally handles irregular time sampling:

```javascript
const { Lowess } = require('fastlowess-wasm');

const n = 100;
const tIrregular = Float64Array.from({ length: n }, (_, i) => i * 1.0 + (i * 31 % 10) * 0.1).sort((a, b) => a - b);
const yIrregular = Float64Array.from(tIrregular, t => 10 + 0.3 * t + 2.0 * Math.sin(t * 0.1));
const model = new Lowess({ fraction: 0.2 });
const result = model.fit(tIrregular, yIrregular);
console.log("y[0]:", result.y[0].toFixed(4));
```

```output
y[0]: 11.3273
```

---

## Multi-Scale Analysis

Use different fractions to extract features at different scales:

```javascript
const { Lowess } = require('fastlowess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const trends = [0.05, 0.2, 0.5].map(f => {
    const model = new Lowess({ fraction: f });
    const result = model.fit(x, y);
    return result.y;
});
console.log("Trend y[0] values:", trends.map(t => t[0].toFixed(4)));
```

```output
Trend y[0] values: [ '0.1317', '0.2244', '0.3344' ]
```

---

## Gene Expression Time Course

Biological application:

```javascript
const { Lowess } = require('fastlowess-wasm');

const hours = Float64Array.from({ length: 49 }, (_, i) => i * 0.5);
const expression = Float64Array.from(hours, (h, i) => 100.0 * (1.0 + 0.5 * Math.sin(h * Math.PI / 12)) + (((i * 7 + 3) % 1.7) - 0.85) * 10.0);
const model = new Lowess({ fraction: 0.3, iterations: 3, confidence_intervals: 0.95, return_diagnostics: true });
const result = model.fit(hours, expression);

console.log("R2:", result.diagnostics.r_squared.toFixed(4));
```

```output
R2: 0.9731
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
- [Cross-Validation](cross-validation.md) — Optimal fraction selection
- [Boundary Handling](boundary.md) — Edge bias in trend extraction
- [API Reference](api.md) — Full parameter reference
