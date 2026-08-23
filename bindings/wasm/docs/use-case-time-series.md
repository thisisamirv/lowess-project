<!-- markdownlint-disable MD024 MD046 MD033 MD037 -->
# Time Series Analysis

LOWESS for trend extraction and temporal smoothing.

## Overview

Time series data often contains noise, seasonality, and trends. LOWESS provides flexible trend extraction without parametric assumptions.

---

## Basic Trend Extraction

`fraction = 0.1` sizes the neighbourhood as 10% of the data at each evaluation point — narrow enough to follow a slowly varying trend without smearing periodic variation. Three robustness `iterations` down-weight noise spikes so they cannot bias the fitted curve; this is especially important when the signal-to-noise ratio is low or when occasional outliers are expected.

```javascript
const { Lowess } = require('fastlowess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const model = new Lowess({ 
    fraction: 0.1, 
    iterations: 3 
});
const result = model.fit(x, y);

// Trend values in result.y
```

---

## Detrending

Remove trend to analyze residual patterns.

Setting `return_residuals = True` stores `observed − smoothed` alongside the smooth. A slightly wider `fraction = 0.3` produces a smoother baseline trend, so short-duration oscillations end up in the residuals rather than being absorbed into the trend component. The residual series is then ready for spectral analysis, seasonality detection, or change-point methods.

```javascript
const { Lowess } = require('fastlowess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const model = new Lowess({ 
    fraction: 0.3, 
    iterations: 3, 
    return_residuals: true 
});
const result = model.fit(x, y);

// Access result.y (trend) and result.residuals (detrended)
```

---

## Forecasting with Prediction Intervals

Prediction intervals widen the uncertainty band to include both the uncertainty in the fitted curve (confidence interval) and the expected scatter of new observations around it. `fraction = 0.2` offers a balance between local detail and stable interval width — too small a fraction produces jagged interval edges; too large a fraction underestimates local variance near turning points.

```javascript
const { Lowess } = require('fastlowess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const model = new Lowess({
    fraction: 0.2,
    iterations: 3,
    prediction_intervals: 0.95
});
const result = model.fit(x, y);

// Access result.prediction_lower and result.prediction_upper
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
```

---

## Multi-Scale Analysis

Use different fractions to extract features at different scales:

```javascript
const { Lowess } = require('fastlowess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const trends = [0.05, 0.2, 0.5].map(f => {
    const model = new Lowess({ fraction: f });
    const result = model.fit(x, y);
    return result.y;
});
```

---

## Gene Expression Time Course

Biological application:

```javascript
const { Lowess } = require('fastlowess-wasm');

const n = 24;
const hours = Float64Array.from({ length: n }, (_, i) => i);
const expression = Float64Array.from(hours, h => 5 + 3 * Math.sin(h * Math.PI / 12) + (h % 3) * 0.2);
const model = new Lowess({ fraction: 0.3, iterations: 3, return_diagnostics: true });
const result = model.fit(hours, expression);

console.log("R²:", result.diagnostics.r_squared);
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

- [Real-Time Processing](real-time.md) — For streaming time series
- [Cross-Validation](../user-guide/cross-validation.md) — Optimal fraction selection
- [Boundary Handling](../user-guide/boundary.md) — Edge bias in trend extraction
- [Parameters](../user-guide/parameters.md) — Full parameter reference
