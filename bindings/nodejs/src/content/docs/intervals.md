---
title: Intervals
---
<!-- markdownlint-disable MD024 MD033 -->
Confidence and prediction intervals for uncertainty quantification.

## Overview

![Confidence and Prediction Intervals](../assets/diagrams/intervals_comparison.svg)

:::note[Adapter support]
Confidence and prediction intervals are available in **Batch** mode only. Streaming and Online modes do not support intervals.
:::

| Type | Represents | Width | Use |
| --- | --- | --- | --- |
| **Confidence** | Uncertainty in mean curve | Narrow | Where is the true trend? |
| **Prediction** | Uncertainty for new points | Wide | Where will new data fall? |

---

## Confidence Intervals

Estimate uncertainty in the smoothed curve itself.

```javascript
const fl = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);

const model = new fl.Lowess({fraction: 0.5, confidence_intervals: 0.95});
const result = model.fit(x, y);

result.y.slice(0, 5).forEach((y, i) => {
    console.log(`x=${result.x[i].toFixed(4)}: y=${y.toFixed(4)} [${result.confidence_lower[i].toFixed(4)}, ${result.confidence_upper[i].toFixed(4)}]`);
});
```

```output
x=0.0000: y=0.1181 [0.0551, 0.1812]
x=0.0635: y=0.1502 [0.0762, 0.2243]
x=0.1269: y=0.1833 [0.1205, 0.2461]
x=0.1904: y=0.2172 [0.1405, 0.2938]
x=0.2539: y=0.2518 [0.1770, 0.3266]
```

---

## Prediction Intervals

Estimate where new observations might fall.

```javascript
const fl = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);

const model = new fl.Lowess({fraction: 0.5, prediction_intervals: 0.95});
const result = model.fit(x, y);
console.log(`Prediction bounds: [${result.prediction_lower[0]}, ${result.prediction_upper[0]}]`);
```

```output
Prediction bounds: [-0.35046106311855035, 0.5866944919326049]
```

---

## Both Intervals

Request both types simultaneously:

```javascript
const fl = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);

const model = new fl.Lowess({fraction: 0.5,
    confidence_intervals: 0.95,
    prediction_intervals: 0.95});
const result = model.fit(x, y);
console.log("95% CI: [" + result.confidence_lower[0].toFixed(4) + ", " + result.confidence_upper[0].toFixed(4) + "]");
```

```output
95% CI: [0.0551, 0.1812]
```

---

## Confidence Levels

Common levels and their z-values:

| Level | z-value | Interpretation |
| --- | --- | --- |
| 0.90 | 1.645 | 90% of intervals contain true value |
| 0.95 | 1.960 | 95% of intervals contain true value |
| 0.99 | 2.576 | 99% of intervals contain true value |

```javascript
const fl = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);

// 99% confidence interval
const model = new fl.Lowess({confidence_intervals: 0.99});
const result = model.fit(x, y);
console.log("99% CI: [" + result.confidence_lower[0].toFixed(4) + ", " + result.confidence_upper[0].toFixed(4) + "]");
```

```output
99% CI: [0.0789, 0.2534]
```

---

## Standard Errors

Access standard errors directly (available when intervals are computed):

```javascript
const fl = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);

const model = new fl.Lowess({confidence_intervals: 0.95});
const result = model.fit(x, y);

result.standard_errors.slice(0, 5).forEach((se, i) => {
    console.log(`Point ${i}: SE = ${se.toFixed(4)}`);
});
```

```output
Point 0: SE = 0.0339
Point 1: SE = 0.0392
Point 2: SE = 0.0345
Point 3: SE = 0.0407
Point 4: SE = 0.0410
```

---

## Availability

:::caution[Batch Mode Only]
Confidence and prediction intervals are only available in **Batch** mode. Streaming and Online modes do not support intervals.
:::

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✗ | ✗ |
| Prediction intervals | ✓ | ✗ | ✗ |
| Standard errors | ✓ | ✗ | ✗ |
