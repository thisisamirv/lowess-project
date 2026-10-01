---
title: Intervals
---
<!-- markdownlint-disable MD024 MD033 -->
Confidence and prediction intervals for uncertainty quantification.

## Overview

![Confidence and Prediction Intervals](../../assets/diagrams/intervals_comparison.svg)

:::note[Adapter support]
Confidence and prediction intervals are available in **Batch** mode, **Streaming** mode (computed per chunk and merged across overlap boundaries via `merge_strategy`, like `y`/`derivative`), and **Online** mode when `update_mode: "full"` is set (throws if combined with the default `"incremental"` mode).
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
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const model = new fl.Lowess({fraction: 0.5, intervals: {confidence: 0.95}});
const result = model.fit(x, y);

result.y.slice(0, 5).forEach((y, i) => {
    console.log(`x=${result.x[i].toFixed(4)}: y=${y.toFixed(4)} [${result.confidence_lower[i].toFixed(4)}, ${result.confidence_upper[i].toFixed(4)}]`);
});
```

```output
x=0.0000: y=0.3325 [0.2929, 0.3722]
x=0.0635: y=0.3593 [0.3190, 0.3995]
x=0.1269: y=0.3871 [0.3462, 0.4280]
x=0.1904: y=0.4160 [0.3743, 0.4576]
x=0.2539: y=0.4458 [0.4034, 0.4881]
```

---

## Prediction Intervals

Estimate where new observations might fall.

```javascript
const fl = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const model = new fl.Lowess({fraction: 0.5, intervals: {prediction: 0.95}});
const result = model.fit(x, y);
console.log(`Prediction bounds: [${result.prediction_lower[0]}, ${result.prediction_upper[0]}]`);
```

```output
Prediction bounds: [-0.04054603685842645, 0.7056287954394038]
```

---

## Both Intervals

Request both types simultaneously:

```javascript
const fl = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const model = new fl.Lowess({fraction: 0.5,
    intervals: {confidence: 0.95, prediction: 0.95}});
const result = model.fit(x, y);
console.log("95% CI: [" + result.confidence_lower[0].toFixed(4) + ", " + result.confidence_upper[0].toFixed(4) + "]");
```

```output
95% CI: [0.2929, 0.3722]
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
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

// 99% confidence interval
const model = new fl.Lowess({intervals: {confidence: 0.99}});
const result = model.fit(x, y);
console.log("99% CI: [" + result.confidence_lower[0].toFixed(4) + ", " + result.confidence_upper[0].toFixed(4) + "]");
```

```output
99% CI: [0.3175, 0.4443]
```

---

## Standard Errors

Access standard errors directly (available when intervals are computed):

```javascript
const fl = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const model = new fl.Lowess({intervals: {confidence: 0.95}});
const result = model.fit(x, y);

result.standard_errors.slice(0, 5).forEach((se, i) => {
    console.log(`Point ${i}: SE = ${se.toFixed(4)}`);
});
```

```output
Point 0: SE = 0.0246
Point 1: SE = 0.0252
Point 2: SE = 0.0258
Point 3: SE = 0.0265
Point 4: SE = 0.0271
```

---

## Residual Bootstrap

Set `bootstrap` to at least `2` to replace analytic uncertainty with residual-bootstrap refits. Batch shares its outer `seed` with CV; Streaming restarts the seed per combined chunk, and Online per full-update window. Online bootstrap requires `update_mode: "full"`.

```javascript
const fl = require('fastlowess');

const x = Float64Array.from({ length: 30 }, (_, i) => i * 0.1);
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1 * Math.cos(7 * xi));
const intervals = { confidence: 0.95, prediction: 0.95, bootstrap: 20 };

const batch = new fl.Lowess({ intervals, seed: 42 }).fit(x, y);
console.log("Batch SEs:", batch.standard_errors.length);

const stream = new fl.StreamingLowess({ intervals, seed: 42 }, { chunk_size: x.length });
console.log("Streaming CI present:", stream.process_chunk(x, y).confidence_lower !== null);

const online = new fl.OnlineLowess({ intervals, seed: 42 }, { min_points: 5, update_mode: "full" });
let last = null;
for (let i = 0; i < 12; i++) {
    last = online.add_point(x[i], y[i]) ?? last;
}
console.log("Online PI present:", last.prediction_lower !== null);
```

```output
Batch SEs: 30
Streaming CI present: true
Online PI present: true
```

---

## Availability

:::note[Supported In All Three Adapters]
Confidence and prediction intervals are available in **Batch**, **Streaming**, and **Online** mode (`update_mode: "full"` only).
:::

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✓ | ✓ (`update_mode: "full"` only) |
| Prediction intervals | ✓ | ✓ | ✓ (`update_mode: "full"` only) |
| Standard errors | ✓ | ✓ | ✓ (`update_mode: "full"` only) |
| Residual bootstrap | ✓ | ✓ | ✓ (`update_mode: "full"` only) |
