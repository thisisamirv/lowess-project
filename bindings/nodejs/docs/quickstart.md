<!-- markdownlint-disable MD024 MD046 -->
# Quick Start

Get up and running with LOWESS in minutes.

## Basic Smoothing

Smooth a noisy sine wave — the kind of signal where LOWESS shines. Each example recovers the underlying trend from 100 points of Gaussian noise.

```javascript
const { Lowess } = require('fastlowess');

// 100-point noisy sine wave
const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const model = new Lowess({ fraction: 0.3, iterations: 3 });
const result = model.fit(x, y);

console.log(`First smoothed: ${result.y[0].toFixed(4)}  (true: ${Math.sin(x[0]).toFixed(4)})`);
```

---

## With Confidence Intervals

```javascript
const { Lowess } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);

const model = new Lowess({
    fraction: 0.5,
    iterations: 3,
    confidence_intervals: 0.95,
    prediction_intervals: 0.95,
    return_diagnostics: true
});
const result = model.fit(x, y);

console.log("Smoothed:", result.y);
console.log("CI Lower:", result.confidence_lower);
console.log("CI Upper:", result.confidence_upper);
console.log("R²:", result.diagnostics.r_squared);
```

---

## Handling Outliers

LOWESS can robustly handle outliers through iterative reweighting:

```javascript
const { Lowess } = require('fastlowess');

const xOut = new Float64Array([1, 2, 3, 4, 5, 6]);
const yWithOutlier = new Float64Array([2.0, 4.0, 6.0, 50.0, 10.0, 12.0]);

const model = new Lowess({
    fraction: 0.5,
    iterations: 5,
    robustness_method: "bisquare",
    return_robustness_weights: true
});
const result = model.fit(xOut, yWithOutlier);

// Outliers will have low robustness weights
result.robustness_weights.forEach((w, i) => {
    if (w < 0.5) {
        console.log(`Point ${i} is likely an outlier (weight: ${w.toFixed(3)})`);
    }
});
```

---

## Streaming Mode

For datasets too large to fit in memory, stream them in fixed-size chunks with overlap.

```javascript
const { StreamingLowess } = require('fastlowess');

const n = 5000;
const x = Float64Array.from({ length: n }, (_, i) => i * 10 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) =>
    Math.sin(xi / Math.PI) * Math.exp(-xi / 30) +
    (((i * 7 + 3) % 17) / 17 - 0.5) * 0.3
);

const model = new StreamingLowess(
    { fraction: 0.2 },
    { chunk_size: 1000, overlap: 100, merge_strategy: 'weighted_average' }
);

const chunk_size = 1000;
for (let start = 0; start <= 4000; start += chunk_size) {
    const end = Math.min(start + chunk_size, n);
    model.process_chunk(x.slice(start, end), y.slice(start, end));
}
const result = model.finalize();
console.log(`Smoothed ${result.y.length} points in streaming mode`);
```

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
