---
title: Execution Modes
---
<!-- markdownlint-disable MD024 MD033 MD046 -->
Choose the right adapter for your use case.

## Overview

Choose the first row below whose condition applies:

| Condition | Adapter |
| --- | --- |
| Data too large to fit in memory | `Streaming` |
| Fits in memory, need real-time/incremental updates | `Online` |
| Fits in memory, no real-time requirement | `Batch` |

| Mode | Use Case | Memory | Features |
| --- | --- | --- | --- |
| **Batch** | Complete datasets | Full | All features |
| **Streaming** | Large files (>100K) | Chunked | Residuals, robustness |
| **Online** | Real-time sensors | Fixed window | Incremental updates |

![Adapter Comparison](../../assets/diagrams/adapter_comparison.svg)

---

## Batch Adapter

Standard mode for complete datasets. **Supports all features.**

### When to Use

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

### Example

```javascript
const { Lowess } = require('fastlowess');

const x = Float64Array.from({ length: 100 }, (_, i) => i * 2 * Math.PI / 99);
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const model = new Lowess({
    fraction: 0.5,
    iterations: 3,
    confidence_intervals: 0.95,
    prediction_intervals: 0.95,
    return_diagnostics: true,
    parallel: true
});
const result = model.fit(x, y);
console.log(`95% CI at midpoint: [${result.confidence_lower[50].toFixed(4)}, ${result.confidence_upper[50].toFixed(4)}]`);
console.log(`R2: ${result.diagnostics.r_squared.toFixed(4)}`);
```

```output
95% CI at midpoint: [0.0393, 0.1077]
R2: 0.9664
```

---

## Streaming Adapter

Process large datasets in chunks with configurable overlap.

### When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `chunk_size` | 5000 | Points per chunk |
| `overlap` | `chunk_size / 10` | Overlap between chunks |
| `merge_strategy` | `"weighted_average"` | How to merge overlaps |

### Merge Strategies

| Strategy | Behavior |
| --- | --- |
| `"average"` | Average overlapping values |
| `"weighted_average"` | Distance-weighted blend |
| `"take_first"` | Keep left chunk values |
| `"take_last"` | Keep right chunk values |

### Example

```javascript
const { StreamingLowess } = require('fastlowess');

const x = Float64Array.from({ length: 100 }, (_, i) => i * 2 * Math.PI / 99);
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const stream = new StreamingLowess(
    { fraction: 0.3, iterations: 2 },
    { chunk_size: 5000, overlap: 500, merge_strategy: "average" }
);
stream.process_chunk(x, y);
const result = stream.finalize();
console.log(`Smoothed y[0]: ${result.y[0].toFixed(4)}`);
```

```output
Smoothed y[0]: 0.2578
```

---

:::caution[Always call finalize()]
Always call `stream.finalize()` after processing all chunks to retrieve buffered overlap data.
:::

## Online Adapter

Incremental updates with a sliding window for real-time data.

### When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `window_capacity` | 1000 | Max points in window |
| `min_points` | 2 | Points before output starts |
| `update_mode` | `"incremental"` | Update strategy |

### Update Modes

| Mode | Behavior | Speed |
| --- | --- | --- |
| `"incremental"` | Update only affected fits | Faster |
| `"full"` | Recompute entire window | More accurate |

### Example

```javascript
const { OnlineLowess } = require('fastlowess');

const x = Float64Array.from({ length: 100 }, (_, i) => i * 2 * Math.PI / 99);
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const online = new OnlineLowess(
    { fraction: 0.2, iterations: 1 },
    { window_capacity: 100, min_points: 5, update_mode: "incremental" }
);
let shown = 0;
for (let i = 0; i < x.length && shown < 5; i++) {
    const result = online.add_point(x[i], y[i]);
    if (result !== null) {
        console.log(result.y);
        shown++;
    }
}
```

```output
0.3511479871810792
0.4120334456984871
0.4716624556603275
0.5297949120891716
0.5861967361004687
```

---

## Feature Comparison

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✗ | ✗ |
| Prediction intervals | ✓ | ✗ | ✗ |
| Cross-validation | ✓ | ✗ | ✗ |
| Diagnostics | ✓ | ✓ | ✗ |
| Residuals | ✓ | ✓ | ✓ |
| Robustness weights | ✓ | ✓ | ✓ |
| Parallel execution | ✓ | ✓ | ✗ |

---

## Next Steps

- [API Reference](../api/api.md) — All configuration options
- [Streaming API](../api/api-streaming.md) · [Online API](../api/api-online.md)
