---
title: Real-Time Processing
---
<!-- markdownlint-disable MD033 -->
Streaming and online LOWESS for live data.

## Overview

When data arrives continuously—from sensors, logs, or streaming pipelines—you need incremental smoothing that doesn't require reprocessing the entire dataset.

---

## Online Mode: Point-by-Point

For true real-time applications where each point must be processed immediately.

`window_capacity = 25` limits the internal buffer to the 25 most recent observations; each `add_point` call costs O(window) rather than growing with total history. `min_points = 5` suppresses output until the window holds enough points for a stable fit — calls made before that threshold return `null`/`None`/`nothing`. `update_mode = "incremental"` re-fits only the most recent point rather than the full window, halving typical latency at a modest accuracy cost.

### Sensor Data Example

```javascript
const { OnlineLowess } = require('fastlowess');

const processor = new OnlineLowess(
    { fraction: 0.3, iterations: 1 },
    { window_capacity: 25, min_points: 5, update_mode: "incremental" }
);

// Simulate real-time data arrival
let count = 0;
for (let i = 0; i < 100; i++) {
    const xi = i;
    const yi = 20.0 + 5.0 * Math.sin(xi / 10.0) + Math.sin(xi * 1.7) * 0.5;

    const res = processor.add_point(xi, yi);
    if (res !== null && res !== undefined) {
        if (count < 5) console.log(`Time ${xi}: smoothed = ${res.y.toFixed(4)}`);
        count++;
    }
}
console.log(`... (${count - 5} more)`);
```

```output
Time 4: smoothed = 22.1941
Time 5: smoothed = 22.7964
Time 6: smoothed = 22.4733
Time 7: smoothed = 22.9120
Time 8: smoothed = 24.0164
... (91 more)
```

---

## Streaming Mode: Chunk Processing

For large datasets that arrive in batches or files.

`chunk_size` controls how many data points are processed in one pass; matching it to your file-read buffer or message-batch size avoids unnecessary copying. `overlap` retains that many points from the previous chunk as context so the neighbourhood at chunk boundaries is not artificially truncated. `merge_strategy = "weighted_average"` blends the overlapping region smoothly; use `"last"` if chunk boundaries are guaranteed to be well separated and no blending is needed.

:::caution[Always call finalize()]
The streaming adapter buffers overlap data. Call `finalize()` after the last chunk to retrieve the buffered tail.
:::

### Log File Processing

```javascript
const { StreamingLowess } = require('fastlowess');

const chunk1_x = Float64Array.from({ length: 50 }, (_, i) => i);
const chunk1_y = Float64Array.from(chunk1_x, xi => Math.sin(xi) + 0.1);
const chunk2_x = Float64Array.from({ length: 50 }, (_, i) => i + 50);
const chunk2_y = Float64Array.from(chunk2_x, xi => Math.sin(xi) + 0.1);

const processor = new StreamingLowess(
    { fraction: 0.1, iterations: 2 },
    { chunk_size: 50, overlap: 10, merge_strategy: "weighted_average" }
);

// Process chunks as they arrive
processor.process_chunk(chunk1_x, chunk1_y);
processor.process_chunk(chunk2_x, chunk2_y);

// CRITICAL: Get buffered overlap data
const finalResult = processor.finalize();
console.log("y[0]:", finalResult.y[0].toFixed(6));
```

```output
y[0]: 0.516484
```

---

## Real-Time Dashboard Example

The dashboard pattern uses a plain LOWESS fit on a manually managed sliding window rather than `OnlineLowess`. This is the simplest approach when your UI framework already owns the data buffer and you only need the most recent smoothed value per frame. The trade-off is a full O(window^2) refit on every tick; for high-frequency streams prefer `OnlineLowess` with `update_mode = "incremental"` to bound per-frame cost.

```javascript
const fl = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

let windowX = [], windowY = [];
let lastSmoothed = 0;

for (let i = 0; i < x.length; i++) {
    windowX.push(x[i]);
    windowY.push(y[i]);

    if (windowX.length > 50) {
        windowX.shift();
        windowY.shift();
    }

    if (windowX.length < 2) continue;
    const model = new fl.Lowess({ fraction: 0.4 });
    const result = model.fit(new Float64Array(windowX), new Float64Array(windowY));
    lastSmoothed = result.y[result.y.length - 1];
}
console.log("Smoothed (dashboard, latest tick):", lastSmoothed.toFixed(4));
```

```output
Smoothed (dashboard, latest tick): -0.0663
```

---

## Choosing Parameters

### Online Mode

| Parameter | Guidance |
| --- | --- |
| `window_capacity` | Enough history for `fraction` to work |
| `min_points` | 2–5 typically; higher for stability |
| `update_mode` | `"incremental"` for speed, `"full"` for accuracy |

### Streaming Mode

| Parameter | Guidance |
| --- | --- |
| `chunk_size` | Balance memory vs. processing overhead |
| `overlap` | 10–20% of chunk_size for smooth transitions |
| `merge_strategy` | `"weighted_average"` for best quality, `"average"` for simplicity |

---

## Performance Considerations

| Mode | Memory | Latency | Use Case |
| --- | --- | --- | --- |
| **Online** | Fixed (window) | ~1ms/point | Sensors, dashboards |
| **Streaming** | ~chunk_size | ~100ms/chunk | Large files, ETL |
| **Batch** | Full dataset | N/A | Analysis, reports |

---

## See Also

- [Execution Modes](../guide/adapter-choice.md) — Detailed mode comparison
- [Merge Strategies](../advanced/merge.md) — Chunk reconciliation in depth
- [Scaling Methods](../weighting/scaling.md) — Robustness scale estimation
- [Time Series](use-case-time-series.md) — General time series analysis
