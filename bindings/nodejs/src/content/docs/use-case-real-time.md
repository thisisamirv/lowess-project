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
for (let i = 0; i < 100; i++) {
    const x = i;
    const y = 20 + 5 * Math.sin(x / 10) + ((i * 7 + 3) % 17) / 17;
    
    const res = processor.add_point(x, y);
    if (res !== null && x % 20 === 0) {
        console.log(`Time ${x}: smoothed = ${res.y.toFixed(2)}`);
    }
}
```

```output
Time 20: smoothed = 24.84
Time 40: smoothed = 16.76
Time 60: smoothed = 19.19
Time 80: smoothed = 25.38
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
const chunk1_y = Float64Array.from(chunk1_x, v => Math.sin(v * 0.1));
const chunk2_x = Float64Array.from({ length: 50 }, (_, i) => i + 50);
const chunk2_y = Float64Array.from(chunk2_x, v => Math.sin(v * 0.1));

const processor = new StreamingLowess(
    { fraction: 0.1, iterations: 2 },
    { chunk_size: 5000, overlap: 500 }
);

// Process chunks
const r1 = processor.process_chunk(chunk1_x, chunk1_y);
const r2 = processor.process_chunk(chunk2_x, chunk2_y);

// Always get buffered data
const finalResult = processor.finalize();
console.log("Smoothed", finalResult.y.length, "points via streaming");
```

```output
Smoothed 100 points via streaming
```

---

## Real-Time Dashboard Example

The dashboard pattern uses a plain LOWESS fit on a manually managed sliding window rather than `OnlineLowess`. This is the simplest approach when your UI framework already owns the data buffer and you only need the most recent smoothed value per frame. The trade-off is a full O(window²) refit on every tick; for high-frequency streams prefer `OnlineLowess` with `update_mode = "incremental"` to bound per-frame cost.

```javascript
const fl = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);

const window_capacity = 50;
let dataX = [], dataY = [];
let lastSmoothed = 0;

for (let i = 0; i < 200; i++) {
    dataX.push(i);
    dataY.push(25.0 + 10 * Math.sin(i / 20) + ((i*7+3)%17)/17*4 - 2);

    if (dataX.length > window_capacity) {
        dataX.shift();
        dataY.shift();
    }

    if (dataX.length >= 5) {
        const xArr = new Float64Array(dataX);
        const yArr = new Float64Array(dataY);
        const model = new fl.Lowess({ fraction: 0.4 });
        const result = model.fit(xArr, yArr);
        lastSmoothed = result.y[result.y.length - 1];
    }
}
console.log("Last smoothed value (sliding window):", lastSmoothed.toFixed(4));
```

```output
Last smoothed value (sliding window): 19.9064
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

- [Execution Modes](adapter-choice.md) — Detailed mode comparison
- [Merge Strategies](merge.md) — Chunk reconciliation in depth
- [Scaling Methods](scaling.md) — Robustness scale estimation
- [Time Series](use-case-time-series.md) — General time series analysis
