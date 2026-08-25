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
const { OnlineLowess } = require('fastlowess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const processor = new OnlineLowess(
    { fraction: 0.3, iterations: 1 },
    { window_capacity: 25, min_points: 5, update_mode: "incremental" }
);

let rt_printed = 0;
for (let i = 0; i < x.length; i++) {
    const res = processor.add_point(x[i], y[i]);
    if (res !== undefined && res !== null) {
        if (rt_printed < 5) console.log("Smoothed y:", res.y.toFixed(4));
        rt_printed++;
    }
}
console.log(`... (${rt_printed - 5} more)`);
```

```output
Smoothed y: 0.4453
Smoothed y: 0.1532
Smoothed y: 0.4599
Smoothed y: 0.1651
Smoothed y: 0.4685
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
const { StreamingLowess } = require('fastlowess-wasm');

const n = 50;
const x1 = Float64Array.from({ length: n }, (_, i) => i);
const y1 = Float64Array.from(x1, xi => Math.sin(xi * 0.1) + 0.1);
const x2 = Float64Array.from({ length: n }, (_, i) => n + i);
const y2 = Float64Array.from(x2, xi => Math.sin(xi * 0.1) + 0.1);

const processor = new StreamingLowess(
    { fraction: 0.1, iterations: 2 },
    { chunk_size: 5000, overlap: 500 }
);

// Process chunks as they arrive
const result1 = processor.process_chunk(x1, y1);
const result2 = processor.process_chunk(x2, y2);
const finalResult = processor.finalize();
console.log("y[0]:", finalResult.y[0].toFixed(4));
```

```output
y[0]: 0.2230
```

---

## Real-Time Dashboard Example

The dashboard pattern uses a plain LOWESS fit on a manually managed sliding window rather than `OnlineLowess`. This is the simplest approach when your UI framework already owns the data buffer and you only need the most recent smoothed value per frame. The trade-off is a full O(window²) refit on every tick; for high-frequency streams prefer `OnlineLowess` with `update_mode = "incremental"` to bound per-frame cost.

```javascript
const { Lowess } = require('fastlowess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

// Sliding window logic
const windowX = [], windowY = [];
for (let i = 0; i < x.length; i++) {
    windowX.push(x[i]);
    windowY.push(y[i]);

    if (windowX.length > 50) {
        windowX.shift();
        windowY.shift();
    }

    if (windowX.length < 2) continue;
    const model = new Lowess({ fraction: 0.4 });
    const result = model.fit(new Float64Array(windowX), new Float64Array(windowY));
    const smoothed = result.y[result.y.length - 1];
    if (i === x.length - 1) console.log("Smoothed:", smoothed.toFixed(4));
}
```

```output
Smoothed: 0.0358
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
