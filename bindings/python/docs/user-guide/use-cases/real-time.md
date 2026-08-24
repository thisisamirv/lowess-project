# Real-Time Processing

Streaming and online LOWESS for live data.

## Overview

When data arrives continuously—from sensors, logs, or streaming pipelines—you need incremental smoothing that doesn't require reprocessing the entire dataset.

---

## Online Mode: Point-by-Point

For true real-time applications where each point must be processed immediately.

`window_capacity = 25` limits the internal buffer to the 25 most recent observations; each `add_point` call costs O(window) rather than growing with total history. `min_points = 5` suppresses output until the window holds enough points for a stable fit — calls made before that threshold return `null`/`None`/`nothing`. `update_mode = "incremental"` re-fits only the most recent point rather than the full window, halving typical latency at a modest accuracy cost.

### Sensor Data Example

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

## Simulate sensor readings arriving over time

np.random.seed(42)
n_readings = 100
times = np.arange(n_readings)
temperatures = 20 + 5 * np.sin(times / 10) + np.random.normal(0, 1, n_readings)

## Process with online mode

online = fl.OnlineLowess(
    fraction=0.3,
    window_capacity=25,    # Keep last 25 points
    min_points=5,          # Wait for 5 points before output
    update_mode="incremental"
)
for xi, yi in zip(times, temperatures):
    result = online.add_point(float(xi), float(yi))
    if result is not None:
        print(f"Time {xi:.0f}: smoothed = {result.y:.2f}")
:::

---

## Streaming Mode: Chunk Processing

For large datasets that arrive in batches or files.

`chunk_size` controls how many data points are processed in one pass; matching it to your file-read buffer or message-batch size avoids unnecessary copying. `overlap` retains that many points from the previous chunk as context so the neighbourhood at chunk boundaries is not artificially truncated. `merge_strategy = "weighted_average"` blends the overlapping region smoothly; use `"last"` if chunk boundaries are guaranteed to be well separated and no blending is needed.

:::{warning} Always call finalize()
The streaming adapter buffers overlap data. Call `finalize()` after the last chunk to retrieve the buffered tail.
:::

### Log File Processing

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

## Simulate large dataset arriving in chunks

total_points = 100000
chunk_size = 10000

## All at once with streaming handles chunking internally

x = np.arange(total_points, dtype=float)
y = np.sin(x / 1000) + np.random.normal(0, 0.1, total_points)

model = fl.StreamingLowess(
    fraction=0.05,
    chunk_size=10000,
    overlap=1000,
    merge_strategy="weighted_average"
)
model.process_chunk(x, y)
result = model.finalize()

print(f"Processed {len(result.y)} points")
:::

---

## Real-Time Dashboard Example

The dashboard pattern uses a plain LOWESS fit on a manually managed sliding window rather than `OnlineLowess`. This is the simplest approach when your UI framework already owns the data buffer and you only need the most recent smoothed value per frame. The trade-off is a full O(window²) refit on every tick; for high-frequency streams prefer `OnlineLowess` with `update_mode = "incremental"` to bound per-frame cost.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

## Simulated real-time dashboard sliding window

window_capacity = 50
data_x, data_y = [], []

for i in range(200):
    x, y = i, 25.0 + 10 * np.sin(i / 20) + np.random.normal(0, 2)
    data_x.append(x)
    data_y.append(y)

    if len(data_x) > window_capacity:
        data_x = data_x[-window_capacity:]
        data_y = data_y[-window_capacity:]
    
    if len(data_x) >= 5:
        model = fl.Lowess(fraction=0.4)
        result = model.fit(np.array(data_x, dtype=float), np.array(data_y, dtype=float))
        current_smoothed = result.y[-1]
:::

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

- [Execution Modes](../adapters.md) — Detailed mode comparison
- [Merge Strategies](../merge.md) — Chunk reconciliation in depth
- [Scaling Methods](../scaling.md) — Robustness scale estimation
- [Time Series](time-series.md) — General time series analysis
