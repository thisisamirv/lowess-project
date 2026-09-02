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

times = np.arange(100, dtype=float)
temperatures = 20.0 + 5.0 *np.sin(times / 10.0) + np.sin(times* 1.7) * 0.5

## Process with online mode

online = fl.OnlineLowess(
    fraction=0.3,
    iterations=1,
    window_capacity=25,    # Keep last 25 points
    min_points=5,          # Wait for 5 points before output
    update_mode="incremental"
)
count = 0
for xi, yi in zip(times, temperatures):
    result = online.add_point(float(xi), float(yi))
    if result is not None:
        if count < 5:
            print(f"Time {xi:.0f}: smoothed = {result.y:.4f}")
        count += 1
print(f"... ({count - 5} more)")
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

chunk1_x = np.arange(50, dtype=float)
chunk1_y = np.sin(chunk1_x) + 0.1
chunk2_x = np.arange(50, 100, dtype=float)
chunk2_y = np.sin(chunk2_x) + 0.1

model = fl.StreamingLowess(
    fraction=0.1,
    iterations=2,
    chunk_size=50,
    overlap=10,
    merge_strategy="weighted_average"
)
model.process_chunk(chunk1_x, chunk1_y)
model.process_chunk(chunk2_x, chunk2_y)
result = model.finalize()

print(f"y[0]: {result.y[0]:.6f}")
:::

---

## Real-Time Dashboard Example

The dashboard pattern uses a plain LOWESS fit on a manually managed sliding window rather than `OnlineLowess`. This is the simplest approach when your UI framework already owns the data buffer and you only need the most recent smoothed value per frame. The trade-off is a full O(window^2) refit on every tick; for high-frequency streams prefer `OnlineLowess` with `update_mode = "incremental"` to bound per-frame cost.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

## Simulated real-time dashboard sliding window

n = 100
x = np.linspace(0, 2 * np.pi, n)
y = np.sin(x) + 0.1

window_capacity = 50
data_x, data_y = [], []
latest = 0.0

for i in range(n):
    data_x.append(x[i])
    data_y.append(y[i])

    if len(data_x) > window_capacity:
        data_x = data_x[-window_capacity:]
        data_y = data_y[-window_capacity:]

    if len(data_x) < 2:
        continue

    model = fl.Lowess(fraction=0.4)
    result = model.fit(np.array(data_x, dtype=float), np.array(data_y, dtype=float))
    latest = result.y[-1]

print(f"Smoothed (dashboard, latest tick): {latest:.4f}")
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

- [Execution Modes](../guide/adapters.md) — Detailed mode comparison
- [Merge Strategies](../advanced/merge.md) — Chunk reconciliation in depth
- [Scaling Methods](../weighting/scaling.md) — Robustness scale estimation
- [Time Series](time-series.md) — General time series analysis
