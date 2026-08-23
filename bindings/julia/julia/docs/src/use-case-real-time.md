# Real-Time Processing

Streaming and online LOWESS for live data.

## Overview

When data arrives continuously—from sensors, logs, or streaming pipelines—you need incremental smoothing that doesn't require reprocessing the entire dataset.

---

## Online Mode: Point-by-Point

For true real-time applications where each point must be processed immediately.

`window_capacity = 25` limits the internal buffer to the 25 most recent observations; each `add_point` call costs O(window) rather than growing with total history. `min_points = 5` suppresses output until the window holds enough points for a stable fit — calls made before that threshold return `null`/`None`/`nothing`. `update_mode = "incremental"` re-fits only the most recent point rather than the full window, halving typical latency at a modest accuracy cost.

### Sensor Data Example

```julia
using FastLOWESS
using Random

rng = MersenneTwister(42)
# Simulate sensor readings 
times = collect(Float64, 1:100)
temperatures = 20.0 .+ 5.0 .* sin.(times ./ 10.0) .+ randn(rng, 100)

# Process with online mode
model = OnlineLowess(;
    fraction=0.3,
    window_capacity=25,
    min_points=5,
    update_mode="incremental"
)
for i in eachindex(times)
    result = add_point(model, times[i], temperatures[i])
    if result !== nothing
        println("Time $(times[i]): smoothed = $(round(result.y; digits=2))")
    end
end
```

---

## Streaming Mode: Chunk Processing

For large datasets that arrive in batches or files.

`chunk_size` controls how many data points are processed in one pass; matching it to your file-read buffer or message-batch size avoids unnecessary copying. `overlap` retains that many points from the previous chunk as context so the neighbourhood at chunk boundaries is not artificially truncated. `merge_strategy = "weighted_average"` blends the overlapping region smoothly; use `"last"` if chunk boundaries are guaranteed to be well separated and no blending is needed.

!!! warning "Always call finalize()"
    The streaming adapter buffers overlap data. Call `finalize()` after the last chunk to retrieve the buffered tail.

### Log File Processing

```julia
using FastLOWESS

# Large dataset
x = collect(0.0:1.0:100000.0)
y = sin.(x ./ 1000) .+ randn(length(x)) .* 0.1

# Streaming mode handles everything internally
model = StreamingLowess(;
    fraction=0.05,
    chunk_size=10000,
    overlap=1000,
    merge_strategy="weighted_average"
)
process_chunk(model, x, y)
result = finalize(model)
```

---

## Real-Time Dashboard Example

The dashboard pattern uses a plain LOWESS fit on a manually managed sliding window rather than `OnlineLowess`. This is the simplest approach when your UI framework already owns the data buffer and you only need the most recent smoothed value per frame. The trade-off is a full O(window²) refit on every tick; for high-frequency streams prefer `OnlineLowess` with `update_mode = "incremental"` to bound per-frame cost.

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
- [Time Series](time-series.md) — General time series analysis
