# Real-Time Processing

Streaming and online LOWESS for live data.

## Overview

When data arrives continuously—from sensors, logs, or streaming pipelines—you need incremental smoothing that doesn't require reprocessing the entire dataset.

---

## Online Mode: Point-by-Point

For true real-time applications where each point must be processed immediately.

`window_capacity = 25` limits the internal buffer to the 25 most recent observations; each `add_point` call costs O(window) rather than growing with total history. `min_points = 5` suppresses output until the window holds enough points for a stable fit — calls made before that threshold return `null`/`None`/`nothing`. `update_mode = "incremental"` re-fits only the most recent point rather than the full window, halving typical latency at a modest accuracy cost.

### Sensor Data Example

```@example use-case-real-time
using FastLOWESS

# Simulate sensor readings
times = collect(Float64, 0:99)
temperatures = 20.0 .+ 5.0 .* sin.(times ./ 10.0) .+ sin.(times .* 1.7) .* 0.5

# Process with online mode
model = OnlineLowess(;
    fraction=0.3,
    iterations=1,
    window_capacity=25,
    min_points=5,
    update_mode="incremental"
)
count = 0
for i in eachindex(times)
    result = add_point(model, times[i], temperatures[i])
    if result !== nothing
        if count < 5
            println("Time $(times[i]): smoothed = $(round(result.y; digits=4))")
        end
        global count += 1
    end
end
println("... ($(count - 5) more)")
```

---

## Streaming Mode: Chunk Processing

For large datasets that arrive in batches or files.

`chunk_size` controls how many data points are processed in one pass; matching it to your file-read buffer or message-batch size avoids unnecessary copying. `overlap` retains that many points from the previous chunk as context so the neighbourhood at chunk boundaries is not artificially truncated. `merge_strategy = "weighted_average"` blends the overlapping region smoothly; use `"last"` if chunk boundaries are guaranteed to be well separated and no blending is needed.

!!! warning "Always call finalize()"
    The streaming adapter buffers overlap data. Call `finalize()` after the last chunk to retrieve the buffered tail.

### Log File Processing

```@example use-case-real-time
using FastLOWESS

chunk1_x = collect(Float64, 0:49)
chunk1_y = sin.(chunk1_x) .+ 0.1
chunk2_x = collect(Float64, 50:99)
chunk2_y = sin.(chunk2_x) .+ 0.1

# Streaming mode handles everything internally
model = StreamingLowess(;
    fraction=0.1,
    iterations=2,
    chunk_size=50,
    overlap=10,
    merge_strategy="weighted_average"
)
process_chunk(model, chunk1_x, chunk1_y)
process_chunk(model, chunk2_x, chunk2_y)
result = finalize(model)
println("y[0]: ", result.y[1])
```

---

## Real-Time Dashboard Example

The dashboard pattern uses a plain LOWESS fit on a manually managed sliding window rather than `OnlineLowess`. This is the simplest approach when your UI framework already owns the data buffer and you only need the most recent smoothed value per frame. The trade-off is a full O(window^2) refit on every tick; for high-frequency streams prefer `OnlineLowess` with `update_mode = "incremental"` to bound per-frame cost.

```@example use-case-real-time
using FastLOWESS

n = 100
x = collect(range(0, 2π, length=n))
y = sin.(x) .+ 0.1

window_x = Float64[]
window_y = Float64[]
latest = 0.0
for i in eachindex(x)
    push!(window_x, x[i])
    push!(window_y, y[i])
    if length(window_x) > 50
        popfirst!(window_x)
        popfirst!(window_y)
    end
    length(window_x) < 2 && continue

    model = Lowess(; fraction=0.4)
    result = fit(model, window_x, window_y)
    global latest = result.y[end]
end
println("Smoothed (dashboard, latest tick): ", latest)
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
