<!-- markdownlint-disable MD033 -->
# Real-Time Processing

Streaming and online LOWESS for live data.

## Overview

When data arrives continuously—from sensors, logs, or streaming pipelines—you need incremental smoothing that doesn't require reprocessing the entire dataset.

---

## Online Mode: Point-by-Point

For true real-time applications where each point must be processed immediately.

`window_capacity = 25` limits the internal buffer to the 25 most recent observations; each `add_point` call costs O(window) rather than growing with total history. `min_points = 5` suppresses output until the window holds enough points for a stable fit — calls made before that threshold return `null`/`None`/`nothing`. `update_mode = "incremental"` re-fits only the most recent point rather than the full window, halving typical latency at a modest accuracy cost.

### Sensor Data Example

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {

    let mut processor = OnlineLowess::new()
        .fraction(0.3)
        .iterations(1)
        .window_capacity(25)
        .min_points(5)
        .update_mode("incremental")
        .build()?;

    // Simulate real-time data arrival
    let mut count = 0;
    for i in 0..100 {
        let xi = i as f64;
        let yi = 20.0 + 5.0 * (xi / 10.0).sin() + (xi * 1.7).sin() * 0.5;

        if let Some(output) = processor.add_point(xi, yi)? {
            if count < 5 {
                println!("Time {}: smoothed = {:.4}", xi, output.y);
            }
            count += 1;
        }
    }
    println!("... ({} more)", count - 5);

    Ok(())
}
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

!!! warning "Always call finalize()"
    The streaming adapter buffers overlap data. Call `finalize()` after the last chunk to retrieve the buffered tail.

### Log File Processing

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let chunk1_x: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let chunk1_y: Vec<f64> = chunk1_x.iter().map(|&xi| xi.sin() + 0.1).collect();
    let chunk2_x: Vec<f64> = (50..100).map(|i| i as f64).collect();
    let chunk2_y: Vec<f64> = chunk2_x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut processor = StreamingLowess::new()
        .fraction(0.1)
        .iterations(2)
        .chunk_size(50)
        .overlap(10)
        .merge_strategy("weighted_average")
        .build()?;

    // Process chunks as they arrive
    processor.process_chunk(&chunk1_x, &chunk1_y)?;
    processor.process_chunk(&chunk2_x, &chunk2_y)?;

    // CRITICAL: Get buffered overlap data
    let final_result = processor.finalize()?;
    println!("First smoothed value (streaming log): {}", final_result.y[0]);

    Ok(())
}
```

```output
First smoothed value (streaming log): 0.5164838198010315
```

---

## Real-Time Dashboard Example

The dashboard pattern uses a plain LOWESS fit on a manually managed sliding window rather than `OnlineLowess`. This is the simplest approach when your UI framework already owns the data buffer and you only need the most recent smoothed value per frame. The trade-off is a full O(window^2) refit on every tick; for high-frequency streams prefer `OnlineLowess` with `update_mode = "incremental"` to bound per-frame cost.

```rust
use fastLowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut window_x: Vec<f64> = Vec::new();
    let mut window_y: Vec<f64> = Vec::new();
    let mut latest = 0.0;

    for i in 0..x.len() {
        window_x.push(x[i]);
        window_y.push(y[i]);
        if window_x.len() > 50 {
            window_x.remove(0);
            window_y.remove(0);
        }
        if window_x.len() < 2 {
            continue;
        }

        let model = Lowess::new().fraction(0.4).build()?;
        let result = model.fit(&window_x, &window_y)?;
        latest = *result.y.last().unwrap();
    }

    println!("Smoothed (dashboard, latest tick): {}", latest);
    Ok(())
}
```

```output
Smoothed (dashboard, latest tick): -0.06634730089857399
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

- [Execution Modes](crate::doc::adapter_choice) — Detailed mode comparison
- [Merge Strategies](crate::doc::merge) — Chunk reconciliation in depth
- [Scaling Methods](crate::doc::scaling) — Robustness scale estimation
- [Time Series](crate::doc::use_cases::time_series) — General time series analysis
