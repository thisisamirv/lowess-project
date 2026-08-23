# Online Adapter

Incremental updates with a sliding window for real-time data.

## When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

![Online Adapter](../assets/diagrams/online_comparison.svg)

## Parameters

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

## Example

```rust
use fastLowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();
    let sensor_stream: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi, yi)).collect();

    let mut processor = OnlineLowess::new()
        .fraction(0.2)
        .iterations(1)
        .window_capacity(100)
        .min_points(5)
        .update_mode("incremental")
        .build()?;

    // Process points as they arrive
    for (x, y) in sensor_stream {
        if let Some(output) = processor.add_point(x, y)? {
            println!("Smoothed: {:.2}", output.y);
        }
    }

    Ok(())
}
```

---
