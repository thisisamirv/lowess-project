# Streaming Adapter

Process large datasets in chunks with configurable overlap.

## When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `chunk_size` | 5000 | Points per chunk |
| `overlap` | 500 | Overlap between chunks |
| `merge_strategy` | `"weighted_average"` | How to merge overlaps |

### Merge Strategies

| Strategy | Behavior |
| --- | --- |
| `"average"` | Average overlapping values |
| `"weighted_average"` | Distance-weighted blend |
| `"take_first"` | Keep left chunk values |
| `"take_last"` | Keep right chunk values |

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

## Example

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn write_output(_data: &[f64]) {}

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let data_chunks = vec![
        (x[..50].to_vec(), y[..50].to_vec()),
        (x[50..].to_vec(), y[50..].to_vec()),
    ];

    let mut processor = StreamingLowess::new()
        .build()?;

    // Process chunks (e.g., from a file reader)
    for (chunk_x, chunk_y) in data_chunks {
        let result = processor.process_chunk(&chunk_x, &chunk_y)?;
        write_output(&result.y);
    }

    // IMPORTANT: Get remaining buffered data
    let final_result = processor.finalize()?;
    write_output(&final_result.y);

    Ok(())
}
```

---

!!! warning "Always call finalize()"
    In Rust, always call `processor.finalize()` after processing all chunks to retrieve buffered overlap data.
