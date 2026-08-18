# StreamingLowess — Rust API Reference

See also: [fastLowess & lowess Rust API Reference](rust.md)

## Struct

### `StreamingLowess`

Streaming mode for large datasets.

**Constructor:**

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let mut processor = StreamingLowess::new();

    Ok(())
}
```

**Methods:**

```rust
use fastLowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut processor = StreamingLowess::new().build()?;
    let result = processor.process_chunk(&x, &y)?;

    Ok(())
}
```

* Processes a chunk of data. Returns `LowessResult<T>` with partial results.

```rust
use fastLowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut processor = StreamingLowess::new().build()?;
    processor.process_chunk(&x, &y)?;
    let final_result = processor.finalize()?;

    Ok(())
}
```

* Finalizes processing and returns remaining buffered results.

## Builder Options

### Streaming Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size(usize)` | `usize` | `5000` | Data chunk size |
| `overlap(usize)` | `usize` | `500` | Overlap size |
| `merge_strategy(...)` | `merge_strategy` | `"weighted_average"` | Merge strategy |

## Options

### merge_strategy

*See: [Merge Strategies](../user-guide/merge.md)*

* `"weighted_average"` (default; alias: `"weighted"`)
* `"average"` (alias: `"mean"`)
* `"take_first"` (alias: `"first"`)
* `"take_last"` (alias: `"last"`)
