# StreamingLowess — Rust API Reference

See also: [lowess & lowess Rust API Reference](rust.md)

## Struct

### `StreamingLowess`

Streaming mode for large datasets.

**Constructor:**

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let mut processor = StreamingLowess::<f64>::new();

    Ok(())
}
```

**Methods:**

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut processor = StreamingLowess::new().fraction(0.5f64).chunk_size(50usize).overlap(10usize).build()?;
    let result = processor.process_chunk(&x[..50], &y[..50])?;
    println!("Fraction used: {}", result.fraction_used);

    Ok(())
}
```

```output
Fraction used: 0.5
```

* Processes a chunk of data. Returns `LowessResult<T>` with partial results.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut processor = StreamingLowess::new().fraction(0.5f64).chunk_size(50usize).overlap(10usize).build()?;
    processor.process_chunk(&x[..50], &y[..50])?;
    processor.process_chunk(&x[50..], &y[50..])?;
    let final_result = processor.finalize()?;
    println!("Fraction used: {}", final_result.fraction_used);

    Ok(())
}
```

```output
Fraction used: 0.5
```

* Finalizes processing and returns remaining buffered results.

## Result Structure

### `LowessResult<T>`

Returned by `process_chunk()` and `finalize()`.

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Vec<T>` | Sorted x values |
| `y` | `Vec<T>` | Smoothed y values |
| `fraction_used` | `T` | Fraction used |
| `iterations_used` | `Option<usize>` | Robustness iterations actually performed |
| `residuals` | `Option<Vec<T>>` | Residuals (if `return_residuals()`) |
| `robustness_weights` | `Option<Vec<T>>` | Robustness weights (if `return_robustness_weights()`) |
| `diagnostics` | `Option<Diagnostics<T>>` | Fit metrics (if `return_diagnostics()`) |
| `dimensions` | `usize` | Number of predictor dimensions |

See [rust.md](rust.md) for the full `LowessResult<T>` field reference.

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
