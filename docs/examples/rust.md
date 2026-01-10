# Rust Examples

Complete Rust examples demonstrating fastLowess with the builder pattern and type safety.

## Batch Smoothing (fastLowess)

Parallel batch processing with confidence intervals, diagnostics, and cross-validation.

```rust
--8<-- "../../examples/fastLowess/fast_batch_smoothing.rs"
```

[:material-download: Download fast_batch_smoothing.rs](https://github.com/thisisamirv/lowess-project/blob/main/examples/fastLowess/fast_batch_smoothing.rs)

---

## Streaming Smoothing (fastLowess)

Process large datasets in memory-efficient chunks with parallel processing.

```rust
--8<-- "../../examples/fastLowess/fast_streaming_smoothing.rs"
```

[:material-download: Download fast_streaming_smoothing.rs](https://github.com/thisisamirv/lowess-project/blob/main/examples/fastLowess/fast_streaming_smoothing.rs)

---

## Online Smoothing (fastLowess)

Real-time smoothing with sliding window for streaming data applications.

```rust
--8<-- "../../examples/fastLowess/fast_online_smoothing.rs"
```

[:material-download: Download fast_online_smoothing.rs](https://github.com/thisisamirv/lowess-project/blob/main/examples/fastLowess/fast_online_smoothing.rs)

---

## Core lowess Examples

The core `lowess` crate provides single-threaded, `no_std`-compatible implementations.

### Batch Smoothing (lowess)

```rust
--8<-- "../../examples/lowess/batch_smoothing.rs"
```

### Streaming Smoothing (lowess)

```rust
--8<-- "../../examples/lowess/streaming_smoothing.rs"
```

### Online Smoothing (lowess)

```rust
--8<-- "../../examples/lowess/online_smoothing.rs"
```

---

## Running the Examples

```bash
# Run fastLowess examples (parallel)
cargo run --example fast_batch_smoothing -p ../../examples
cargo run --example fast_streaming_smoothing -p ../../examples
cargo run --example fast_online_smoothing -p ../../examples

# Run lowess examples (single-threaded)
cargo run --example batch_smoothing -p ../../examples
cargo run --example streaming_smoothing -p ../../examples
cargo run --example online_smoothing -p ../../examples
```

## Quick Start

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    // Build and fit the model
    let model = Lowess::new()
        .fraction(0.3)
        .iterations(3)
        .confidence_intervals(0.95)
        .return_diagnostics()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    
    println!("R²: {:.4}", result.diagnostics.unwrap().r_squared);
    Ok(())
}
```
