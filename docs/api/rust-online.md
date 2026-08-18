# OnlineLowess — Rust API Reference

See also: [fastLowess & lowess Rust API Reference](rust.md)

## Struct

### `OnlineLowess`

Online mode for real-time data.

**Constructor:**

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let mut processor = OnlineLowess::new();

    Ok(())
}
```

**Methods:**

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let mut processor = OnlineLowess::new().build()?;
    let output = processor.add_point(1.0f64, 2.0f64)?;

    Ok(())
}
```

* Adds a single point `(x, y)` to the window.
* Returns `Result<Option<OnlineOutput<T>>, LowessError>`.

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let mut processor = OnlineLowess::new().build()?;
    processor.reset();

    Ok(())
}
```

* Clears the internal window buffer. **Rust-only** — this method is not exposed in other language bindings, where creating a new instance is the idiomatic alternative.

## Builder Options

### Online Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity(usize)` | `usize` | `1000` | Max points in sliding window |
| `min_points(usize)` | `usize` | `3` | Min points before smoothing starts |
| `update_mode(...)` | `update_mode` | `"full"` | Update mode |
| `parallel(bool)` | `bool` | `false` | Enable parallel execution (off by default; online LOWESS fits one point at a time) |

## Result Structure

### `OnlineOutput<T>`

Returned by `add_point()` inside `Option`. Is `None` while the window is still filling.

| Field | Type | Description |
| --- | --- | --- |
| `smoothed` | `T` | Smoothed value for the latest point |
| `std_error` | `Option<T>` | Standard error (if requested) |
| `residual` | `Option<T>` | Residual y − smoothed (if requested) |
| `robustness_weight` | `Option<T>` | Robustness weight (if requested) |
| `iterations_used` | `Option<usize>` | Robustness iterations performed |

## Options

### update_mode

*See: [Execution Modes](../user-guide/adapters.md)*

* `"full"` (default; alias: `"resmooth"`)
* `"incremental"` (alias: `"single"`)
