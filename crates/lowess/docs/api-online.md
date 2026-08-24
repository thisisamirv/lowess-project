# OnlineLowess — Rust API Reference

See also: [lowess & lowess Rust API Reference](rust.md)

## Struct

### `OnlineLowess`

Online mode for real-time data.

**Constructor:**

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let mut processor = OnlineLowess::<f64>::new();

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

    let mut processor = OnlineLowess::new().fraction(0.5f64).window_capacity(50usize).min_points(3usize).build()?;;

    // Returns None until min_points (3) are reached
    let r1 = processor.add_point(x[0], y[0])?;  // None
    let r2 = processor.add_point(x[1], y[1])?;  // None

    // Returns Some(OnlineOutput) once enough points are available
    let r3 = processor.add_point(x[2], y[2])?;
    if let Some(output) = r3 {
        println!("Smoothed value: {}", output.y);
    }

    Ok(())
}
```

```output
Smoothed value: 0.22659245357374927
```

* Adds a single point `(x, y)` to the window.
* Returns `Result<Option<OnlineOutput<T>>, LowessError>`.

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let mut processor = OnlineLowess::<f64>::new().build()?;
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
| `y` | `T` | Smoothed value for the latest point |
| `standard_error` | `Option<T>` | Standard error (if requested) |
| `residual` | `Option<T>` | Residual y − smoothed (if requested) |
| `robustness_weight` | `Option<T>` | Robustness weight (if requested) |
| `iterations_used` | `Option<usize>` | Robustness iterations performed |

## Options

### update_mode

*See: [Execution Modes](../user-guide/adapters.md)*

* `"full"` (default; alias: `"resmooth"`)
* `"incremental"` (alias: `"single"`)
