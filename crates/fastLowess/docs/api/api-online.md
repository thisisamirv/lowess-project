# OnlineLowess API

See also: [fastLowess](crate::doc::api)

## When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

![Online Adapter](https://raw.githubusercontent.com/thisisamirv/lowess-project/main/crates/fastLowess/assets/diagrams/online_comparison.svg)

## Class

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

#### `add_point`

Adds a single point to the sliding window and returns the smoothed value for that point, or `None` while the window is still filling up (fewer than `min_points` seen so far). Once the window reaches `window_capacity`, each new point evicts the oldest one, so memory stays bounded regardless of how much history has passed through. `update_mode` controls how much work each call does: `"incremental"` re-fits only the newest point, while `"full"` re-smooths the entire window for a more accurate but slower result.

```rust
use fastLowess::prelude::*;
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

- Returns `Result<Option<OnlineOutput<T>>, LowessError>`.

#### `reset`

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let mut processor = OnlineLowess::new().build()?;
    processor.reset();

    Ok(())
}
```

- Clears the internal window buffer. **Rust-only** — this method is not exposed in other language bindings, where creating a new instance is the idiomatic alternative.

## Options Structures

### Online Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `fraction(T)` | `T: Float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations(usize)` | `usize` | `3` | Number of robustifying iterations |
| `delta(T)` | `T: Float` | `NaN` | Interpolation distance (`NaN` auto-sets it to 0.0 in Online, i.e. interpolation disabled) |
| `weight_function(...)` | `weight_function` | `"tricube"` | Weight function name |
| `robustness_method(...)` | `robustness_method` | `"bisquare"` | Robustness method name |
| `scaling_method(...)` | `scaling_method` | `"mad"` | Residual scaling method |
| `boundary_policy(...)` | `boundary_policy` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback(...)` | `zero_weight_fallback` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge(T)` | `T: Float` | `NaN` | Auto-convergence tolerance |
| `return_robustness_weights()` | `bool` | `false` | Include `robustness_weight` in result |
| `window_capacity(usize)` | `usize` | `1000` | Max points in sliding window |
| `min_points(usize)` | `usize` | `2` | Min points before smoothing starts |
| `update_mode(...)` | `update_mode` | `"incremental"` | Update mode (`"full"` or `"incremental"`) |

Confidence/prediction intervals, standard errors, cross-validation, GPU `backend`, `custom_weights`, `return_sorted`, `return_diagnostics()`, `return_residuals()`, and `parallel()` are Batch-only (or Batch/Streaming-only) and not available here; see [fastLowess](crate::doc::api) for those.

## Options

### fraction

`fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

### iterations

`iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

### delta

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `NaN` (default) auto-sets it to `0` in Online mode, i.e. interpolation is disabled and every point is fit exactly.

### weight_function

*See: [Weight Functions](crate::doc::weighting::kernels)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](crate::doc::weighting::robustness)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### scaling_method

*See: [Scaling Methods](crate::doc::weighting::scaling)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### boundary_policy

*See: [Boundary Handling](crate::doc::advanced::boundary)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### auto_converge

*See: [Robustness](crate::doc::weighting::robustness)*

Convergence tolerance for early stopping of robustness iterations. `NaN` (default) disables early stopping.

### return_robustness_weights

Include the robustness weight from the latest point's fit in the result.

- `false` (default) — leaves `output.robustness_weight` as `None`
- `true` — populates `output.robustness_weight`

### window_capacity

Maximum number of most recent points kept in the sliding window; older points are discarded as new ones arrive. Each `add_point()` call costs O(`window_capacity`) rather than growing with total history.

### min_points

Minimum number of points required before smoothing starts. `add_point()` returns `None` until the window reaches this size.

### update_mode

*See: [Execution Modes](crate::doc::guide::adapter_choice)*

| Mode | Alias | Behavior | Speed |
| --- | --- | --- | --- |
| `"incremental"` (default) | `"single"` | Update only affected fits | Faster |
| `"full"` | `"resmooth"` | Recompute entire window | More accurate |

## Result Structure

### `OnlineOutput<T>`

Returned by `add_point()` inside `Option`. Is `None` while the window is still filling.

| Field | Type | Description |
| --- | --- | --- |
| `y` | `T` | Smoothed value for the latest point |
| `standard_error` | `Option<T>` | Always `None` — standard errors require `return_se()`/confidence intervals, which are Batch-only |
| `residual` | `Option<T>` | Residual y − smoothed; always present (there is no `return_residuals()` option for Online) |
| `robustness_weight` | `Option<T>` | Robustness weight, if `return_robustness_weights()` was set |
| `iterations_used` | `Option<usize>` | Robustness iterations performed |

There is no `Diagnostics<T>` or `return_diagnostics()` option for `OnlineLowess`: `OnlineOutput<T>` carries no diagnostics field, since diagnostics like RMSE/R2 need more than one point's worth of history to be meaningful.
