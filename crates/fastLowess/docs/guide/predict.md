<!-- markdownlint-disable MD024 MD033 -->
# Out-of-Sample Prediction

Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

!!! note "Adapter support"
    Out-of-sample prediction is available in **Batch** mode only. Streaming and Online modes do not support it.

`Predict::new()...build()?.call(&result, new_x)` evaluates the local WLS fit at arbitrary `x`-values, like R's `predict(model, newdata)`.

It reuses `fit()`'s own (possibly `delta`-interpolated) smoothed curve for its `y` output — so predicting at a training `x` always exactly reproduces that point's `fit()` output, regardless of `delta()`. A fresh local WLS fit is only run when `"derivative"`, `"se"` (or an interval method), or `max_neighbor_distance` needs the actual regression slope or standard error at the query point.

Requires `.retain_model(true)` on the builder before `fit()`, otherwise `.call(...)` returns `LowessError::PredictionUnavailable`.

---

## Builder Configuration

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `outputs([&str])` | `&[&str]` | `[]` | Select `"se"` and/or `"derivative"` |
| `confidence_intervals(T)` | `T: Float` | disabled | Confidence interval coverage level (e.g. `0.95`) |
| `prediction_intervals(T)` | `T: Float` | disabled | Prediction interval coverage level (e.g. `0.95`) |
| `return_derivative()` | `bool` | `false` | Include the local fit's derivative (slope) at each query point |
| `extrapolation(...)` | `&str` | `"clamp"` | Behavior for query points outside the training `x`-range |
| `max_extrapolation_distance(T)` | `T: Float` | disabled | Under `"linear"` extrapolation, the max allowed distance beyond the training boundary before erroring |
| `max_neighbor_distance(T)` | `T: Float` | disabled | Max allowed distance to the farthest training point in a query's local window before erroring |

## Options

### outputs

Select optional prediction output components with `.outputs(["se", "derivative"])`. `"se"` computes standard errors for each query point, using the retained model's residual scale and per-point leverage. It is required for `confidence_intervals`/`prediction_intervals` to be populated.

### confidence_intervals

Confidence level for the confidence interval around the mean response at each query point (e.g. `0.95`). Uses the same z-score convention as `fit()`'s own confidence intervals. Disabled by default.

### prediction_intervals

Confidence level for the prediction interval for a new observation at each query point (e.g. `0.95`). Widens using the same MAD-based residual scale `fit()` uses for its own intervals. Disabled by default.

`"derivative"` includes the local fit's derivative (slope) at each query point. Unknown names are reported by `.build()` as `LowessError::ParseErrors`.

### extrapolation

Behavior for query points outside `[min(x_train), max(x_train)]`:

| Policy | Behavior |
| --- | --- |
| `"clamp"` (default) | Clamps the query to the nearest boundary window |
| `"linear"` | Linearly extrapolates from the nearest boundary point's local fit and slope |
| `"error"` | Fails the whole call with `LowessError::PredictOutOfRange` |

### max_extrapolation_distance

Under `"linear"` extrapolation, the maximum allowed distance beyond the training boundary before `.call(...)` errors with `LowessError::ExtrapolationTooFar`, instead of returning an unbounded value. Disabled by default (uncapped).

### max_neighbor_distance

Maximum allowed distance to the farthest training point in a query's local window before `.call(...)` errors with `LowessError::SparseNeighborhood`. Guards against an "empty range" blind spot: a query point can fall within `[min(x_train), max(x_train)]` yet still be far from any real training point (e.g. training `x` in `[0,10]` and `[90,100]`, query at `x=50`). Disabled by default (uncapped); applies regardless of `extrapolation`.

## Example

### Basic Usage

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.1_f64, 4.0, 6.2, 8.0, 10.1];

    let model = Lowess::new().fraction(0.7).retain_model(true).build()?;
    let result = model.fit(&x, &y)?;

    let new_x = vec![1.5_f64, 4.5];
    let prediction = Predict::new().build()?.call(&result, &new_x)?;
    println!("Predicted y: {:?}", prediction.y);

    Ok(())
}
```

```output
Predicted y: [3.05, 9.05]
```

### Standard Errors and Derivative

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.1_f64, 4.0, 6.2, 8.0, 10.1];

    let model = Lowess::new().fraction(0.7).retain_model(true).build()?;
    let result = model.fit(&x, &y)?;

    let options = Predict::new().outputs(["se", "derivative"]).build()?;
    let prediction = options.call(&result, &[2.5_f64])?;

    println!("y: {:?}", prediction.y);
    println!("SE: {:?}", prediction.standard_errors);
    println!("Derivative: {:?}", prediction.derivative);

    Ok(())
}
```

```output
y: [5.1]
SE: Some([0.0])
Derivative: Some([2.2])
```

### Linear Extrapolation

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.1_f64, 4.0, 6.2, 8.0, 10.1];

    let model = Lowess::new().fraction(0.7).retain_model(true).build()?;
    let result = model.fit(&x, &y)?;

    let options = Predict::new().extrapolation("linear").build()?;
    let prediction = options.call(&result, &[10.0_f64])?;
    println!("Extrapolated y: {:?}", prediction.y);

    Ok(())
}
```

```output
Extrapolated y: [10.1]
```
