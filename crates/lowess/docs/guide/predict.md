<!-- markdownlint-disable MD024 MD033 -->
# Out-of-Sample Prediction

Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

!!! note "Adapter support"
    Out-of-sample prediction is available in **Batch** mode only. Streaming and Online modes do not support `predict()`.

`predict()` evaluates the local WLS fit at arbitrary `x`-values, similar to R's `predict(model, newdata)`. It always fits an exact local regression at each query point, unlike `fit()` with the default `delta > 0` (which only fits exactly at anchor points and linearly interpolates the rest) — so predicting at an `x` already in the training set may not exactly reproduce that point's `fit()` output unless `delta(0.0)` was used.

Enable it by calling `.retain_model(true)` on the builder before `fit()`; this retains the fitted model's (boundary-padded) training data, smoothed values, final robustness weights, and residual SD. Calling `predict()` without `.retain_model(true)` returns `LowessError::PredictionUnavailable`.

## Basic Usage

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.1_f64, 4.0, 6.2, 8.0, 10.1];

    let model = Lowess::new().fraction(0.7).retain_model(true).build()?;
    let result = model.fit(&x, &y)?;

    let new_x = vec![1.5_f64, 4.5];
    let prediction = result.predict(&new_x, PredictOptions::default())?;
    println!("Predicted y: {:?}", prediction.y);

    Ok(())
}
```

```output
Predicted y: [3.05, 9.05]
```

---

## Options

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `return_se` | `bool` | `false` | Include standard errors in the output |
| `confidence_level` | `Option<T>` | `None` | Confidence interval coverage level (e.g. `Some(0.95)`) |
| `prediction_level` | `Option<T>` | `None` | Prediction interval coverage level (e.g. `Some(0.95)`) |
| `return_derivative` | `bool` | `false` | Include the local fit's derivative (slope) at each query point |
| `extrapolation` | `str` or `ExtrapolationPolicy` | `"clamp"` | Behavior for query points outside the training `x`-range |
| `max_extrapolation_distance` | `T` | none | Under `"linear"` extrapolation, the max allowed distance beyond the training boundary before erroring |
| `max_neighbor_distance` | `T` | none | Max allowed distance to the farthest training point in a query's local window before erroring |

`PredictOptions` is configured the same way as the `Lowess` builder itself: chained setter methods, starting from `PredictOptions::default()`. Standard errors/intervals use the same z-score convention as `fit()`'s existing intervals.

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.1_f64, 4.0, 6.2, 8.0, 10.1];

    let model = Lowess::new().fraction(0.7).retain_model(true).build()?;
    let result = model.fit(&x, &y)?;

    let options = PredictOptions::default().return_se().return_derivative();
    let prediction = result.predict(&[2.5_f64], options)?;

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

---

## Extrapolation Policies

Behavior for query points outside `[min(x_train), max(x_train)]`:

| Policy | Behavior |
| --- | --- |
| `Clamp` (default) | Clamps the query to the nearest boundary window |
| `Linear` | Linearly extrapolates from the nearest boundary point's local fit and slope |
| `Error` | Fails the whole call with `LowessError::PredictOutOfRange` |

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.1_f64, 4.0, 6.2, 8.0, 10.1];

    let model = Lowess::new().fraction(0.7).retain_model(true).build()?;
    let result = model.fit(&x, &y)?;

    let options = PredictOptions::default().extrapolation("linear");
    let prediction = result.predict(&[10.0_f64], options)?;
    println!("Extrapolated y: {:?}", prediction.y);

    Ok(())
}
```

```output
Extrapolated y: [10.1]
```

Under `Linear`, `.max_extrapolation_distance(...)` caps how far beyond the boundary the extrapolation may extend before `predict()` fails with `LowessError::ExtrapolationTooFar`, instead of returning an unbounded value.

`.max_neighbor_distance(...)` guards a separate blind spot: a query point can fall within `[min(x_train), max(x_train)]` yet still be far from any real training point (e.g. training `x` in `[0,10]` and `[90,100]`, query at `x=50`). Setting it makes `predict()` fail with `LowessError::SparseNeighborhood` instead of silently predicting there. It applies regardless of `extrapolation`.

---

## Availability

!!! warning "Batch Mode Only"
    Out-of-sample prediction is only available in **Batch** mode. Streaming and Online modes do not support `predict()`.

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| `retain_model` | ✓ | ✗ | ✗ |
| `predict()` | ✓ | ✗ | ✗ |
