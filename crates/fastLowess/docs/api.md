# fastLowess

The Rust bindings provide the core implementation and high-performance extensions. The API uses a Builder pattern consistent across both the `lowess` (pure Rust) and `fastLowess` (accelerated) crates.

> **StreamingLowess** and **OnlineLowess** are documented separately: [Streaming Adapter](crate::doc::api::streaming), [Online Adapter](crate::doc::api::online)

## When to Use Batch Adapter

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

## Structs & Usage

The `fastLowess` crate exposes three dedicated wrapper structs — `Lowess`, `StreamingLowess`, and `OnlineLowess` — that mirror the distinct classes available in other language bindings. Each struct wraps a `LowessBuilder<f64>` and its `build()` method delegates to the corresponding parallel adapter.

### `Lowess`

Standard in-memory smoothing (batch, parallel by default).

**Constructor:**

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let builder = Lowess::new(); // Batch is default

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

    let model = Lowess::new().fraction(0.5f64).build()?;
    let result = model.fit(&x, &y)?;
    println!("Fraction used: {}", result.fraction_used);
    println!("Iterations used: {:?}", result.iterations_used);

    Ok(())
}
```

```output
Fraction used: 0.5
Iterations used: None
```

- Fits the model to the provided `x` and `y` arrays.
- Returns `Result<LowessResult<T>, LowessError>`.

See [Streaming Adapter](crate::doc::api::streaming) for `StreamingLowess`.

See [Online Adapter](crate::doc::api::online) for `OnlineLowess`.

## Builder Configuration

These chained methods configure the builder. They correspond to the "Options Structures" in other bindings.

### Lowess Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `fraction(T)` | `T: Float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations(usize)` | `usize` | `3` | Number of robustifying iterations |
| `delta(T)` | `T: Float` | `NaN` | Interpolation distance (`NaN` auto-sets it to 1% of the x-range in Batch, or 0.0 in Streaming/Online) |
| `weight_function(...)` | `weight_function` | `"tricube"` | Weight function |
| `robustness_method(...)` | `robustness_method` | `"bisquare"` | Robustness method |
| `scaling_method(...)` | `scaling_method` | `"mad"` | Residual scaling method |
| `boundary_policy(...)` | `boundary_policy` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback(...)` | `zero_weight_fallback` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge(T)` | `T: Float` | `NaN` | Auto-convergence tolerance |
| `confidence_intervals(T)` | `T: Float` | `NaN` | Confidence level (e.g., 0.95) — see [Intervals](crate::doc::intervals) |
| `prediction_intervals(T)` | `T: Float` | `NaN` | Prediction level (e.g., 0.95) — see [Intervals](crate::doc::intervals) |
| `custom_weights(Vec<T>)` | `Vec<T: Float>` | `None` | Per-observation weights (Batch only) — see [Custom Weights](crate::doc::custom_weights) |
| `return_diagnostics()` | `bool` | `false` | Compute RMSE, MAE, R2, AIC |
| `return_residuals()` | `bool` | `false` | Include residuals in result |
| `return_robustness_weights()` | `bool` | `false` | Include robustness weights in result |
| `return_se()` | `bool` | `false` | Return standard errors |
| `parallel(bool)` | `bool` | `true` | Enable parallel execution |
| `cv_method(str)` | `&str` | `"kfold"` | CV strategy: `"kfold"` (fast) or `"loocv"` (slow, exhaustive) — defaults to `"kfold"` when `cv_fractions` is provided |
| `cv_k(usize)` | `usize` | `5` | K for k-fold CV |
| `cv_fractions(Vec<f64>)` | `Vec<f64>` | `None` | Fraction grid for CV |
| `cv_seed(u64)` | `u64` | `None` | RNG seed for CV |
| `backend(...)` | `Backend` | `CPU` | `fastLowess` only: `CPU` or `GPU` |

`fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

`iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

**Note:** In other language bindings `custom_weights` is a `fit()` argument; in Rust it is a builder step because all configuration lives on the builder and `fit()` consumes `self`.

See [Streaming Adapter](crate::doc::api::streaming) for Streaming Options.

See [Online Adapter](crate::doc::api::online) for Online Options.

## GPU Acceleration

The `fastLowess` crate provides an optional GPU-accelerated backend using `wgpu`, for high-throughput processing of large datasets (10k+ points). GPU support applies to the batch adapter only. See the [GPU Backend guide](crate::doc::gpu_backend) for the `gpu` Cargo feature, usage, supported features, and hardware requirements.

## Result Structure

See [Online Adapter](crate::doc::api::online) for `OnlineOutput<T>`.

### `LowessResult<T>`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Array1<T>` | Sorted x values |
| `y` | `Array1<T>` | Smoothed y values |
| `fraction_used` | `T` | Fraction used (set or selected by CV) |
| `iterations_used` | `Option<usize>` | Robustness iterations actually performed |
| `standard_errors` | `Option<Array1<T>>` | Per-point standard errors |
| `confidence_lower` | `Option<Array1<T>>` | Lower confidence bounds |
| `confidence_upper` | `Option<Array1<T>>` | Upper confidence bounds |
| `prediction_lower` | `Option<Array1<T>>` | Lower prediction bounds |
| `prediction_upper` | `Option<Array1<T>>` | Upper prediction bounds |
| `residuals` | `Option<Array1<T>>` | Residuals (if `return_residuals`) |
| `robustness_weights` | `Option<Array1<T>>` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `Option<Array1<T>>` | CV score per tested fraction |
| `diagnostics` | `Option<Diagnostics<T>>` | Fit metrics (if `return_diagnostics`) |

### `Diagnostics<T>`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `T` | Root Mean Squared Error |
| `mae` | `T` | Mean Absolute Error |
| `r_squared` | `T` | R-squared |
| `residual_sd` | `T` | Residual standard deviation |
| `effective_df` | `Option<T>` | Effective degrees of freedom (`None` if not computed) |
| `aic` | `Option<T>` | AIC (`None` if not computed) |
| `aicc` | `Option<T>` | AICc (`None` if not computed) |

## Options

### weight_function

*See: [Weight Functions](crate::doc::kernels)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](crate::doc::robustness)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### boundary_policy

*See: [Boundary Handling](crate::doc::boundary)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](crate::doc::scaling)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

## Example

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.1_f64, 4.0, 6.2, 8.0, 10.1];

    // Configure model
    let model = Lowess::new()
        .fraction(0.5)
        .iterations(3)
        .build()?;

    // Fit data
    let result = model.fit(&x, &y)?;

    println!("Smoothed Y: {:?}", result.y);
    Ok(())
}
```

```output
Smoothed Y: [2.1, 4.0, 6.2, 8.0, 10.1]
```
