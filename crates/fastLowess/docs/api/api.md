# fastLowess

The Rust bindings provide the core implementation and high-performance extensions. The API uses a Builder pattern consistent across both the `lowess` (pure Rust) and `fastLowess` (accelerated) crates.

> **StreamingLowess** and **OnlineLowess** are documented separately: [Streaming Adapter](crate::doc::api::streaming), [Online Adapter](crate::doc::api::online)

## When to Use Batch Adapter

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

## Classes

The `fastLowess` crate exposes a `Lowess` wrapper struct that mirrors the class available in other language bindings. It wraps a `LowessBuilder<f64>`, and its `build()` method delegates to the parallel adapter.

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

#### `fit(x, y)`

Fits the model to the provided `x` and `y` arrays. Returns `Result<LowessResult<T>, LowessError>`.

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

## Options Structures

These chained methods configure the builder. They correspond to the "Options Structures" in other bindings.

### Lowess Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `fraction(T)` | `T: Float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations(usize)` | `usize` | `3` | Number of robustifying iterations |
| `delta(T)` | `T: Float` | `NaN` | Interpolation distance (`NaN` auto-sets it to 1% of the x-range) |
| `weight_function(...)` | `weight_function` | `"tricube"` | Weight function |
| `robustness_method(...)` | `robustness_method` | `"bisquare"` | Robustness method |
| `scaling_method(...)` | `scaling_method` | `"mad"` | Residual scaling method |
| `boundary_policy(...)` | `boundary_policy` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback(...)` | `zero_weight_fallback` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge(T)` | `T: Float` | `NaN` | Auto-convergence tolerance |
| `confidence_intervals(T)` | `T: Float` | `NaN` | Confidence level (e.g., 0.95) |
| `prediction_intervals(T)` | `T: Float` | `NaN` | Prediction level (e.g., 0.95) |
| `return_diagnostics()` | `bool` | `false` | Include diagnostics in result |
| `return_residuals()` | `bool` | `false` | Include residuals in result |
| `return_robustness_weights()` | `bool` | `false` | Include weights in result |
| `return_se()` | `bool` | `false` | Return standard errors |
| `return_sorted()` | `bool` | `false` | Return results sorted ascending by `x` instead of in original input order |
| `parallel(bool)` | `bool` | `true` | Enable parallel execution |
| `backend(...)` | `Backend` | `CPU` | `fastLowess` only: `CPU` or `GPU` |
| `cv_method(str)` | `&str` | `"kfold"` | CV strategy: `"kfold"` (fast) or `"loocv"` (slow, exhaustive) — defaults to `"kfold"` when `cv_fractions` is provided |
| `cv_k(usize)` | `usize` | `5` | K for k-fold CV |
| `cv_fractions(Vec<f64>)` | `Vec<f64>` | `None` | Fraction grid for CV |
| `cv_seed(u64)` | `u64` | `None` | RNG seed for CV |
| `custom_weights(Vec<T>)` | `Vec<T: Float>` | `None` | Per-observation weights |

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

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `NaN` (default) auto-sets it to 1% of the x-range. Set it to `0.0` explicitly to disable interpolation and fit every point exactly.

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

### confidence_intervals

*See: [Intervals](crate::doc::guide::intervals)*

Confidence level for the confidence interval around the mean response (e.g. `0.95`). `NaN` (default) disables confidence intervals.

### prediction_intervals

*See: [Intervals](crate::doc::guide::intervals)*

Confidence level for the prediction interval for new observations (e.g. `0.95`). `NaN` (default) disables prediction intervals.

### return_diagnostics

*See: [`Diagnostics`](#diagnosticst)*

Include a `Diagnostics` object (RMSE, MAE, R2, AIC/AICc, effective degrees of freedom) in the result. AIC/AICc/`effective_df` additionally require `return_se(true)` (or confidence/prediction intervals) to be populated, since they depend on hat-matrix statistics.

- `false` (default) — leaves `result.diagnostics` as `None`
- `true` — populates `result.diagnostics`

### return_residuals

Include per-point residuals (`y - fitted`) in the result.

- `false` (default) — leaves `result.residuals` as `None`
- `true` — populates `result.residuals`

### return_robustness_weights

Include the final per-point robustness weights (from the last robustness iteration) in the result.

- `false` (default) — leaves `result.robustness_weights` as `None`
- `true` — populates `result.robustness_weights`

### return_se

*See: [Intervals](crate::doc::guide::intervals)*

Computes hat-matrix statistics (effective degrees of freedom, leverage, delta1/delta2) in addition to standard errors.

### return_sorted

When set to `true`, it reorders every result field (residuals, intervals, etc.) by `x` in an ascending manner, instead of in original input order.
To get both orderings, sort the default result client-side instead of calling `fit()` twice.

### parallel

Enable multi-threaded execution via Rayon.

- `true` (default) — parallelizes the local regression fits across CPU cores
- `false` — forces single-threaded execution (useful for benchmarking or deterministic profiling)

### backend

The `fastLowess` crate provides an optional GPU-accelerated backend using `wgpu`, for high-throughput processing of large datasets (10k+ points). GPU support applies to the batch adapter only. See the [GPU Backend guide](crate::doc::advanced::gpu_backend) for the `gpu` Cargo feature, usage, supported features, and hardware requirements.

- `CPU` (default)
- `GPU` — requires the `gpu` Cargo feature

### CV Options

*See: [Cross-Validation](crate::doc::guide::cross_validation)*

- `cv_method`: `"kfold"` (default) — fast, evaluates each candidate fraction over `cv_k` folds; `"loocv"` — slow, exhaustive leave-one-out cross-validation
- `cv_k`: Number of folds for k-fold CV. Ignored when `cv_method="loocv"`.
- `cv_fractions`: Candidate fractions to evaluate. Cross-validation is disabled unless this is set.
- `cv_seed`: Seed for reproducible k-fold shuffling. `None` (default) uses a random seed.

### custom_weights

*See: [Custom Weights](crate::doc::weighting::custom_weights)*

**Note:** In other language bindings `custom_weights` is a `fit()` argument; in Rust it is a builder step because all configuration lives on the builder and `fit()` consumes `self`.

## Result Structure

### `LowessResult<T>`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Array1<T>` | x values (same order as input) |
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
