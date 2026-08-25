# fastLowess & lowess Rust API Reference

The Rust bindings provide the core implementation and high-performance extensions. The API uses a Builder pattern consistent across both the `lowess` (pure Rust) and `fastLowess` (accelerated) crates.

> **StreamingLowess** and **OnlineLowess** are documented separately: [rust-streaming.md](api-streaming.md), [rust-online.md](api-online.md)

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

* Fits the model to the provided `x` and `y` arrays.
* Returns `Result<LowessResult<T>, LowessError>`.

See [rust-streaming.md](api-streaming.md) for `StreamingLowess`.

See [rust-online.md](api-online.md) for `OnlineLowess`.

## Builder Configuration

These chained methods configure the builder. They correspond to the "Options Structures" in other bindings.

### Lowess Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `fraction(T)` | `T: Float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations(usize)` | `usize` | `3` | Number of robustifying iterations |
| `delta(T)` | `T: Float` | `NaN` | Interpolation distance (NaN for auto) |
| `weight_function(...)` | `weight_function` | `"tricube"` | Weight function |
| `robustness_method(...)` | `robustness_method` | `"bisquare"` | Robustness method |
| `scaling_method(...)` | `scaling_method` | `"mad"` | Residual scaling method |
| `boundary_policy(...)` | `boundary_policy` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback(...)` | `zero_weight_fallback` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge(T)` | `T: Float` | `NaN` | Auto-convergence tolerance |
| `confidence_intervals(T)` | `T: Float` | `NaN` | Confidence level (e.g., 0.95) |
| `prediction_intervals(T)` | `T: Float` | `NaN` | Prediction level (e.g., 0.95) |
| `custom_weights(Vec<T>)` | `Vec<T: Float>` | `None` | Per-observation weights (Batch only) |
| `return_diagnostics()` | `bool` | `false` | Compute RMSE, MAE, R², AIC |
| `return_residuals()` | `bool` | `false` | Include residuals in result |
| `return_robustness_weights()` | `bool` | `false` | Include robustness weights in result |
| `return_se()` | `bool` | `false` | Return standard errors |
| `parallel(bool)` | `bool` | `true` | Enable parallel execution |
| `cv_method(str)` | `&str` | `"kfold"` | CV strategy: `"kfold"` or `"loocv"` (defaults to `"kfold"` when `cv_fractions` is provided) |
| `cv_k(usize)` | `usize` | `5` | K for k-fold CV |
| `cv_fractions(Vec<f64>)` | `Vec<f64>` | `None` | Fraction grid for CV |
| `cv_seed(u64)` | `u64` | `None` | RNG seed for CV |
| `backend(...)` | `Backend` | `CPU` | `fastLowess` only: `CPU` or `GPU` |

**Note:** In other language bindings `custom_weights` is a `fit()` argument; in Rust it is a builder step because all configuration lives on the builder and `fit()` consumes `self`.

See [rust-streaming.md](api-streaming.md) for Streaming Options.

See [rust-online.md](api-online.md) for Online Options.

## GPU Acceleration

The `fastLowess` crate provides a GPU-accelerated backend using `wgpu`. This backend is designed for high-throughput processing of large datasets (10k+ points) where parallel regression fitting on the GPU significantly outperforms CPU execution.

### Enabling GPU Support

GPU support is optional and must be enabled via the `gpu` feature in `fastLowess`:

```toml
[dependencies]
fastLowess = { version = "*", features = ["gpu"] }
```

### Usage

To use the GPU backend, configure the builder with `Backend::GPU`:

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let model = Lowess::new()
        .backend(Backend::GPU)
        .confidence_intervals(0.95)
        .build()?;

    Ok(())
}
```

### Supported Features

The GPU backend implements almost the entire LOWESS pipeline in WGSL compute shaders, providing native support for the following features:

* **Weight Functions**: All standard kernels are supported (`Tricube`, `Epanechnikov`, `Gaussian`, `Uniform`, `Biweight`, `Triangle`, `Cosine`).
* **Robustness Methods**: Support for `Bisquare`, `Huber`, and `Talwar` robustness weighting.
* **Scaling Methods**: Residual scaling using `MAD` (Median Absolute Deviation), `MAR` (Median Absolute Residual), and `Mean` (Mean Absolute Residual).
* **Interval Bounds**: GPU-native computation of `Standard Errors`, `Confidence Intervals`, and `Prediction Intervals`.
* **Optimization**:
  * **Parallel Fitting**: Local regression for all anchor points is computed in parallel.
  * **Robustness Loops**: Iterative weight updates and convergence checks occur entirely on the GPU.
  * **Distance-based Skipping**: Support for the `delta` parameter to accelerate smoothing on dense grids.
* **Validation**: GPU-accelerated `K-Fold` and `LOOCV` (Leave-One-Out Cross-Validation).

#### Feature Comparison

| Feature | CPU | GPU (fastLowess) | Notes |
| --- | --- | --- | --- |
| Batch Smoothing | ✅ | ✅ | GPU recommended for N > 10,000 |
| Streaming/Online | ✅ | ❌ | GPU optimized for static batch data |
| All Weight Functions | ✅ | ✅ | Identical numerical implementation |
| Robustness (Bisquare+) | ✅ | ✅ | Full support for all methods |
| Scaling (MAD/MAR/Mean) | ✅ | ✅ | Full support for all methods |
| Boundary Policies | ✅ | ✅ | Extend, Reflect, Zero, NoBoundary |
| Auto-Convergence | ✅ | ✅ | Tolerance checking occurs on GPU |
| Intervals & SE | ✅ | ✅ | Native GPU interval calculation |
| Cross-Validation | ✅ | ✅ | Parallel CV folders on GPU |
| Interpolation (Delta) | ✅ | ✅ | Anchor-based skipping supported |

### Hardware Requirements

The GPU backend leverages `wgpu` and supports:

* **Vulkan** (Linux/Windows)
* **Metal** (macOS/iOS)
* **DirectX 12** (Windows)

It requires a device supporting compute shaders. If no compatible GPU is found at runtime, the initialization will return a `LowessError::RuntimeError`.

### Performance Considerations

The GPU backend is optimized for large datasets (N > 100,000) and provides parallelization through compute shaders. For smaller datasets, the CPU backend is faster.

## Result Structure

See [rust-online.md](api-online.md) for `OnlineOutput<T>`.

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

*See: [Weight Functions](kernels.md)*

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"` (alias: `"boxcar"`)
* `"biweight"` (alias: `"bisquare"`)
* `"triangle"` (alias: `"triangular"`)
* `"cosine"`

### robustness_method

*See: [Robustness](robustness.md)*

* `"bisquare"` (default; alias: `"biweight"`)
* `"huber"`
* `"talwar"`

### boundary_policy

*See: [Boundary Handling](boundary.md)*

* `"extend"` (default; alias: `"pad"`)
* `"reflect"` (alias: `"mirror"`)
* `"zero"`
* `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](scaling.md)*

* `"mad"` (default; alias: `"median_absolute_deviation"`)
* `"mar"` (alias: `"median_absolute_residual"`)
* `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

*See: [Parameters](parameters.md)*

* `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`)
* `"return_original"` (alias: `"original"`)
* `"return_none"` (alias: `"none"`)

### merge_strategy

See [rust-streaming.md](api-streaming.md).

### update_mode

See [rust-online.md](api-online.md).

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
