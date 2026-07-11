# fastLowess & lowess Rust API Reference

The Rust bindings provide the core implementation and high-performance extensions. The API uses a Builder pattern consistent across both the `lowess` (pure Rust) and `fastLowess` (accelerated) crates.

## Structs & Usage

The `fastLowess` crate exposes three dedicated wrapper structs — `Lowess`, `StreamingLowess`, and `OnlineLowess` — that mirror the distinct classes available in other language bindings. Each struct wraps a `LowessBuilder<f64>` and its `build()` method delegates to the corresponding parallel adapter.

### `Lowess`

Standard in-memory smoothing (batch, parallel by default).

**Constructor:**

```rust
let builder = Lowess::new(); // Batch is default
```

**Methods:**

```rust
let result = model.fit(&x, &y)?;
```

* Fits the model to the provided `x` and `y` arrays.
* Returns `Result<LowessResult<T>, LowessError>`.

### `StreamingLowess`

Streaming mode for large datasets.

**Constructor:**

```rust
let mut processor = StreamingLowess::new();
```

**Methods:**

```rust
let result = processor.process_chunk(&x, &y)?;
```

* Processes a chunk of data. Returns `LowessResult<T>` with partial results.

```rust
let final_result = processor.finalize()?;
```

* Finalizes processing and returns remaining buffered results.

### `OnlineLowess`

Online mode for real-time data.

**Constructor:**

```rust
let mut processor = OnlineLowess::new();
```

**Methods:**

```rust
let output = processor.add_point(x, y)?;
```

* Adds a single point `(x, y)` to the window.
* Returns `Result<Option<OnlineOutput<T>>, LowessError>`.

```rust
processor.reset();
```

* Clears the internal window buffer.

## Builder Configuration

These chained methods configure the builder. They correspond to the "Options Structures" in other bindings.

### Lowess Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `fraction(T)` | `T: Float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations(usize)` | `usize` | `3` | Number of robustifying iterations |
| `delta(T)` | `T: Float` | `NaN` | Interpolation distance (NaN for auto) |
| `weight_function(...)` | `weight_function` | `Tricube` | Weight function enum |
| `robustness_method(...)` | `robustness_method` | `Bisquare` | Robustness method enum |
| `scaling_method(...)` | `scaling_method` | `MAD` | Residual scaling method enum |
| `boundary_policy(...)` | `boundary_policy` | `Extend` | Boundary handling policy enum |
| `zero_weight_fallback(...)` | `zero_weight_fallback` | `UseLocalMean` | Zero-weight handling enum |
| `auto_converge(T)` | `T: Float` | `NaN` | Auto-convergence tolerance |
| `confidence_intervals(T)` | `T: Float` | `NaN` | Confidence level (e.g., 0.95) |
| `prediction_intervals(T)` | `T: Float` | `NaN` | Prediction level (e.g., 0.95) |
| `custom_weights(Vec<T>)` | `Vec<T: Float>` | `None` | Per-observation weights (Batch only) |
| `return_diagnostics()` | `bool` | `false` | Compute RMSE, MAE, R², AIC |
| `return_residuals()` | `bool` | `false` | Include residuals in result |
| `return_robustness_weights()` | `bool` | `false` | Include robustness weights in result |
| `return_se()` | `bool` | `false` | Return standard errors |
| `parallel(bool)` | `bool` | `true` | Enable parallel execution |
| `cv_method(str)` | `&str` | `None` | CV strategy: `"kfold"` or `"loocv"` |
| `cv_k(usize)` | `usize` | `5` | K for k-fold CV |
| `cv_fractions(Vec<f64>)` | `Vec<f64>` | `None` | Fraction grid for CV |
| `cv_seed(u64)` | `u64` | `None` | RNG seed for CV |
| `backend(...)` | `Backend` | `CPU` | `fastLowess` only: `CPU` or `GPU` |

### Streaming Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size(usize)` | `usize` | `5000` | Data chunk size |
| `overlap(usize)` | `usize` | `500` | Overlap size |
| `merge_strategy(...)` | `merge_strategy` | `WeightedAverage` | Merge strategy enum |

### Online Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity(usize)` | `usize` | `1000` | Max points in sliding window |
| `min_points(usize)` | `usize` | `3` | Min points before smoothing starts |
| `update_mode(...)` | `update_mode` | `Full` | Update mode enum |
| `parallel(bool)` | `bool` | `false` | Enable parallel execution (off by default; online LOWESS fits one point at a time) |

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
let model = Lowess::new()
    .backend(Backend::GPU)
    .confidence_intervals(0.95)
    .build()?;
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

### `OnlineOutput<T>`

Returned by `add_point()` inside `Option`. Is `None` while the window is still filling.

| Field | Type | Description |
| --- | --- | --- |
| `smoothed` | `T` | Smoothed value for the latest point |
| `std_error` | `Option<T>` | Standard error (if requested) |
| `residual` | `Option<T>` | Residual y − smoothed (if requested) |
| `robustness_weight` | `Option<T>` | Robustness weight (if requested) |
| `iterations_used` | `Option<usize>` | Robustness iterations performed |

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
| `effective_df` | `Option<T>` | Effective degrees of freedom (NaN if not computed) |
| `aic` | `Option<T>` | AIC (NaN if not computed) |
| `aicc` | `Option<T>` | AICc (NaN if not computed) |

## Options

### weight_function

* `Tricube` (default; string alias: `"tricube"`)
* `Epanechnikov` (alias: `"epanechnikov"`)
* `Gaussian` (alias: `"gaussian"`)
* `Uniform` (aliases: `"uniform"`, `"boxcar"`)
* `Biweight` (aliases: `"biweight"`, `"bisquare"`)
* `Triangle` (aliases: `"triangle"`, `"triangular"`)
* `Cosine` (alias: `"cosine"`)

### robustness_method

* `Bisquare` (default; aliases: `"bisquare"`, `"biweight"`)
* `Huber` (alias: `"huber"`)
* `Talwar` (alias: `"talwar"`)

### boundary_policy

* `Extend` (default; aliases: `"extend"`, `"pad"`)
* `Reflect` (aliases: `"reflect"`, `"mirror"`)
* `Zero` (alias: `"zero"`)
* `NoBoundary` (aliases: `"noboundary"`, `"none"`)

### scaling_method

* `MAD` (default; aliases: `"mad"`, `"median_absolute_deviation"`)
* `MAR` (aliases: `"mar"`, `"median_absolute_residual"`)
* `Mean` (aliases: `"mean"`, `"mean_absolute_residual"`)

### zero_weight_fallback

* `UseLocalMean` (default; aliases: `"use_local_mean"`, `"local_mean"`, `"mean"`)
* `ReturnOriginal` (aliases: `"return_original"`, `"original"`)
* `ReturnNone` (aliases: `"return_none"`, `"none"`)

### merge_strategy

* `WeightedAverage` (default; aliases: `"weighted_average"`, `"weighted"`)
* `Average` (aliases: `"average"`, `"mean"`)
* `TakeFirst` (aliases: `"take_first"`, `"first"`)
* `TakeLast` (aliases: `"take_last"`, `"last"`)

### update_mode

* `Full` (default; aliases: `"full"`, `"resmooth"`)
* `Incremental` (aliases: `"incremental"`, `"single"`)

## Example

```rust
use fastLowess::prelude::*;
use ndarray::array;

fn main() -> Result<(), LowessError> {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![2.1, 4.0, 6.2, 8.0, 10.1];

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
