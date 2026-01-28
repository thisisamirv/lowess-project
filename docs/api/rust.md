# fastLowess & lowess Rust API Reference

The Rust bindings provide the core implementation and high-performance extensions. The API uses a Builder pattern consistent across both the `lowess` (pure Rust) and `fastLowess` (accelerated) crates.

## Structs & Usage

Unlike other bindings which have distinct classes, Rust uses a single `Lowess` builder that produces different model types based on the configured `.adapter()`.

### `Lowess` (Batch)

Standard in-memory smoothing.

**Constructor:**

```rust
let builder = Lowess::new().adapter(Batch); // Batch is default
```

**Methods:**

```rust
let result = model.fit(&x, &y)?;
```

* Fits the model to the provided `x` and `y` arrays.
* Returns `Result<LowessResult<T>, LowessError>`.

### `Lowess` (Streaming)

Streaming mode for large datasets.

**Constructor:**

```rust
let mut processor = Lowess::new().adapter(Streaming);
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

### `Lowess` (Online)

Online mode for real-time data.

**Constructor:**

```rust
let mut processor = Lowess::new().adapter(Online);
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

| Method                        | Argument Type          | Default            | Description                           |
| ----------------------------- | ---------------------- | ------------------ | ------------------------------------- |
| `fraction(T)`                 | `T: Float`             | `0.67`             | Smoothing fraction (bandwidth)        |
| `iterations(usize)`           | `usize`                | `3`                | Number of robustifying iterations     |
| `delta(T)`                    | `T: Float`             | `NaN`              | Interpolation distance (NaN for auto) |
| `weight_function(...)`        | `WeightFunction`       | `Tricube`          | Weight function enum                  |
| `robustness_method(...)`      | `RobustnessMethod`     | `Bisquare`         | Robustness method enum                |
| `scaling_method(...)`         | `ScalingMethod`        | `MAD`              | Residual scaling method enum          |
| `boundary_policy(...)`        | `BoundaryPolicy`       | `Extend`           | Boundary handling policy enum         |
| `zero_weight_fallback(...)`   | `ZeroWeightFallback`   | `UseLocalMean`     | Zero-weight handling enum             |
| `auto_converge(T)`            | `T: Float`             | `NaN`              | Auto-convergence tolerance            |
| `confidence_intervals(T)`     | `T: Float`             | `NaN`              | Confidence level (e.g., 0.95)         |
| `prediction_intervals(T)`     | `T: Float`             | `NaN`              | Prediction level (e.g., 0.95)         |
| `return_diagnostics()`        | -                      | `false`            | Include diagnostics in result         |
| `return_residuals()`          | -                      | `false`            | Include residuals in result           |
| `return_robustness_weights()` | -                      | `false`            | Include weights in result             |
| `parallel(bool)`              | `bool`                 | `true`             | Enable parallel execution             |
| `cross_validate(...)`         | `impl CrossValidation` | `None`             | CV strategy (`KFold`, `LOOCV`)        |
| `backend(...)`                | `Backend`              | `CPU`              | `fastLowess` only: `CPU` or `GPU`     |

### Streaming Options

| Method               | Argument Type   | Default             | Description                |
| -------------------- | --------------- | ------------------- | -------------------------- |
| `chunk_size(usize)`  | `usize`         | `5000`              | Data chunk size            |
| `overlap(usize)`     | `usize`         | `500`               | Overlap size               |
| `merge_strategy(...)`| `MergeStrategy` | `WeightedAverage`   | Merge strategy enum        |

### Online Options

| Method                  | Argument Type | Default       | Description                           |
| ----------------------- | ------------- | ------------- | ------------------------------------- |
| `window_capacity(usize)`| `usize`       | `1000`        | Max window size                       |
| `min_points(usize)`     | `usize`       | `2`           | Min points before smoothing           |
| `update_mode(...)`      | `UpdateMode`  | `Incremental` | Update mode enum                      |

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

| Feature                | CPU          | GPU (fastLowess) | Notes                                     |
| ---------------------- | ------------ | ---------------- | ----------------------------------------- |
| Batch Smoothing        | ✅           | ✅               | GPU recommended for N > 10,000            |
| Streaming/Online       | ✅           | ❌               | GPU optimized for static batch data       |
| All Weight Functions   | ✅           | ✅               | Identical numerical implementation        |
| Robustness (Bisquare+) | ✅           | ✅               | Full support for all methods              |
| Scaling (MAD/MAR/Mean) | ✅           | ✅               | Full support for all methods              |
| Boundary Policies      | ✅           | ✅               | Extend, Reflect, Zero, NoBoundary         |
| Auto-Convergence       | ✅           | ✅               | Tolerance checking occurs on GPU          |
| Intervals & SE         | ✅           | ✅               | Native GPU interval calculation           |
| Cross-Validation       | ✅           | ✅               | Parallel CV folders on GPU                |
| Interpolation (Delta)  | ✅           | ✅               | Anchor-based skipping supported           |

### Hardware Requirements

The GPU backend leverages `wgpu` and supports:

* **Vulkan** (Linux/Windows)
* **Metal** (macOS/iOS)
* **DirectX 12** (Windows)

It requires a device supporting compute shaders. If no compatible GPU is found at runtime, the initialization will return a `LowessError::RuntimeError`.

## Result Structure

### `LowessResult<T>`

| Field                | Type                     | Description               |
| -------------------- | ------------------------ | ------------------------- |
| `x`                  | `Array1<T>`              | Smoothed X coordinates    |
| `y`                  | `Array1<T>`              | Smoothed Y coordinates    |
| `fraction_used`      | `T`                      | Actual fraction used      |
| `residuals`          | `Option<Array1<T>>`      | Residuals (if requested)  |
| `confidence_lower`   | `Option<Array1<T>>`      | Lower CI bounds           |
| `confidence_upper`   | `Option<Array1<T>>`      | Upper CI bounds           |
| `prediction_lower`   | `Option<Array1<T>>`      | Lower PI bounds           |
| `prediction_upper`   | `Option<Array1<T>>`      | Upper PI bounds           |
| `robustness_weights` | `Option<Array1<T>>`      | Robustness weights        |
| `diagnostics`        | `Option<Diagnostics<T>>` | Diagnostic metrics struct |
| `cv_results`         | `Option<CVResults<T>>`   | Cross-validation results  |

### `Diagnostics<T>`

| Field          | Type | Description                 |
| -------------- | ---- | --------------------------- |
| `rmse`         | `T`  | Root Mean Squared Error     |
| `mae`          | `T`  | Mean Absolute Error         |
| `r_squared`    | `T`  | R-squared                   |
| `residual_sd`  | `T`  | Residual standard deviation |
| `effective_df` | `T`  | Effective degrees of freedom|
| `aic`          | `T`  | AIC                         |
| `aicc`         | `T`  | AICc                        |

## Enum Options

### WeightFunction

* `Tricube` (default)
* `Epanechnikov`
* `Gaussian`
* `Uniform`
* `Biweight`
* `Triangle`
* `Cosine`

### RobustnessMethod

* `Bisquare` (default)
* `Huber`
* `Talwar`

### BoundaryPolicy

* `Extend` (default - linear extrapolation)
* `Reflect`
* `Zero`
* `NoBoundary`

### ScalingMethod

* `MAD` (default - Median Absolute Deviation)
* `MAR` (Median Absolute Residual)
* `Mean` (Mean Absolute Residual)

### ZeroWeightFallback

* `UseLocalMean` (default)
* `ReturnOriginal`
* `ReturnNone`

### MergeStrategy (Streaming)

* `WeightedAverage` (default)
* `Average`
* `TakeFirst` (Left)
* `TakeLast` (Right)

### UpdateMode (Online)

* `Incremental` (default)
* `Full`

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
