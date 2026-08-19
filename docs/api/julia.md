# FastLOWESS Julia API Reference

The Julia bindings provide a modern interface to the core Rust library, mirroring the Rust API structure.

> **StreamingLowess** and **OnlineLowess** are documented separately: [julia-streaming.md](julia-streaming.md), [julia-online.md](julia-online.md)

## Classes

### `Lowess`

The `Lowess` struct allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```julia
using FastLOWESS

model = Lowess(fraction=0.5)
```

**Methods:**

```julia
using FastLOWESS

x = collect(range(0, 2π, length=100))
y = sin.(x) .+ 0.1

model = Lowess(fraction=0.5)
result = fit(model, x, y)
println(result.fraction_used)
# 0.5
```

* Fits the model to the provided `x` and `y` data vectors.
* `custom_weights`: Optional per-observation weights. All values must be ≥ 0 and length must match `x`. Batch only.
* Returns a `LowessResult` struct containing the smoothed values and optional diagnostics.

See [julia-streaming.md](julia-streaming.md) for the `StreamingLowess` struct.

See [julia-online.md](julia-online.md) for the `OnlineLowess` struct.

## Options Structures

### `LowessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `Float64` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `Int` | `3` | Number of robustifying iterations |
| `delta` | `Float64` | `NaN` | Interpolation distance (NaN for auto) |
| `weight_function` | `String` | `"tricube"` | Weight function name |
| `robustness_method` | `String` | `"bisquare"` | Robustness method name |
| `scaling_method` | `String` | `"mad"` | Residual scaling method |
| `boundary_policy` | `String` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `String` | `"use_local_mean"` | Zero-weight handling strategy |
| `auto_converge` | `Float64` | `NaN` | Auto-convergence tolerance |
| `confidence_intervals` | `Float64` | `NaN` | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `Float64` | `NaN` | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `Bool` | `false` | Include diagnostics in result |
| `return_residuals` | `Bool` | `false` | Include residuals in result |
| `return_robustness_weights` | `Bool` | `false` | Include weights in result |
| `return_se` | `Bool` | `false` | Return standard errors |
| `parallel` | `Bool` | `true` | Enable parallel execution |
| `backend` | `String` | `"cpu"` | Execution backend (`"cpu"` or `"gpu"`); GPU requires the library to be built with the `gpu` Cargo feature (Batch only) |
| `cv_method` | `String` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) (Batch only) |
| `cv_k` | `Int` | `5` | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `Vector{Float64}` | `Float64[]` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `Union{Int, Nothing}` | `nothing` | Random seed for cross-validation shuffling (Batch only) |
| `custom_weights` | `Union{Vector{Float64}, Nothing}` | `nothing` | Per-observation case weights — passed to `fit()`, not the constructor (Batch only) |

See [julia-streaming.md](julia-streaming.md) for `StreamingOptions`.

See [julia-online.md](julia-online.md) for `OnlineOptions`.

## GPU Acceleration

The batch `Lowess` constructor can run on a GPU-accelerated backend powered by `wgpu`. This backend is designed for high-throughput processing of large datasets (10k+ points) where parallel regression fitting on the GPU significantly outperforms CPU execution.

> GPU support applies to `Lowess` (batch) only. `StreamingLowess`/`OnlineLowess` remain CPU-only — see [rust.md](rust.md#gpu-acceleration) for why.

### Enabling GPU Support

GPU support is opt-in and **not included in prebuilt JLL binaries**. Build the shared library locally with the `gpu` Cargo feature enabled:

```sh
cd bindings/julia
cargo build --release --features gpu
```

`FastLOWESS.jl` auto-detects the freshly built library under `target/release/`. If it is not picked up automatically, point at it explicitly:

```sh
export FASTLOWESS_LIB=/path/to/lowess-project/target/release/libfastlowess_jl.so  # .dylib on macOS, .dll on Windows
```

### Usage

To use the GPU backend, pass `backend="gpu"` to the constructor:

```julia
using FastLOWESS

model = Lowess(fraction=0.5, backend="gpu", confidence_intervals=0.95)
result = fit(model, x, y)
```

If the library was not built with the `gpu` feature, requesting `backend="gpu"` raises an error explaining how to enable it.

### Supported Features

The GPU backend implements almost the entire LOWESS pipeline in WGSL compute shaders, providing native support for the following features:

* **Weight Functions**: All standard kernels are supported (`"tricube"`, `"epanechnikov"`, `"gaussian"`, `"uniform"`, `"biweight"`, `"triangle"`, `"cosine"`).
* **Robustness Methods**: Support for `"bisquare"`, `"huber"`, and `"talwar"` robustness weighting.
* **Scaling Methods**: Residual scaling using `"mad"` (Median Absolute Deviation), `"mar"` (Median Absolute Residual), and `"mean"` (Mean Absolute Residual).
* **Interval Bounds**: GPU-native computation of standard errors, confidence intervals, and prediction intervals.
* **Optimization**:
  * **Parallel Fitting**: Local regression for all anchor points is computed in parallel.
  * **Robustness Loops**: Iterative weight updates and convergence checks occur entirely on the GPU.
  * **Distance-based Skipping**: Support for the `delta` parameter to accelerate smoothing on dense grids.
* **Validation**: GPU-accelerated `"kfold"` and `"loocv"` cross-validation.

#### Feature Comparison

| Feature | CPU | GPU | Notes |
| --- | --- | --- | --- |
| Batch Smoothing | ✅ | ✅ | GPU recommended for N > 10,000 |
| Streaming/Online | ✅ | ❌ | GPU optimized for static batch data |
| All Weight Functions | ✅ | ✅ | Identical numerical implementation |
| Robustness (bisquare+) | ✅ | ✅ | Full support for all methods |
| Scaling (mad/mar/mean) | ✅ | ✅ | Full support for all methods |
| Boundary Policies | ✅ | ✅ | extend, reflect, zero, noboundary |
| Auto-Convergence | ✅ | ✅ | Tolerance checking occurs on GPU |
| Intervals & SE | ✅ | ✅ | Native GPU interval calculation |
| Cross-Validation | ✅ | ✅ | Parallel CV folds on GPU |
| Interpolation (Delta) | ✅ | ✅ | Anchor-based skipping supported |

### Hardware Requirements

The GPU backend leverages `wgpu` and supports:

* **Vulkan** (Linux/Windows)
* **Metal** (macOS/iOS)
* **DirectX 12** (Windows)

It requires a device supporting compute shaders. If no compatible GPU is found at runtime, model construction raises an error.

### Performance Considerations

The GPU backend is optimized for large datasets (N > 100,000) and provides parallelization through compute shaders. For smaller datasets, the CPU backend (`backend="cpu"`, the default) is faster.

## Result Structure

See [julia-online.md](julia-online.md) for `OnlineOutput`.

### `LowessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Vector{Float64}` | Sorted x values |
| `y` | `Vector{Float64}` | Smoothed y values |
| `fraction_used` | `Float64` | Fraction used (set or selected by CV) |
| `iterations_used` | `Union{Int, Nothing}` | Robustness iterations actually performed |
| `standard_errors` | `Union{Vector{Float64}, Nothing}` | Per-point standard errors |
| `confidence_lower` | `Union{Vector{Float64}, Nothing}` | Lower confidence bounds |
| `confidence_upper` | `Union{Vector{Float64}, Nothing}` | Upper confidence bounds |
| `prediction_lower` | `Union{Vector{Float64}, Nothing}` | Lower prediction bounds |
| `prediction_upper` | `Union{Vector{Float64}, Nothing}` | Upper prediction bounds |
| `residuals` | `Union{Vector{Float64}, Nothing}` | Residuals (if `return_residuals`) |
| `robustness_weights` | `Union{Vector{Float64}, Nothing}` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `Union{Vector{Float64}, Nothing}` | CV score per tested fraction |
| `diagnostics` | `Union{Diagnostics, Nothing}` | Fit metrics (if `return_diagnostics`) |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `Float64` | Root Mean Squared Error |
| `mae` | `Float64` | Mean Absolute Error |
| `r_squared` | `Float64` | R-squared |
| `residual_sd` | `Float64` | Residual standard deviation |
| `effective_df` | `Float64` | Effective degrees of freedom (NaN if not computed) |
| `aic` | `Float64` | AIC (NaN if not computed) |
| `aicc` | `Float64` | AICc (NaN if not computed) |

## Options

### weight_function

*See: [Weight Functions](../user-guide/kernels.md)*

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"` (alias: `"boxcar"`)
* `"biweight"` (alias: `"bisquare"`)
* `"triangle"` (alias: `"triangular"`)
* `"cosine"`

### robustness_method

*See: [Robustness](../user-guide/robustness.md)*

* `"bisquare"` (default; alias: `"biweight"`)
* `"huber"`
* `"talwar"`

### boundary_policy

*See: [Boundary Handling](../user-guide/boundary.md)*

* `"extend"` (default; alias: `"pad"`)
* `"reflect"` (alias: `"mirror"`)
* `"zero"`
* `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](../user-guide/scaling.md)*

* `"mad"` (default; alias: `"median_absolute_deviation"`)
* `"mar"` (alias: `"median_absolute_residual"`)
* `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

*See: [Parameters](../user-guide/parameters.md)*

* `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`)
* `"return_original"` (alias: `"original"`)
* `"return_none"` (alias: `"none"`)

### merge_strategy

See [julia-streaming.md](julia-streaming.md).

### update_mode

See [julia-online.md](julia-online.md).

## Example

```julia
using FastLOWESS

x = collect(range(0, 2π, length=100))
y = sin.(x) .+ 0.1

# Configure model
model = Lowess(fraction=0.5, iterations=3)

# Fit data (throws on error)
result = fit(model, x, y)

println("Smoothed Y: ", result.y)
```
