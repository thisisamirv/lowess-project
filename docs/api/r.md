# fastLowess R API Reference

The R bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

> **StreamingLowess** and **OnlineLowess** are documented separately: [r-streaming.md](r-streaming.md), [r-online.md](r-online.md)

## Classes

### `Lowess`

The `Lowess` class allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```r
library(rfastlowess)

model <- Lowess(fraction = 0.5, iterations = 3)
print(model)
#> <Lowess Model>
#>   Fraction:          0.5
#>   Iterations:        3
#>   Weight Function:   tricube
#>   Parallel:          TRUE
```

**Methods:**

```r
library(rfastlowess)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + 0.1

model <- Lowess(fraction = 0.5)
result <- fit(model, x, y, custom_weights = NULL)
print(result)
#> <LowessResult>
#>   Points:            100
#>   Fraction Used:     0.5
```

* Fits the model to the provided `x` and `y` numeric vectors.
* Returns a `LowessResult` S3 object containing the smoothed values and optional diagnostics.
* `custom_weights`: Optional numeric vector of per-observation weights. All values must be ≥ 0 and length must match `x`. Batch only.
* `print(model)`: Displays the model configuration.

See [r-streaming.md](r-streaming.md) for the `StreamingLowess` class.

See [r-online.md](r-online.md) for the `OnlineLowess` class.

## Options Structures

### `LowessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `numeric` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `integer` | `3` | Number of robustifying iterations |
| `delta` | `numeric` | `NULL` | Interpolation distance (NULL for auto) |
| `weight_function` | `character` | `"tricube"` | Weight function name |
| `robustness_method` | `character` | `"bisquare"` | Robustness method name |
| `scaling_method` | `character` | `"mad"` | Residual scaling method |
| `boundary_policy` | `character` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `character` | `"use_local_mean"` | Zero-weight handling strategy |
| `auto_converge` | `numeric` | `NULL` | Auto-convergence tolerance |
| `confidence_intervals` | `numeric` | `NULL` | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `numeric` | `NULL` | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `logical` | `FALSE` | Include diagnostics in result |
| `return_residuals` | `logical` | `FALSE` | Include residuals in result |
| `return_robustness_weights` | `logical` | `FALSE` | Include weights in result |
| `return_se` | `logical` | `FALSE` | Return standard errors |
| `parallel` | `logical` | `TRUE` | Enable parallel execution |
| `backend` | `character` | `"cpu"` | Execution backend (`"cpu"` or `"gpu"`); GPU requires the package to be built locally with `WITH_GPU=1` (Batch only) |
| `cv_method` | `character` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) (Batch only) |
| `cv_k` | `integer` | `5L` | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `numeric` | `NULL` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `integer` | `NULL` | Random seed for cross-validation shuffling (Batch only) |
| `custom_weights` | `numeric` | `NULL` | Per-observation case weights — passed to `fit()`, not the constructor (Batch only) |

See [r-streaming.md](r-streaming.md) for `StreamingOptions`.

See [r-online.md](r-online.md) for `OnlineOptions`.

## GPU Acceleration

The batch `Lowess()` constructor can run on a GPU-accelerated backend powered by `wgpu`. This backend is designed for high-throughput processing of large datasets (10k+ points) where parallel regression fitting on the GPU significantly outperforms CPU execution.

> GPU support applies to `Lowess()` (batch) only. `StreamingLowess()`/`OnlineLowess()` remain CPU-only — see [rust.md](rust.md#gpu-acceleration) for why.

### Enabling GPU Support

GPU support is opt-in and **not included in CRAN/Bioconductor releases** (the default build strips `wgpu` and its dependencies to keep the package submission-safe). Instead of building from source, run the one-time installer, which downloads a prebuilt GPU-enabled shared library from the matching [GitHub Release](https://github.com/thisisamirv/lowess-project/releases) and installs it in place of the current (CPU-only) library:

```r
library(rfastlowess)

install_gpu() # prompts for confirmation, then downloads and installs
```

Or non-interactively: `install_gpu(yes = TRUE)`.

A running R session cannot swap an already-loaded shared library, so **restart R** after installing for the change to take effect. Check with `gpu_available()`.

Alternatively, build from source locally with `WITH_GPU=1`:

```sh
make -f bindings/r/Makefile WITH_GPU=1
```

This skips stripping the `wgpu`/`bytemuck`/`pollster`/`futures-intrusive` dependencies from the vendored `fastLowess` crate and passes `--features gpu` to the Rust build.

### Usage

To use the GPU backend, pass `backend = "gpu"` to the constructor:

```r
library(rfastlowess)

model <- Lowess(fraction = 0.5, backend = "gpu", confidence_intervals = 0.95)
result <- fit(model, x, y)
```

If the package was not built with `WITH_GPU=1`, requesting `backend = "gpu"` raises an error pointing to `install_gpu()`.

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

The GPU backend is optimized for large datasets (N > 100,000) and provides parallelization through compute shaders. For smaller datasets, the CPU backend (`backend = "cpu"`, the default) is faster.

## Result Structure

See [r-online.md](r-online.md) for `OnlineOutput`.

### `LowessResult`

An S3 list with class `"LowessResult"` containing:

**Supported S3 Methods:** `print(result)`, `plot(result)`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `numeric` | Sorted x values |
| `y` | `numeric` | Smoothed y values |
| `fraction_used` | `numeric` | Fraction used (set or selected by CV) |
| `iterations_used` | `integer \| NULL` | Robustness iterations actually performed |
| `standard_errors` | `numeric \| NULL` | Per-point standard errors |
| `confidence_lower` | `numeric \| NULL` | Lower confidence bounds |
| `confidence_upper` | `numeric \| NULL` | Upper confidence bounds |
| `prediction_lower` | `numeric \| NULL` | Lower prediction bounds |
| `prediction_upper` | `numeric \| NULL` | Upper prediction bounds |
| `residuals` | `numeric \| NULL` | Residuals (if `return_residuals`) |
| `robustness_weights` | `numeric \| NULL` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `numeric \| NULL` | CV score per tested fraction |
| `diagnostics` | `list \| NULL` | Fit metrics (if `return_diagnostics`) |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `numeric` | Root Mean Squared Error |
| `mae` | `numeric` | Mean Absolute Error |
| `r_squared` | `numeric` | R-squared |
| `residual_sd` | `numeric` | Residual standard deviation |
| `effective_df` | `numeric` | Effective degrees of freedom (NaN if not computed) |
| `aic` | `numeric` | AIC (NaN if not computed) |
| `aicc` | `numeric` | AICc (NaN if not computed) |

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

See [r-streaming.md](r-streaming.md).

### update_mode

See [r-online.md](r-online.md).

## Example

```r
library(rfastlowess)

x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + 0.1

# Configure model
model <- Lowess(fraction = 0.5)

# Fit data
result <- fit(model, x, y)

# Print summary
print(result)
#> <LowessResult>
#>   Points:            100
#>   Fraction Used:     0.5

# Plot result
plot(result)
```
