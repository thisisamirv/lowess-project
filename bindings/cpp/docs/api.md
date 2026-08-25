# API

The C++ bindings provide a modern, object-oriented wrapper around the core Rust library, mirroring the Rust API structure.

> **StreamingLowess** and **OnlineLowess** are documented separately: [cpp-streaming.md](api-streaming.md), [cpp-online.md](api-online.md)

## When to Use

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

![Gap Handling](gap_handling.svg)

## Classes

### `fastlowess::Lowess`

The `Lowess` class allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 10;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) { x[i] = i; y[i] = i + 0.1; }

    fastlowess::LowessOptions opts;
    opts.fraction = 0.5;
    fastlowess::Lowess model(opts);
    auto result = model.fit(x, y).value();

    std::cout << "y[0]: " << result.y_vector()[0] << "\n";
    return 0;
}
```

```output
y[0]: 0.528325
```

- `options`: A `LowessOptions` struct containing configuration parameters.

**Methods:**

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    fastlowess::LowessOptions opts;
    opts.fraction = 0.5;
    fastlowess::Lowess model(opts);
    auto result = model.fit(x, y).value();
    std::cout << result.fraction_used() << std::endl;  // 0.5
    std::cout << result.iterations_used() << std::endl;  // -1
    // or with custom weights:
    std::vector<double> weights(x.size(), 1.0);
    auto resultW = model.fit(x, y, weights).value();

    return 0;
}
```

```output
0.5
-1
```

- Fits the model to the provided `x` and `y` data vectors.
- The second overload applies `custom_weights` — non-negative per-observation weights of length `n`. Batch only.
- Returns a `LowessResult` object containing the smoothed values and optional diagnostics.

See [cpp-streaming.md](api-streaming.md) for the `StreamingLowess` class.

See [cpp-online.md](api-online.md) for the `OnlineLowess` class.

## Options Structures

### `LowessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `double` | 0.67 | Smoothing fraction (bandwidth) |
| `iterations` | `int` | 3 | Number of robustifying iterations |
| `delta` | `double` | NaN | Interpolation distance (NaN for auto) |
| `weight_function` | `std::string` | "tricube" | Weight function name |
| `robustness_method` | `std::string` | "bisquare" | Robustness method name |
| `scaling_method` | `std::string` | "mad" | Residual scaling method |
| `boundary_policy` | `std::string` | "extend" | Boundary handling policy |
| `zero_weight_fallback` | `std::string` | "use_local_mean" | Zero-weight handling strategy |
| `auto_converge` | `double` | NaN | Auto-convergence tolerance |
| `confidence_intervals` | `double` | NaN | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `double` | NaN | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `bool` | false | Compute RMSE, MAE, R², AIC |
| `return_residuals` | `bool` | false | Include residuals in result |
| `return_robustness_weights` | `bool` | false | Include robustness weights in result |
| `return_se` | `bool` | false | Return standard errors |
| `parallel` | `bool` | true | Enable parallel execution |
| `backend` | `std::string` | `"cpu"` | Execution backend (`"cpu"` or `"gpu"`); GPU requires the library to be built with the `gpu` Cargo feature (Batch only) |
| `cv_method` | `std::string` | "kfold" | CV method (`"kfold"` or `"loocv"`) (Batch only) |
| `cv_k` | `int` | 5 | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `std::vector<double>` | `{}` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `uint64_t` | `0` | Random seed for CV shuffling (Batch only; 0 = random) |
| `custom_weights` | `std::vector<double>` | `{}` | Per-observation case weights — passed to `fit()`, not the constructor (Batch only) |

See [cpp-streaming.md](api-streaming.md) for `StreamingOptions`.

See [cpp-online.md](api-online.md) for `OnlineOptions`.

## GPU Acceleration

The batch `fastlowess::Lowess` class can run on a GPU-accelerated backend powered by `wgpu`. This backend is designed for high-throughput processing of large datasets (10k+ points) where parallel regression fitting on the GPU significantly outperforms CPU execution.

> GPU support applies to `Lowess` (batch) only. `StreamingLowess`/`OnlineLowess` remain CPU-only — see [rust.md](gpu-backend.md) for why.

### Enabling GPU Support

GPU support is opt-in and **not included in prebuilt releases**. Instead of building from source, call the one-time installer, which downloads a prebuilt GPU-enabled shared library from the matching [GitHub Release](https://github.com/thisisamirv/lowess-project/releases):

```cpp
#include <fastlowess.hpp>

fastlowess::gpu::install(); // prompts for confirmation via curl, then downloads
```

Or non-interactively: `fastlowess::gpu::install(/*yes=*/true)`. Requires `curl` on `PATH` (ships with Linux, macOS, and Windows 10+).

Unlike Python/Julia, a running C++ process cannot swap the backend of a library it already linked against — after downloading, relink/rebuild your application against the downloaded library (or `dlopen`/`LoadLibrary` it manually) and restart. Check with `fastlowess::gpu::available()`.

Alternatively, build from source locally with the `gpu` Cargo feature enabled, then link against it and the platform GPU libraries (Vulkan/Metal/DX12):

```sh
cd bindings/cpp
cargo build --release --features gpu
```

### Usage

To use the GPU backend, set `backend` on `LowessOptions`:

```cpp
#include <fastlowess.hpp>

fastlowess::LowessOptions opts;
opts.fraction = 0.5;
opts.backend = "gpu";
opts.confidence_intervals = 0.95;
fastlowess::Lowess model(opts);
auto result = model.fit(x, y);
```

If the library was not built with the `gpu` feature, requesting `backend = "gpu"` raises a runtime error pointing to `fastlowess::gpu::install()`.

### Supported Features

The GPU backend implements almost the entire LOWESS pipeline in WGSL compute shaders, providing native support for the following features:

- **Weight Functions**: All standard kernels are supported (`"tricube"`, `"epanechnikov"`, `"gaussian"`, `"uniform"`, `"biweight"`, `"triangle"`, `"cosine"`).
- **Robustness Methods**: Support for `"bisquare"`, `"huber"`, and `"talwar"` robustness weighting.
- **Scaling Methods**: Residual scaling using `"mad"` (Median Absolute Deviation), `"mar"` (Median Absolute Residual), and `"mean"` (Mean Absolute Residual).
- **Interval Bounds**: GPU-native computation of standard errors, confidence intervals, and prediction intervals.
- **Optimization**:
  - **Parallel Fitting**: Local regression for all anchor points is computed in parallel.
  - **Robustness Loops**: Iterative weight updates and convergence checks occur entirely on the GPU.
  - **Distance-based Skipping**: Support for the `delta` parameter to accelerate smoothing on dense grids.
- **Validation**: GPU-accelerated `"kfold"` and `"loocv"` cross-validation.

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

- **Vulkan** (Linux/Windows)
- **Metal** (macOS/iOS)
- **DirectX 12** (Windows)

It requires a device supporting compute shaders. If no compatible GPU is found at runtime, model construction raises an error.

### Performance Considerations

The GPU backend is optimized for large datasets (N > 100,000) and provides parallelization through compute shaders. For smaller datasets, the CPU backend (`backend = "cpu"`, the default) is faster.

## Result Structure

See [cpp-online.md](api-online.md) for `OnlineOutput`.

### `fastlowess::LowessResult`

A RAII wrapper around the C result struct `fastlowess_CppLowessResult`.

| Method | Return Type | Description |
| --- | --- | --- |
| `x_vector()` | `std::vector<double>` | Sorted x values |
| `y_vector()` | `std::vector<double>` | Smoothed y values |
| `fraction_used()` | `double` | Fraction used (set or selected by CV) |
| `iterations_used()` | `int` | Robustness iterations actually performed (-1 = N/A) |
| `standard_errors()` | `std::vector<double>` | Per-point standard errors (empty if not computed) |
| `confidence_lower()` | `std::vector<double>` | Lower confidence bounds (empty if not computed) |
| `confidence_upper()` | `std::vector<double>` | Upper confidence bounds (empty if not computed) |
| `prediction_lower()` | `std::vector<double>` | Lower prediction bounds (empty if not computed) |
| `prediction_upper()` | `std::vector<double>` | Upper prediction bounds (empty if not computed) |
| `residuals()` | `std::vector<double>` | Residuals (if `return_residuals`; empty if not computed) |
| `robustness_weights()` | `std::vector<double>` | Robustness weights (if `return_robustness_weights`; empty if not computed) |
| `cv_scores()` | `std::vector<double>` | CV score per tested fraction (empty if CV not run) |
| `diagnostics()` | `Diagnostics` | Fit metrics — check `diagnostics().has_value()` before use (if `return_diagnostics`) |

### `fastlowess::Diagnostics`

All accessors are const methods (not public fields):

| Method | Return Type | Description |
| --- | --- | --- |
| `rmse()` | `double` | Root Mean Squared Error |
| `mae()` | `double` | Mean Absolute Error |
| `r_squared()` | `double` | R-squared |
| `residual_sd()` | `double` | Residual standard deviation |
| `effective_df()` | `double` | Effective degrees of freedom (NaN if not computed) |
| `aic()` | `double` | AIC (NaN if not computed) |
| `aicc()` | `double` | AICc (NaN if not computed) |

## Options

### weight_function

*See: [Weight Functions](kernels.md)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](robustness.md)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### boundary_policy

*See: [Boundary Handling](boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

*See: [Parameters](parameters.md)*

- `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`)
- `"return_original"` (alias: `"original"`)
- `"return_none"` (alias: `"none"`)

### merge_strategy

See [cpp-streaming.md](api-streaming.md).

### update_mode

See [cpp-online.md](api-online.md).

## Example

```cpp
#include <fastlowess.hpp>
#include <iostream>

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2.1, 4.0, 6.2, 8.0, 10.1};

    fastlowess::LowessOptions opts;
    opts.fraction = 0.5;
    
    fastlowess::Lowess model(opts);
    auto expected = model.fit(x, y);

    if (expected.has_value()) {
        auto y_hat = expected.value().y_vector();
        for (double val : y_hat) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
```

```output
2.1 4 6.2 8 10.1
```
