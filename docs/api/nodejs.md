# fastLowess Node.js API Reference

The Node.js bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

> **StreamingLowess** and **OnlineLowess** are documented separately: [nodejs-streaming.md](nodejs-streaming.md), [nodejs-online.md](nodejs-online.md)

## Classes

### `Lowess`

The `Lowess` class allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```javascript
const { Lowess } = require('fastlowess');

const model = new Lowess({ fraction: 0.5, iterations: 3 });
```

* `options`: An object containing `LowessOptions` fields.

**Methods:**

```javascript
const { Lowess } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const model = new Lowess({ fraction: 0.5 });
const result = model.fit(x, y);
console.log(result.fraction_used);  // 0.5
```

* Fits the model to the provided `x` and `y` typed arrays.
* Returns a `LowessResult` object containing the smoothed values and optional diagnostics.

See [nodejs-streaming.md](nodejs-streaming.md) for the `StreamingLowess` class.

See [nodejs-online.md](nodejs-online.md) for the `OnlineLowess` class.

## Options Structures

### `LowessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `number` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `number` | `3` | Number of robustifying iterations |
| `delta` | `number` | `NaN` | Interpolation distance (NaN for auto) |
| `weight_function` | `string` | `"tricube"` | Weight function name |
| `robustness_method` | `string` | `"bisquare"` | Robustness method name |
| `scaling_method` | `string` | `"mad"` | Residual scaling method |
| `boundary_policy` | `string` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `string` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge` | `number` | `null` | Auto-convergence tolerance |
| `confidence_intervals` | `number` | `null` | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `number` | `null` | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `boolean` | `false` | Include diagnostics in result |
| `return_residuals` | `boolean` | `false` | Include residuals in result |
| `return_robustness_weights` | `boolean` | `false` | Include weights in result |
| `return_se` | `boolean` | `false` | Return standard errors |
| `parallel` | `boolean` | `true` | Enable parallel execution |
| `backend` | `string` | `"cpu"` | Execution backend (`"cpu"` or `"gpu"`); GPU requires the package to be built with the `gpu` Cargo feature (Batch only) |
| `cv_method` | `string` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) (Batch only) |
| `cv_k` | `number` | `5` | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `number[]` | `null` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `number` | `null` | Random seed for cross-validation shuffling (Batch only) |
| `custom_weights` | `Float64Array` | `null` | Per-observation case weights — passed to `fit()`, not the options object (Batch only) |

See [nodejs-streaming.md](nodejs-streaming.md) for `StreamingOptions`.

See [nodejs-online.md](nodejs-online.md) for `OnlineOptions`.

## GPU Acceleration

The batch `Lowess` class can run on a GPU-accelerated backend powered by `wgpu`. This backend is designed for high-throughput processing of large datasets (10k+ points) where parallel regression fitting on the GPU significantly outperforms CPU execution.

> GPU support applies to `Lowess` (batch) only. `StreamingLowess`/`OnlineLowess` remain CPU-only — see [rust.md](rust.md#gpu-acceleration) for why.

### Enabling GPU Support

GPU support is opt-in and **not included in prebuilt npm binaries**. Instead of building from source, run the one-time installer, which downloads a prebuilt GPU-enabled native addon from the matching [GitHub Release](https://github.com/thisisamirv/lowess-project/releases) and saves it as `fastlowess.node` next to `index.js` — the same local-override path the loader already checks first, so nothing else needs configuring:

```javascript
const fastlowess = require('fastlowess');

await fastlowess.installGpu(); // prompts for confirmation, then downloads
```

Or non-interactively:

```sh
node -e "require('fastlowess').installGpu({ yes: true })"
# or, via the console script installed alongside the package:
npx fastlowess-install-gpu
```

A running Node.js process cannot swap an already-loaded native addon, so **restart Node.js** after installing for the change to take effect. Check whether the GPU backend is currently active with `fastlowess.gpuAvailable()`.

Alternatively, build from source locally with the `gpu` Cargo feature enabled:

```sh
cd bindings/nodejs
npx napi build --release --features gpu
```

### Usage

To use the GPU backend, pass `backend: "gpu"` in the options object:

```javascript
const { Lowess } = require('fastlowess');

const model = new Lowess({ fraction: 0.5, backend: "gpu", confidence_intervals: 0.95 });
const result = model.fit(x, y);
```

If the addon was not built with the `gpu` feature, requesting `backend: "gpu"` raises a runtime error pointing to `installGpu()`.

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

The GPU backend is optimized for large datasets (N > 100,000) and provides parallelization through compute shaders. For smaller datasets, the CPU backend (`backend: "cpu"`, the default) is faster.

## Result Structure

See [nodejs-online.md](nodejs-online.md) for `OnlineOutput`.

### `LowessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Float64Array` | Sorted x values |
| `y` | `Float64Array` | Smoothed y values |
| `fraction_used` | `number` | Fraction used (set or selected by CV) |
| `iterations_used` | `number \| null` | Robustness iterations actually performed |
| `standard_errors` | `Float64Array \| null` | Per-point standard errors |
| `confidence_lower` | `Float64Array \| null` | Lower confidence bounds |
| `confidence_upper` | `Float64Array \| null` | Upper confidence bounds |
| `prediction_lower` | `Float64Array \| null` | Lower prediction bounds |
| `prediction_upper` | `Float64Array \| null` | Upper prediction bounds |
| `residuals` | `Float64Array \| null` | Residuals (if `return_residuals`) |
| `robustness_weights` | `Float64Array \| null` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `Float64Array \| null` | CV score per tested fraction |
| `diagnostics` | `Diagnostics \| null` | Fit metrics (if `return_diagnostics`) |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `number` | Root Mean Squared Error |
| `mae` | `number` | Mean Absolute Error |
| `r_squared` | `number` | R-squared |
| `residual_sd` | `number` | Residual standard deviation |
| `effective_df` | `number` \| `null` | Effective degrees of freedom |
| `aic` | `number` \| `null` | AIC |
| `aicc` | `number` \| `null` | AICc |

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

See [nodejs-streaming.md](nodejs-streaming.md).

### update_mode

See [nodejs-online.md](nodejs-online.md).

## Example

```javascript
const { Lowess } = require('fastlowess');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

// Configure model
const model = new Lowess({ fraction: 0.5 });

// Fit data
const result = model.fit(x, y);

console.log("Smoothed Y:", result.y);
```
