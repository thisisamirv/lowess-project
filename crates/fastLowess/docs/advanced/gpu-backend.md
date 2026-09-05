<!-- markdownlint-disable MD024 MD033 MD046 -->
# GPU Backend

Run the batch LOWESS fit on the GPU via `wgpu` (Vulkan/Metal/DX12) instead of the CPU.

## Overview

The batch `Lowess` type can execute on a GPU-accelerated backend powered by `wgpu`. It reimplements almost the entire LOWESS pipeline — local regression fitting, robustness iterations, interval bounds, and cross-validation — as WGSL compute shaders, so all anchor points are fit in parallel instead of one at a time on CPU cores.

This is worth enabling for high-throughput processing of large datasets (roughly 10k+ points); for smaller inputs the CPU backend (optionally with `parallel = true`) is typically faster once you account for GPU dispatch overhead. See [BENCHMARKS.md](https://github.com/thisisamirv/lowess-project/blob/main/BENCHMARKS.md) for crossover points measured on real hardware.

> **Batch only.** GPU support applies to the batch `Lowess` type only. `StreamingLowess`/`OnlineLowess` remain CPU-only — the Rust core documents GPU as optimized for static batch data, not incremental chunk/point processing.

GPU support is **opt-in**, gated behind the `gpu` Cargo feature (not enabled by default).

### Supported Features

* **Weight Functions**: All standard kernels (`tricube`, `epanechnikov`, `gaussian`, `uniform`, `biweight`, `triangle`, `cosine`).
* **Robustness Methods**: `bisquare`, `huber`, and `talwar` robustness weighting.
* **Scaling Methods**: Residual scaling using `mad` (Median Absolute Deviation), `mar` (Median Absolute Residual), and `mean` (Mean Absolute Residual).
* **Interval Bounds**: GPU-native computation of standard errors, confidence intervals, and prediction intervals.
* **Optimization**:
  * **Parallel Fitting**: Local regression for all anchor points is computed in parallel.
  * **Robustness Loops**: Iterative weight updates and convergence checks occur entirely on the GPU.
  * **Distance-based Skipping**: Support for the `delta` parameter to accelerate smoothing on dense grids.
* **Validation**: GPU-accelerated `kfold` and `loocv` cross-validation.

### Feature Comparison

| Feature | CPU | GPU | Notes |
| --- | --- | --- | --- |
| Batch fitting | ✅ | ✅ | |
| Streaming/Online | ✅ | ❌ | GPU optimized for static batch data |
| All weight/robustness/scaling methods | ✅ | ✅ | |
| Confidence/prediction intervals | ✅ | ✅ | |
| Cross-validation (k-fold, LOOCV) | ✅ | ✅ | |
| Custom per-observation weights | ✅ | ✅ | |

---

## Checking Availability

Before requesting `backend = "gpu"`, check whether the crate was built with GPU support. Requesting the GPU backend when it isn't available returns a `LowessError` at build time instead of panicking.

```rust

# use fastLowess::prelude::*;
# use std::f64::consts::TAU;
# fn main() -> Result<(), LowessError> {
// The `gpu` feature is a compile-time choice, not a runtime check —
// if the crate was built with `features = ["gpu"]`, `Backend::GPU` is available.

# Ok(())
# }
```

---

## Enabling GPU Support

There is no installer for this crate — enable the `gpu` Cargo feature directly:

```toml
[dependencies]
fastLowess = { version = "*", features = ["gpu"] }
```

---

## Usage

Once GPU support is available, request it by setting the backend option on the batch constructor.

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let model = Lowess::new()
        .backend("gpu")
        .confidence_intervals(0.95)
        .build()?;

    Ok(())
}
```

If GPU support isn't available, requesting `backend = "gpu"` returns `Err(LowessError::RuntimeError(..))` from `.build()` instead of panicking.

## Hardware Requirements

The GPU backend leverages `wgpu` and supports:

* **Vulkan** (Linux/Windows)
* **Metal** (macOS/iOS)
* **DirectX 12** (Windows)

It requires a device supporting compute shaders. If no compatible GPU is found at runtime, model construction returns a `LowessError::RuntimeError`.

## Performance Considerations

The GPU backend is optimized for large datasets (N > 100,000) and provides parallelization through compute shaders. For smaller datasets, the CPU backend is faster.
