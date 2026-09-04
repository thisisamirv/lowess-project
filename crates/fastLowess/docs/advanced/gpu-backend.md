<!-- markdownlint-disable MD024 MD033 MD046 -->
# GPU Backend

Run the batch LOWESS fit on the GPU via `wgpu` (Vulkan/Metal/DX12) instead of the CPU.

## Overview

Every binding's batch `Lowess` type can execute on a GPU-accelerated backend powered by `wgpu`. It reimplements almost the entire LOWESS pipeline — local regression fitting, robustness iterations, interval bounds, and cross-validation — as WGSL compute shaders, so all anchor points are fit in parallel instead of one at a time on CPU cores.

This is worth enabling for high-throughput processing of large datasets (roughly 10k+ points); for smaller inputs the CPU backend (optionally with `parallel = true`) is typically faster once you account for GPU dispatch overhead. See [BENCHMARKS.md](https://github.com/thisisamirv/lowess-project/blob/main/BENCHMARKS.md) for crossover points measured on real hardware.

> **Batch only.** GPU support applies to the batch `Lowess` type only. `StreamingLowess`/`OnlineLowess` remain CPU-only in every binding — the Rust core documents GPU as optimized for static batch data, not incremental chunk/point processing.

GPU support is **opt-in** and not included in the default published artifacts (PyPI wheels, npm binaries, CRAN/Bioconductor releases, JLL binaries, prebuilt C++ releases) — it requires either downloading a prebuilt GPU-enabled build via a one-time installer, or building from source with the `gpu` Cargo feature. Both paths are documented per language below.

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

Before requesting `backend = "gpu"` (or the language's equivalent spelling), check whether the currently loaded library was built with GPU support. Requesting the GPU backend when it isn't available raises a clear error pointing at the installer for that language, rather than a raw panic.

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

## Installing GPU Support

Each binding (except Rust, where the `gpu` Cargo feature is enabled directly, and WebAssembly, which does not support the GPU backend) ships a one-time installer that downloads a prebuilt GPU-enabled build from the matching [GitHub Release](https://github.com/thisisamirv/lowess-project/releases) (built by `.github/workflows/release-gpu.yml` for Linux/macOS/Windows x86_64 and macOS arm64). Building from source with the `gpu` Cargo feature is always available as an alternative.

Whether you need to **restart** afterwards depends on how each language loads its native library:

| Language | Restart required? | Why |
| --- | --- | --- |
| Python | Yes | The native extension module is loaded once per process. |
| Node.js | Yes | The native addon (`.node` file) is loaded once per process. |
| R | Yes | The shared library is loaded once per R session. |
| C++ | Yes (relink/rebuild) | Your application links against the library at build/load time. |
| Julia | **No** | `install_gpu()` re-points the internal library reference immediately. |

No installer — enable the `gpu` Cargo feature directly:

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

If GPU support isn't available, requesting `backend = "gpu"` (or the equivalent) raises a runtime error pointing at the installer for that language rather than a raw Rust panic.

## Hardware Requirements

The GPU backend leverages `wgpu` and supports:

* **Vulkan** (Linux/Windows)
* **Metal** (macOS/iOS)
* **DirectX 12** (Windows)

It requires a device supporting compute shaders. If no compatible GPU is found at runtime, model construction returns a `LowessError::RuntimeError`.

## Performance Considerations

The GPU backend is optimized for large datasets (N > 100,000) and provides parallelization through compute shaders. For smaller datasets, the CPU backend is faster.
