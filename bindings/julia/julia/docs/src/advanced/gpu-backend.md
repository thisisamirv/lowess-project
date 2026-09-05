# GPU Backend

Run the batch LOWESS fit on the GPU via `wgpu` (Vulkan/Metal/DX12) instead of the CPU.

## Overview

The batch `Lowess` type can execute on a GPU-accelerated backend powered by `wgpu`. It reimplements almost the entire LOWESS pipeline — local regression fitting, robustness iterations, interval bounds, and cross-validation — as WGSL compute shaders, so all anchor points are fit in parallel instead of one at a time on CPU cores.

This is worth enabling for high-throughput processing of large datasets (roughly 10k+ points); for smaller inputs the CPU backend (optionally with `parallel = true`) is typically faster once you account for GPU dispatch overhead. See [BENCHMARKS.md](https://github.com/thisisamirv/lowess-project/blob/main/BENCHMARKS.md) for crossover points measured on real hardware.

> **Batch only.** GPU support applies to the batch `Lowess` type only. `StreamingLowess`/`OnlineLowess` remain CPU-only.

GPU support is **opt-in** and not included in the default JLL binary — download a prebuilt GPU-enabled build via the one-time installer below, or build from source with the `gpu` Cargo feature.

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

Before requesting `backend = "gpu"`, check whether the currently loaded library was built with GPU support. Requesting the GPU backend when it isn't available raises a clear error pointing at `install_gpu()`, rather than a raw panic.

```julia
using FastLOWESS

gpu_available()
```

---

## Installing GPU Support

This binding ships a one-time installer that downloads a prebuilt GPU-enabled build from the [`gpu-builds` release](https://github.com/thisisamirv/lowess-project/releases/tag/gpu-builds) (built by `.github/workflows/release-gpu.yml`) — a single perpetual release holding GPU artifacts for every version, so individual version release pages stay uncluttered; the source version is embedded in each asset's filename instead. Building from source with the `gpu` Cargo feature is always available as an alternative.

```julia
using FastLOWESS

install_gpu()  # prompts for confirmation, then downloads and activates the GPU library
```

Non-interactively: `install_gpu(yes=true)`. **Restart Julia afterwards** — `ccall`'s dynamic-library resolution caches a resolved function pointer per call site after its first use, so code already run in the current session keeps calling into the old library even after `install_gpu()` updates the reference. To use it automatically in future sessions, set the printed path as `ENV["FASTLOWESS_LIB"]` in your Julia startup config.

Build from source instead:

```sh
cd bindings/julia
cargo build --release --features gpu
```

---

## Usage

Once GPU support is available, request it by setting the backend option on the batch constructor.

```julia
using FastLOWESS

model = Lowess(fraction=0.5, backend="gpu", confidence_intervals=0.95)
result = fit(model, x, y)
```

If GPU support isn't available, requesting `backend = "gpu"` raises a runtime error pointing at `install_gpu()` rather than a raw Rust panic.

## See also
