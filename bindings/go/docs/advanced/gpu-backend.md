---
title: "GPU Backend"
weight: 80
---

Run the batch LOWESS fit on the GPU via `wgpu` (Vulkan/Metal/DX12) instead of the CPU.

## Overview

The batch `Lowess` type can execute on a GPU-accelerated backend powered by `wgpu`. It reimplements almost the entire LOWESS pipeline — local regression fitting, robustness iterations, interval bounds, and cross-validation — as WGSL compute shaders, so all anchor points are fit in parallel instead of one at a time on CPU cores.

This is worth enabling for high-throughput processing of large datasets (roughly 10k+ points); for smaller inputs the CPU backend (optionally with `Parallel: true`) is typically faster once you account for GPU dispatch overhead. See [BENCHMARKS.md](https://github.com/thisisamirv/lowess-project/blob/main/BENCHMARKS.md) for crossover points measured on real hardware.

> **Batch only.** GPU support applies to the batch `Lowess` type only. `StreamingLowess`/`OnlineLowess` remain CPU-only — the Rust core documents GPU as optimized for static batch data, not incremental chunk/point processing.

GPU support is **opt-in** and not included in a default build of `fastlowess-go` — it requires building the native library from source with the `gpu` Cargo feature.

### Supported Features

- **Weight Functions**: All standard kernels (`tricube`, `epanechnikov`, `gaussian`, `uniform`, `biweight`, `triangle`, `cosine`).
- **Robustness Methods**: `bisquare`, `huber`, and `talwar` robustness weighting.
- **Scaling Methods**: Residual scaling using `mad` (Median Absolute Deviation), `mar` (Median Absolute Residual), and `mean` (Mean Absolute Residual).
- **Interval Bounds**: GPU-native computation of standard errors, confidence intervals, and prediction intervals.
- **Optimization**:
  - **Parallel Fitting**: Local regression for all anchor points is computed in parallel.
  - **Robustness Loops**: Iterative weight updates and convergence checks occur entirely on the GPU.
  - **Distance-based Skipping**: Support for the `Delta` parameter to accelerate smoothing on dense grids.
- **Validation**: GPU-accelerated `kfold` and `loocv` cross-validation.

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

Before requesting `Backend: "gpu"`, check whether the currently loaded native library was built with GPU support:

```go
if !fastlowess.GPUEnabled() {
 log.Fatal("this build of fastlowess_go was not compiled with the gpu feature")
}
```

Requesting the GPU backend when it isn't available returns a clear error from `NewLowess`, rather than a raw panic.

---

## Building with GPU Support

Unlike the Python, Node.js, and R bindings, the Go binding does not ship a one-time GPU installer. Build the native library from source with the `gpu` Cargo feature instead:

```sh
cargo build -p fastlowess-go --profile release-c --features gpu
```

Since `cgo` links against the static library at build time, rebuild your Go binary (`go build ./...`) after replacing the library with a GPU-enabled one — there's no runtime re-pointing like Julia's `install_gpu()`.

---

## Usage

Once GPU support is available, request it by setting `Backend` on the batch options:

```go
opts := fastlowess.DefaultOptions()
opts.Backend = "gpu"
ci := 0.95
opts.ConfidenceIntervals = &ci

model, err := fastlowess.NewLowess(opts)
if err != nil {
 log.Fatal(err)
}
defer model.Close()
```

If GPU support isn't available, `NewLowess` returns an error pointing at how to build a GPU-enabled library, rather than panicking.

## Hardware Requirements

The GPU backend leverages `wgpu` and supports:

- **Vulkan** (Linux/Windows)
- **Metal** (macOS/iOS)
- **DirectX 12** (Windows)

It requires a device supporting compute shaders. If no compatible GPU is found at runtime, `NewLowess` returns an error.

## Performance Considerations

The GPU backend is optimized for large datasets (N > 100,000) and provides parallelization through compute shaders. For smaller datasets, the CPU backend is faster.
