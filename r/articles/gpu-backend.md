# GPU Backend

## Overview

The batch `Lowess` type can execute on a GPU-accelerated backend powered
by `wgpu` (Vulkan / Metal / DX12). The GPU backend reimplements almost
the entire LOWESS pipeline — local regression fitting, robustness
iterations, interval bounds, and cross-validation — as compute shaders,
so all anchor points are fitted in parallel.

> **Batch only.** GPU support applies to the batch `Lowess` type only.
> `StreamingLowess` and `OnlineLowess` remain CPU-only.

GPU acceleration is worth enabling for roughly 10K+ points. For smaller
inputs the CPU backend (optionally with `parallel = TRUE`) is typically
faster once GPU dispatch overhead is accounted for.

------------------------------------------------------------------------

## Supported Features on GPU

| Feature                                   | CPU | GPU |
|-------------------------------------------|-----|-----|
| Batch fitting                             | ✅  | ✅  |
| Streaming / Online                        | ✅  | ❌  |
| All weight / robustness / scaling methods | ✅  | ✅  |
| Confidence / prediction intervals         | ✅  | ✅  |
| Cross-validation (k-fold, LOOCV)          | ✅  | ✅  |
| Custom per-observation weights            | ✅  | ✅  |

------------------------------------------------------------------------

## Checking GPU Availability

``` r

library(rfastlowess)

gpu_available()
```

Returns `TRUE` if the currently loaded library was built with GPU
support, `FALSE` otherwise.

------------------------------------------------------------------------

## Installing GPU Support

GPU support is opt-in and not included in standard CRAN / R-universe
builds. Install it with a one-time download:

``` r

library(rfastlowess)

install_gpu()          # interactive — prompts for confirmation
install_gpu(yes = TRUE) # non-interactive
```

[`install_gpu()`](https://thisisamirv.github.io/lowess-project/r/reference/install_gpu.md)
downloads a prebuilt GPU-enabled library from the matching GitHub
release and copies it into the installed package’s `libs/` directory.

**Restart R after installation.**

Alternatively, build from source with GPU support:

``` sh
make -f bindings/r/Makefile WITH_GPU=1
```

------------------------------------------------------------------------

## Using the GPU Backend

Once GPU support is installed, pass `backend = "gpu"` to
[`Lowess()`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md):

``` r

library(rfastlowess)
set.seed(42)
n <- 100000
x <- seq(0, 2 * pi, length.out = n)
y <- sin(x) + rnorm(n, sd = 0.3)

model <- Lowess(
    fraction  = 0.5,
    iterations = 3,
    backend   = "gpu"
)
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## Performance Guidelines

| Scenario                   | Recommendation                          |
|----------------------------|-----------------------------------------|
| n \< 10K                   | CPU (`parallel = TRUE`)                 |
| n 10K–100K                 | Try both; GPU wins for larger fractions |
| n \> 100K                  | GPU backend recommended                 |
| `fraction = 0.5`, n ≥ 50K  | GPU outperforms CPU-parallel            |
| `fraction = 0.1`, n ≥ 250K | GPU crossover point                     |

At n = 1M (`fraction = 0.5`, 3 iterations), the GPU backend is
approximately **6.6×** faster than CPU-parallel.

------------------------------------------------------------------------

## Fallback Behaviour

If `backend = "gpu"` is requested but GPU support is not installed, an
informative error is raised pointing to
[`install_gpu()`](https://thisisamirv.github.io/lowess-project/r/reference/install_gpu.md).
No silent fallback to CPU occurs.
