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
#> [1] FALSE
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

``` r

sessionInfo()
#> R version 4.6.1 (2026-06-24)
#> Platform: x86_64-pc-linux-gnu
#> Running under: Ubuntu 24.04.4 LTS
#> 
#> Matrix products: default
#> BLAS:   /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 
#> LAPACK: /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so;  LAPACK version 3.12.0
#> 
#> locale:
#>  [1] LC_CTYPE=C.UTF-8       LC_NUMERIC=C           LC_TIME=C.UTF-8       
#>  [4] LC_COLLATE=C.UTF-8     LC_MONETARY=C.UTF-8    LC_MESSAGES=C.UTF-8   
#>  [7] LC_PAPER=C.UTF-8       LC_NAME=C              LC_ADDRESS=C          
#> [10] LC_TELEPHONE=C         LC_MEASUREMENT=C.UTF-8 LC_IDENTIFICATION=C   
#> 
#> time zone: UTC
#> tzcode source: system (glibc)
#> 
#> attached base packages:
#> [1] stats     graphics  grDevices utils     datasets  methods   base     
#> 
#> other attached packages:
#> [1] rfastlowess_3.2.0
#> 
#> loaded via a namespace (and not attached):
#>  [1] digest_0.6.39     desc_1.4.3        R6_2.6.1          fastmap_1.2.0    
#>  [5] xfun_0.60         cachem_1.1.0      knitr_1.51        htmltools_0.5.9  
#>  [9] rmarkdown_2.32    lifecycle_1.0.5   cli_3.6.6         sass_0.4.10      
#> [13] pkgdown_2.2.1     textshaping_1.0.5 jquerylib_0.1.4   systemfonts_1.3.2
#> [17] compiler_4.6.1    tools_4.6.1       ragg_1.5.2        bslib_0.12.0     
#> [21] evaluate_1.0.5    yaml_2.3.12       otel_0.2.0        jsonlite_2.0.0   
#> [25] rlang_1.3.0       fs_2.1.0          htmlwidgets_1.6.4
```
