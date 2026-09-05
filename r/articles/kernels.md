# Weight Functions (Kernels)

## Overview

Weight functions (kernels) determine how neighboring points contribute
to each local fit. Points closer to the target receive higher weights.

![Weight function
comparison](../reference/figures/kernel_comparison.svg)

Weight function comparison

------------------------------------------------------------------------

## Available Kernels

| Kernel           | Efficiency | Smoothness  | Support   |
|------------------|------------|-------------|-----------|
| **Tricube**      | 0.998      | Very smooth | Compact   |
| **Epanechnikov** | 1.000      | Smooth      | Compact   |
| **Gaussian**     | 0.961      | Infinite    | Unbounded |
| **Biweight**     | 0.995      | Very smooth | Compact   |
| **Cosine**       | 0.999      | Smooth      | Compact   |
| **Triangle**     | 0.989      | Moderate    | Compact   |
| **Uniform**      | 0.943      | None        | Compact   |

**Efficiency** = AMISE relative to Epanechnikov (1.0 = optimal)

------------------------------------------------------------------------

## Tricube (Default)

Cleveland’s original choice. Best all-around performance.

``` math
w(u) = (1 - |u|^3)^3
```

**Use when**: Default choice for most applications.

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(weight_function = "tricube")
result <- fit(model, x, y)
cat("Smoothed y[0]:", result$y[1], "\n")
#> Smoothed y[0]: 0.4862648
```

------------------------------------------------------------------------

## Epanechnikov

Theoretically optimal for kernel density estimation.

``` math
w(u) = \frac{3}{4}(1 - u^2)
```

**Use when**: Optimal MSE properties desired.

``` r

model <- Lowess(weight_function = "epanechnikov")
result <- fit(model, x, y)
cat("Smoothed y[0]:", result$y[1], "\n")
#> Smoothed y[0]: 0.4980927
```

------------------------------------------------------------------------

## Gaussian

Infinitely smooth. No boundary effects.

``` math
w(u) = \exp(-u^2/2)
```

**Use when**: Maximum smoothness needed, computational cost acceptable.

``` r

model <- Lowess(weight_function = "gaussian")
result <- fit(model, x, y)
cat("Smoothed y[0]:", result$y[1], "\n")
#> Smoothed y[0]: 0.5254291
```

------------------------------------------------------------------------

## Biweight

Good balance of efficiency and smoothness.

``` math
w(u) = (1 - u^2)^2
```

**Use when**: Alternative to Tricube with slightly different properties.

``` r

model <- Lowess(weight_function = "biweight")
result <- fit(model, x, y)
cat("Smoothed y[0]:", result$y[1], "\n")
#> Smoothed y[0]: 0.482559
```

------------------------------------------------------------------------

## Cosine

Smooth and computationally efficient.

``` math
w(u) = \cos(\pi u / 2)
```

**Use when**: Want smooth kernel with simple form.

``` r

model <- Lowess(weight_function = "cosine")
result <- fit(model, x, y)
cat("Smoothed y[0]:", result$y[1], "\n")
#> Smoothed y[0]: 0.4941485
```

------------------------------------------------------------------------

## Triangle

Simple linear taper.

``` math
w(u) = 1 - |u|
```

**Use when**: Simple, interpretable weights.

``` r

model <- Lowess(weight_function = "triangle")
result <- fit(model, x, y)
cat("Smoothed y[0]:", result$y[1], "\n")
#> Smoothed y[0]: 0.4813591
```

------------------------------------------------------------------------

## Uniform

Equal weights within window. Fastest but least smooth.

``` math
w(u) = 1
```

**Use when**: Speed is critical, smoothness less important.

``` r

model <- Lowess(weight_function = "uniform")
result <- fit(model, x, y)
cat("Smoothed y[0]:", result$y[1], "\n")
#> Smoothed y[0]: 0.537598
```

------------------------------------------------------------------------

## Choosing a Kernel

Choose the first row below whose condition applies:

| Condition                   | Kernel           |
|-----------------------------|------------------|
| Need maximum smoothness     | `"gaussian"`     |
| Default is acceptable       | `"tricube"`      |
| Need optimal asymptotic MSE | `"epanechnikov"` |
| Speed is critical           | `"uniform"`      |
| None of the above           | `"biweight"`     |

> **Recommendation:** Stick with **Tricube** (default) unless you have
> specific requirements. The differences between kernels are usually
> small in practice.

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
#> [1] rfastlowess_4.0.0
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
