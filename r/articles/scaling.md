# Scaling Methods

## Overview

When `iterations > 0`, LOWESS computes robustness weights by comparing
each residual to the current residual scale estimate. The
`scaling_method` parameter controls how that scale is measured.

The robustness weight for point $`i`$ is:

``` math
w_i = B\!\left(\frac{|r_i|}{6 \cdot \hat{\sigma}}\right)
```

where $`B`$ is the bisquare function and $`\hat{\sigma}`$ is the scale
estimate. A larger $`\hat{\sigma}`$ makes the algorithm more tolerant of
large residuals; a smaller one makes it more aggressive.

| Method   | Formula                  | Robustness  | Speed    |
|----------|--------------------------|-------------|----------|
| `"mad"`  | Median \|r − median(r)\| | Very robust | Moderate |
| `"mar"`  | Median of \|residuals\|  | Robust      | Fast     |
| `"mean"` | Mean of \|residuals\|    | Less robust | Fastest  |

![Scaling method
comparison](../reference/figures/scaling_comparison.svg)

Scaling method comparison

------------------------------------------------------------------------

## MAD — Median Absolute Deviation (Default)

``` math
\hat{\sigma} = \text{median}(|r_i - \text{median}(r_i)|)
```

First centers residuals at their median, then takes the median of the
absolute deviations. Double use of the median makes it highly resistant
to extreme outliers. This is the standard choice for robust regression.

**Use when**: Data may contain outliers (default for most applications).

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(iterations = 3, scaling_method = "mad")
result <- fit(model, x, y)
cat("Smoothed y[0]:", result$y[1], "\n")
#> Smoothed y[0]: 0.4862648
```

------------------------------------------------------------------------

## MAR — Median Absolute Residual

``` math
\hat{\sigma} = \text{median}(|r_i|)
```

Uses the uncentered median — unlike MAD it does not subtract the
residual median first. Still robust (median-based) but slightly less
resistant than MAD when residuals are systematically shifted. Faster
than MAD in practice because it requires only one partial sort.

**Use when**: Speed matters and data have minimal systematic bias in
residuals.

``` r

model <- Lowess(iterations = 3, scaling_method = "mar")
result <- fit(model, x, y)
cat("Smoothed y[0]:", result$y[1], "\n")
#> Smoothed y[0]: 0.4870733
```

------------------------------------------------------------------------

## Mean — Mean Absolute Residual

``` math
\hat{\sigma} = \frac{1}{n}\sum_i |r_i|
```

Arithmetic mean of absolute residuals. Non-robust: a single extreme
outlier inflates $`\hat{\sigma}`$, causing the algorithm to
under-downweight it. Fastest to compute (no sort required). Useful when
data are believed to be clean and speed is a priority.

**Use when**: Clean data with no outliers; maximum computation speed
required.

``` r

model <- Lowess(iterations = 3, scaling_method = "mean")
result <- fit(model, x, y)
cat("Smoothed y[0]:", result$y[1], "\n")
#> Smoothed y[0]: 0.5023604
```

------------------------------------------------------------------------

## Choosing a Scaling Method

| Situation                                             | Recommended Method |
|-------------------------------------------------------|--------------------|
| General purpose, possible outliers                    | `"mad"` (default)  |
| Speed matters; residuals have minimal systematic bias | `"mar"`            |
| Clean data, no outliers                               | `"mean"`           |

See
[Robustness](https://thisisamirv.github.io/lowess-project/r/articles/robustness.md)
for a broader discussion of outlier handling.

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
