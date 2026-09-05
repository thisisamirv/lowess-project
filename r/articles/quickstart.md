# Quick Start

Get up and running with LOWESS in minutes.

## Basic Smoothing

Smooth a noisy sine wave — the kind of signal where LOWESS shines. Each
example recovers the underlying trend from 100 points of Gaussian noise.

``` r

library(rfastlowess)

# 100-point noisy sine wave
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(fraction = 0.3, iterations = 3)
result <- fit(model, x, y)

cat(sprintf("First smoothed value: %.4f (true: %.4f)\n",
            result$y[1], sin(x[1])))
#> First smoothed value: 0.4246 (true: 0.0000)
```

------------------------------------------------------------------------

## With Confidence Intervals

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(
    fraction = 0.5,
    iterations = 3,
    confidence_intervals = 0.95,
    prediction_intervals = 0.95,
    return_diagnostics = TRUE
)
result <- fit(model, x, y)

cat("Smoothed (first 5):", head(result$y, 5), "\n")
#> Smoothed (first 5): 0.4722882 0.4822189 0.4927151 0.5037882 0.5153461
cat("CI lower (first 5):", head(result$confidence_lower, 5), "\n")
#> CI lower (first 5): 0.4029128 0.4444254 0.4229649 0.4249155 0.4334419
cat("CI upper (first 5):", head(result$confidence_upper, 5), "\n")
#> CI upper (first 5): 0.5416635 0.5200124 0.5624654 0.5826608 0.5972503
cat("R2:", result$diagnostics$r_squared, "\n")
#> R2: 0.7841212
```

------------------------------------------------------------------------

## Handling Outliers

LOWESS can robustly handle outliers through iterative reweighting:

``` r

library(rfastlowess)

x_out <- seq(1, 6, length.out = 6)
y_with_outlier <- c(2.0, 4.0, 6.0, 50.0, 10.0, 12.0)

model <- Lowess(
    fraction = 0.7,
    iterations = 5,
    robustness_method = "bisquare",
    return_robustness_weights = TRUE
)
result <- fit(model, x_out, y_with_outlier)

# Outliers will have low robustness weights
for (i in seq_along(result$robustness_weights)) {
    if (result$robustness_weights[i] < 0.5)
        cat(sprintf("Point %d is likely an outlier (weight: %.3f)\n",
                    i, result$robustness_weights[i]))
}
#> Point 4 is likely an outlier (weight: 0.000)
```

------------------------------------------------------------------------

## Streaming Mode

For datasets too large to fit in memory, stream them in fixed-size
chunks with overlap.

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 10 * pi, length.out = 5000)
y <- sin(x / pi) * exp(-x / 30) + rnorm(5000, sd = 0.15)

model <- StreamingLowess(
    fraction = 0.2,
    chunk_size = 1000,
    overlap = 100,
    merge_strategy = "weighted_average"
)

chunk_size <- 1000
for (start in seq(1, 4001, by = chunk_size)) {
    end <- min(start + chunk_size - 1, length(x))
    process_chunk(model, x[start:end], y[start:end])
}
result <- finalize(model)
cat(sprintf("Smoothed %d points in streaming mode\n", length(result$y)))
#> Smoothed 100 points in streaming mode
```

------------------------------------------------------------------------

## Next Steps

| Topic | Link |
|----|----|
| How LOWESS works | [Concepts](https://thisisamirv.github.io/lowess-project/r/articles/concepts.md) |
| All parameters explained | [`?Lowess`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md) |
| Batch vs Streaming vs Online | [Execution Modes](https://thisisamirv.github.io/lowess-project/r/articles/adapter-choice.md) |
| Edge handling | [Boundary](https://thisisamirv.github.io/lowess-project/r/articles/boundary.md) |
| Outlier handling in depth | [Robustness](https://thisisamirv.github.io/lowess-project/r/articles/robustness.md) |

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
