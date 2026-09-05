# Use Case: Time Series Analysis

LOWESS for trend extraction and temporal smoothing.

## Overview

Time series data often contains noise, seasonality, and trends. LOWESS
provides flexible trend extraction without parametric assumptions.

------------------------------------------------------------------------

## Basic Trend Extraction

`fraction = 0.1` sizes the neighbourhood as 10% of the data at each
evaluation point — narrow enough to follow a slowly varying trend
without smearing periodic variation. Three robustness `iterations`
down-weight noise spikes so they cannot bias the fitted curve; this is
especially important when the signal-to-noise ratio is low or when
occasional outliers are expected.

``` r

library(rfastlowess)

n <- 500
t <- seq(0, 100, length.out = n)
i <- 0:(n - 1)
y <- 10 + 0.5 * t + 3 * sin(t / 10) + ((i * 7 + 3) %% 1.7 - 0.85) * 3

# t and y are your time series vectors
model <- Lowess(
    fraction = 0.1,
    iterations = 3
)
result <- fit(model, t, y)

cat(sprintf("y[0]: %.4f\n", result$y[1]))
#> y[0]: 11.3216
```

------------------------------------------------------------------------

## Detrending

Remove trend to analyze residual patterns.

Setting `return_residuals = TRUE` stores `observed - smoothed` alongside
the smooth. A slightly wider `fraction = 0.3` produces a smoother
baseline trend, so short-duration oscillations end up in the residuals
rather than being absorbed into the trend component. The residual series
is then ready for spectral analysis, seasonality detection, or
change-point methods.

``` r

library(rfastlowess)

n <- 100
t <- seq(0, 2 * pi, length.out = n)
y <- sin(t) + 0.1

model <- Lowess(
    fraction = 0.3,
    iterations = 3,
    return_residuals = TRUE
)
result <- fit(model, t, y)

trend <- result$y
detrended <- result$residuals
cat(sprintf("Trend y[0]: %.4f  residual: %.4f\n", trend[1], detrended[1]))
#> Trend y[0]: 0.2582  residual: -0.1582
```

------------------------------------------------------------------------

## Forecasting with Prediction Intervals

Prediction intervals widen the uncertainty band to include both the
uncertainty in the fitted curve (confidence interval) and the expected
scatter of new observations around it. `fraction = 0.2` offers a balance
between local detail and stable interval width — too small a fraction
produces jagged interval edges; too large a fraction underestimates
local variance near turning points.

``` r

library(rfastlowess)

n <- 100
t <- seq(0, 2 * pi, length.out = n)
y <- sin(t) + 0.1

model <- Lowess(
    fraction = 0.2,
    iterations = 3,
    confidence_intervals = 0.95,
    prediction_intervals = 0.95
)
result <- fit(model, t, y)

cat(sprintf(
    "95%% PI: [%.4f, %.4f]\n",
    result$prediction_lower[1], result$prediction_upper[1]
))
#> 95% PI: [0.1580, 0.2909]
```

------------------------------------------------------------------------

## Handling Missing Data

LOWESS naturally handles irregular time sampling:

``` r

library(rfastlowess)

t_irregular <- vapply(
    0:99, function(i) i * 1.0 + (i * 31 %% 10) * 0.1, numeric(1)
)
y_irregular <- 10 + 0.3 * t_irregular + 2.0 * sin(t_irregular * 0.1)

# No special handling needed for irregular spacing
model <- Lowess(fraction = 0.2)
result <- fit(model, t_irregular, y_irregular)
cat(sprintf("y[0]: %.4f\n", result$y[1]))
#> y[0]: 11.4324
```

------------------------------------------------------------------------

## Multi-Scale Analysis

Use different fractions to extract features at different scales:

``` r

library(rfastlowess)

n <- 100
t <- seq(0, 2 * pi, length.out = n)
y <- sin(t) + 0.1

scales <- c(0.05, 0.2, 0.5)
trends <- lapply(scales, function(f) {
    model <- Lowess(fraction = f)
    fit(model, t, y)$y
})
cat("Trend y[0] values:", vapply(trends, function(tr) tr[1], numeric(1)), "\n")
#> Trend y[0] values: 0.131712 0.2244466 0.3343704
```

------------------------------------------------------------------------

## Gene Expression Time Course

Biological application:

``` r

library(rfastlowess)

hours <- seq(0, 24, by = 0.5)
i <- seq_along(hours) - 1
expression <- 100 * (1 + 0.5 * sin(hours * pi / 12)) +
    ((i * 7 + 3) %% 1.7 - 0.85) * 10

model <- Lowess(
    fraction = 0.3,
    iterations = 3,
    confidence_intervals = 0.95,
    return_diagnostics = TRUE
)
result <- fit(model, hours, expression)
cat(sprintf("R2: %.3f\n", result$diagnostics$r_squared))
#> R2: 0.973
```

------------------------------------------------------------------------

## Choosing Fraction for Time Series

| Data Type             | Recommended Fraction | Rationale                    |
|-----------------------|----------------------|------------------------------|
| Daily data (years)    | 0.3–0.5              | Capture annual trends        |
| Hourly data (days)    | 0.1–0.2              | Capture daily patterns       |
| Sensor data (minutes) | 0.05–0.1             | Preserve short-term features |
| Noisy data            | Higher               | Reduce noise impact          |
| Clean data            | Lower                | Preserve detail              |

------------------------------------------------------------------------

## See Also

- [Real-Time
  Processing](https://thisisamirv.github.io/lowess-project/r/articles/use-case-real-time.md)
  — For streaming time series
- [Cross-Validation](https://thisisamirv.github.io/lowess-project/r/articles/cross-validation.md)
  — Optimal fraction selection
- [Boundary
  Handling](https://thisisamirv.github.io/lowess-project/r/articles/boundary.md)
  — Edge bias in trend extraction
- [`?Lowess`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md)
  — Full parameter reference

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
