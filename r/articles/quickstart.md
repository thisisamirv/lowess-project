# Quick Start

## Basic Smoothing

Smooth a noisy sine wave. `fraction = 0.3` and `iterations = 3` are good
starting values for most signals.

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

## With Confidence and Prediction Intervals

Set `confidence_intervals` and/or `prediction_intervals` to a coverage
level (0–1).

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

print(result$confidence_lower)
#>   [1]  0.402912806  0.444425371  0.422964886  0.424915481  0.433441883
#>   [6]  0.446929963  0.457679203  0.460112021  0.503347796  0.476936803
#>  [11]  0.508467445  0.575237027  0.528723036  0.519214195  0.530327602
#>  [16]  0.557746593  0.548293232  0.620499438  0.610611587  0.636836532
#>  [21]  0.574779503  0.588079172  0.583440472  0.649965404  0.705346779
#>  [26]  0.573558526  0.571724435  0.566052600  0.580009989  0.537916583
#>  [31]  0.554940183  0.551396997  0.551839318  0.478585919  0.484934281
#>  [36]  0.450798591  0.415028558  0.389982680  0.405612481  0.336616301
#>  [41]  0.306209221  0.264875040  0.246130697  0.192691651  0.166265570
#>  [46]  0.117829369  0.076952391  0.059458319 -0.007620676 -0.046040786
#>  [51] -0.091271794 -0.123734424 -0.155686707 -0.215585454 -0.257481664
#>  [56] -0.299136467 -0.339823494 -0.378562970 -0.333851372 -0.458325202
#>  [61] -0.485519279 -0.531178489 -0.569305373 -0.599106397 -0.606963060
#>  [66] -0.661662984 -0.687227029 -0.714555460 -0.735973378 -0.753598831
#>  [71] -0.720614877 -0.767806703 -0.789504958 -0.751148996 -0.773300911
#>  [76] -0.802878896 -0.804292857 -0.799280589 -0.754409589 -0.736875309
#>  [81] -0.773122205 -0.761792936 -0.744189930 -0.722750878 -0.663413089
#>  [86] -0.688968767 -0.653976063 -0.627282024 -0.600788991 -0.568507847
#>  [91] -0.521750850 -0.492030543 -0.460702506 -0.406396679 -0.371907161
#>  [96] -0.344340162 -0.305085025 -0.264666341 -0.247019614 -0.203001699
print(result$confidence_upper)
#>   [1]  0.54166355  0.52001237  0.56246539  0.58266084  0.59725030  0.60773573
#>   [7]  0.62172140  0.64481765  0.62802315  0.68189806  0.67873672  0.64102362
#>  [13]  0.71664953  0.75448571  0.77010616  0.76725791  0.79878487  0.74593967
#>  [19]  0.77227333  0.75954061  0.83191842  0.82542677  0.83309601  0.76562380
#>  [25]  0.70534678  0.82824896  0.81722409  0.80603541  0.77130084  0.78905597
#>  [31]  0.74445466  0.71727226  0.68279219  0.71854782  0.67085631  0.65958302
#>  [37]  0.64534351  0.61530697  0.53957380  0.54375970  0.50509551  0.47350188
#>  [43]  0.41603254  0.39044743  0.33551058  0.30074916  0.25732106  0.18998448
#>  [49]  0.17192246  0.12495629  0.08464427  0.03158521 -0.02163454 -0.04609485
#>  [55] -0.08751076 -0.12812421 -0.16879326 -0.21032377 -0.33385137 -0.28627512
#>  [61] -0.33365675 -0.35988377 -0.39045015 -0.42567271 -0.47873757 -0.48043475
#>  [67] -0.50626119 -0.52490908 -0.54384033 -0.56065236 -0.62201028 -0.59737545
#>  [73] -0.59271548 -0.64296661 -0.62795121 -0.60091924 -0.59736114 -0.59526085
#>  [79] -0.62790574 -0.62807473 -0.56934679 -0.55321459 -0.53856942 -0.52297251
#>  [85] -0.54027560 -0.46769448 -0.45078137 -0.42104518 -0.38712180 -0.35553728
#>  [91] -0.33546335 -0.29624951 -0.25750939 -0.24160246 -0.20657384 -0.16587739
#>  [97] -0.13837350 -0.11376134 -0.06828002 -0.05109043
print(result$diagnostics$r_squared)
#> [1] 0.7841212
```

------------------------------------------------------------------------

## Handling Outliers

LOWESS can robustly handle outliers through iterative reweighting:

``` r

library(rfastlowess)

x_out <- seq(1, 6, length.out = 6)
y_with_outlier <- c(2.0, 4.0, 6.0, 50.0, 10.0, 12.0)

model <- Lowess(
    fraction = 0.5,
    iterations = 5,
    robustness_method = "bisquare",
    return_robustness_weights = TRUE
)
result <- fit(model, x_out, y_with_outlier)

for (i in seq_along(result$robustness_weights)) {
    if (result$robustness_weights[i] < 0.5)
        cat(sprintf("Point %d is likely an outlier (weight: %.3f)\n",
                    i, result$robustness_weights[i]))
}
```

------------------------------------------------------------------------

## Streaming Mode

For large datasets (\>100K points) that may not fit in memory.

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 10000)
y <- sin(x) + rnorm(10000, sd = 0.3)

model <- StreamingLowess(
    fraction = 0.3,
    iterations = 2,
    chunk_size = 5000,
    overlap = 500,
    merge_strategy = "weighted_average"
)

# Process one chunk at a time
chunk_x <- x[1:5000]
chunk_y <- y[1:5000]
result <- process_chunk(model, chunk_x, chunk_y)

# Finalize after all chunks
final <- finalize(model)
print(head(final$y))
#> [1] 0.3234034 0.3229753 0.3225477 0.3221207 0.3216941 0.3212681
```

------------------------------------------------------------------------

## Online Mode

For real-time / point-by-point processing.

``` r

library(rfastlowess)
set.seed(42)
times <- 1:100
temperatures <- 20 + 5 * sin(times / 10) + rnorm(100)

model <- OnlineLowess(
    fraction = 0.3,
    window_capacity = 25,
    min_points = 5,
    update_mode = "incremental"
)

for (i in seq_along(times)) {
    result <- add_point(model, times[i], temperatures[i])
    if (!is.null(result))
        cat(sprintf("Time %d: %.2f\n", times[i], result$y))
}
#> Time 5: 22.80
#> Time 6: 22.72
#> Time 7: 24.73
#> Time 8: 23.49
#> Time 9: 25.94
#> Time 10: 24.14
#> Time 11: 25.35
#> Time 12: 27.00
#> Time 13: 23.99
#> Time 14: 24.04
#> Time 15: 24.64
#> Time 16: 25.60
#> Time 17: 25.13
#> Time 18: 23.01
#> Time 19: 21.96
#> Time 20: 24.34
#> Time 21: 24.42
#> Time 22: 23.41
#> Time 23: 23.21
#> Time 24: 23.86
#> Time 25: 24.49
#> Time 26: 23.48
#> Time 27: 22.44
#> Time 28: 20.54
#> Time 29: 20.50
#> Time 30: 20.03
#> Time 31: 20.29
#> Time 32: 20.35
#> Time 33: 20.27
#> Time 34: 18.98
#> Time 35: 18.52
#> Time 36: 16.82
#> Time 37: 16.15
#> Time 38: 15.74
#> Time 39: 14.60
#> Time 40: 15.17
#> Time 41: 15.68
#> Time 42: 15.58
#> Time 43: 15.97
#> Time 44: 15.21
#> Time 45: 14.17
#> Time 46: 14.57
#> Time 47: 14.31
#> Time 48: 15.51
#> Time 49: 15.33
#> Time 50: 15.71
#> Time 51: 15.76
#> Time 52: 15.26
#> Time 53: 16.38
#> Time 54: 16.85
#> Time 55: 16.94
#> Time 56: 17.17
#> Time 57: 17.66
#> Time 58: 17.85
#> Time 59: 16.47
#> Time 60: 17.56
#> Time 61: 18.29
#> Time 62: 19.43
#> Time 63: 20.61
#> Time 64: 21.85
#> Time 65: 21.41
#> Time 66: 22.33
#> Time 67: 22.58
#> Time 68: 23.29
#> Time 69: 23.88
#> Time 70: 24.23
#> Time 71: 23.47
#> Time 72: 23.59
#> Time 73: 24.27
#> Time 74: 24.04
#> Time 75: 24.19
#> Time 76: 24.92
#> Time 77: 25.52
#> Time 78: 25.75
#> Time 79: 25.01
#> Time 80: 24.18
#> Time 81: 25.10
#> Time 82: 25.15
#> Time 83: 25.01
#> Time 84: 24.54
#> Time 85: 23.35
#> Time 86: 23.48
#> Time 87: 23.18
#> Time 88: 22.87
#> Time 89: 23.08
#> Time 90: 22.94
#> Time 91: 22.92
#> Time 92: 21.62
#> Time 93: 21.08
#> Time 94: 20.99
#> Time 95: 19.46
#> Time 96: 18.41
#> Time 97: 17.43
#> Time 98: 16.54
#> Time 99: 16.88
#> Time 100: 17.46
```

------------------------------------------------------------------------

## Plotting Results

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(fraction = 0.5, confidence_intervals = 0.95)
result <- fit(model, x, y)

plot(x, y, pch = 16, col = "gray", main = "LOWESS Smoothing")
lines(result$x, result$y, col = "blue", lwd = 2)
lines(result$x, result$confidence_lower, col = "blue", lty = 2)
lines(result$x, result$confidence_upper, col = "blue", lty = 2)
legend("topright", c("Data", "Smoothed", "95% CI"),
        pch = c(16, NA, NA), lty = c(NA, 1, 2),
        col = c("gray", "blue", "blue"))
```

![](quickstart_files/figure-html/quickstart_6-1.png)

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
#> [1] rfastlowess_3.0.0 BiocStyle_2.40.0 
#> 
#> loaded via a namespace (and not attached):
#>  [1] cli_3.6.6           knitr_1.51          rlang_1.3.0        
#>  [4] xfun_0.60           otel_0.2.0          generics_0.1.4     
#>  [7] textshaping_1.0.5   jsonlite_2.0.0      htmltools_0.5.9    
#> [10] ragg_1.5.2          sass_0.4.10         rmarkdown_2.31     
#> [13] evaluate_1.0.5      jquerylib_0.1.4     fastmap_1.2.0      
#> [16] yaml_2.3.12         lifecycle_1.0.5     bookdown_0.47      
#> [19] BiocManager_1.30.27 compiler_4.6.1      fs_2.1.0           
#> [22] htmlwidgets_1.6.4   systemfonts_1.3.2   digest_0.6.39      
#> [25] R6_2.6.1            bslib_0.12.0        tools_4.6.1        
#> [28] BiocGenerics_0.58.1 pkgdown_2.2.1       cachem_1.1.0       
#> [31] desc_1.4.3
```
