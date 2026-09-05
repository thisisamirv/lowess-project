# Cross-Validation

## Overview

![Cross-validation comparison](../reference/figures/cv_comparison.svg)

Cross-validation comparison

Cross-validation helps select optimal parameters (especially `fraction`)
by evaluating performance on held-out data.

------------------------------------------------------------------------

## K-Fold Cross-Validation

Split data into K folds, train on K-1, validate on 1.

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(
    cv_method = "kfold",
    cv_k = 5,
    cv_fractions = c(0.2, 0.3, 0.5, 0.7)
)
result <- fit(model, x, y)

cat("Selected fraction:", result$fraction_used, "\n")
#> Selected fraction: 0.3
cat("CV scores:", result$cv_scores, "\n")
#> CV scores: 0.4753221 0.43492 0.491747 0.5442841
```

------------------------------------------------------------------------

## Leave-One-Out (LOOCV)

Each point is held out once. Most thorough but slowest.

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(
    cv_method = "loocv",
    cv_fractions = c(0.2, 0.3, 0.5, 0.7)
)
result <- fit(model, x, y)

cat("Selected fraction (CV):", result$fraction_used, "\n")
#> Selected fraction (CV): 0.3
```

------------------------------------------------------------------------

## Seeded Randomization

Set a seed for reproducible fold assignments:

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(
    cv_method = "kfold",
    cv_k = 5,
    cv_fractions = c(0.3, 0.5, 0.7),
    cv_seed = 42L
)
result <- fit(model, x, y)
cat("Selected fraction (CV):", result$fraction_used, "\n")
#> Selected fraction (CV): 0.7
```

------------------------------------------------------------------------

## Comparison

| Method        | Folds | Speed  | Variance | Bias   |
|---------------|-------|--------|----------|--------|
| **KFold(5)**  | 5     | Fast   | Moderate | Low    |
| **KFold(10)** | 10    | Medium | Lower    | Lower  |
| **LOOCV**     | N     | Slow   | Lowest   | Lowest |

> **Recommendation:** Use **5-fold** or **10-fold** CV for most
> applications. LOOCV is only worth it for small datasets (N \< 100).

------------------------------------------------------------------------

## CV Metrics

Cross-validation uses MSE (Mean Squared Error) by default:

``` text
MSE = mean((y_true - y_pred)^2)
```

Lower MSE indicates better fit on held-out data.

------------------------------------------------------------------------

## Interpreting Results

`result$cv_scores` contains the CV score for each candidate fraction in
the order they were passed. The fraction with **lowest CV score** is
automatically selected.

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

fractions <- c(0.1, 0.3, 0.5, 0.7)
model <- Lowess(cv_method = "kfold", cv_k = 5, cv_fractions = fractions)
result <- fit(model, x, y)

plot(fractions, result$cv_scores, type = "b",
    xlab = "Fraction", ylab = "CV Score (MSE)",
    main = "Cross-Validation Score by Fraction")
abline(v = result$fraction_used, col = "red", lty = 2)
```

![](cross-validation_files/figure-html/cross_validation_4-1.png)

``` r


cat("Selected fraction (CV):", result$fraction_used, "\n")
#> Selected fraction (CV): 0.3
```

------------------------------------------------------------------------

## Availability

> **Batch Mode Only:** Cross-validation is only available in **Batch**
> mode.

| Feature   | Batch | Streaming | Online |
|-----------|-------|-----------|--------|
| K-Fold CV | ✓     | ✗         | ✗      |
| LOOCV     | ✓     | ✗         | ✗      |

------------------------------------------------------------------------

## Best Practices

1.  **Test a range**: Include fractions from 0.1 to 0.9
2.  **Use enough folds**: 5-10 folds balance speed and accuracy
3.  **Set a seed**: For reproducible results
4.  **Check the curve**: CV optimizes MSE, but visual inspection matters

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
