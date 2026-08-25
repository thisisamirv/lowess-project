# Batch Mode (Lowess)

> **See also:** [Choosing an Execution
> Mode](https://thisisamirv.github.io/lowess-project/r/articles/adapter-choice.md)
> for a comparison of all three modes.

The default mode. Processes the entire dataset at once and supports all
features.

![Gap Handling](../reference/figures/gap_handling.svg)

Gap Handling

## When to Use

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

## Parameters

| Parameter              | Default | Description             |
|------------------------|---------|-------------------------|
| `fraction`             | 0.67    | Neighbourhood size      |
| `iterations`           | 3       | Robustness iterations   |
| `confidence_intervals` | `NULL`  | CI coverage (e.g. 0.95) |
| `prediction_intervals` | `NULL`  | PI coverage (e.g. 0.95) |
| `parallel`             | `FALSE` | Enable CPU parallelism  |
| `backend`              | `"cpu"` | `"cpu"` or `"gpu"`      |

## Example

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

cat("Confidence lower bounds (first 5):\n")
#> Confidence lower bounds (first 5):
print(head(result$confidence_lower, 5))
#> [1] 0.4029128 0.4444254 0.4229649 0.4249155 0.4334419
cat("Confidence upper bounds (first 5):\n")
#> Confidence upper bounds (first 5):
print(head(result$confidence_upper, 5))
#> [1] 0.5416635 0.5200124 0.5624654 0.5826608 0.5972503
cat("R²:", result$diagnostics$r_squared, "\n")
#> R²: 0.7841212
```

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
#> [1] rfastlowess_3.1.0
#> 
#> loaded via a namespace (and not attached):
#>  [1] digest_0.6.39     desc_1.4.3        R6_2.6.1          fastmap_1.2.0    
#>  [5] xfun_0.60         cachem_1.1.0      knitr_1.51        htmltools_0.5.9  
#>  [9] rmarkdown_2.31    lifecycle_1.0.5   cli_3.6.6         sass_0.4.10      
#> [13] pkgdown_2.2.1     textshaping_1.0.5 jquerylib_0.1.4   systemfonts_1.3.2
#> [17] compiler_4.6.1    tools_4.6.1       ragg_1.5.2        bslib_0.12.0     
#> [21] evaluate_1.0.5    yaml_2.3.12       otel_0.2.0        jsonlite_2.0.0   
#> [25] rlang_1.3.0       fs_2.1.0          htmlwidgets_1.6.4
```
