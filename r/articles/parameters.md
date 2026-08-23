# Parameters Reference

## All Parameters

| Parameter | Default | Range | Description | Mode |
|----|----|----|----|----|
| **fraction** | 0.67 | (0, 1\] | Neighbourhood size | All |
| **iterations** | 3 | \[0, 1000\] | Robustness iterations | All |
| **delta** | `NULL` | \[0, ∞) | Distance threshold for skipping | All |
| **weight_function** | `"tricube"` | 7 options | Distance kernel | All |
| **robustness_method** | `"bisquare"` | 3 options | Outlier weighting | All |
| **scaling_method** | `"mad"` | 3 options | Scale estimate | All |
| **zero_weight_fallback** | `"use_local_mean"` | 3 | Zero-weight | All |
| **boundary_policy** | `"extend"` | 4 options | Edge padding | All |
| **auto_converge** | `NULL` | tolerance | Early stopping | All |
| **return_residuals** | `FALSE` | logical | Include residuals | All |
| **return_robustness_weights** | `FALSE` | logical | Include weights | All |
| **return_se** | `FALSE` | logical | Return standard errors | All |
| **return_diagnostics** | `FALSE` | logical | Metrics | Batch, Streaming |
| **custom_weights** | `NULL` | positive | Per-observation weights | Batch |
| **backend** | `"cpu"` | `"cpu"`, `"gpu"` | Compute backend | Batch |
| **parallel** | `FALSE` | `TRUE`/`FALSE` | Parallel CPU execution | Batch |
| **confidence_intervals** | `NULL` | (0, 1) | CI coverage level | Batch |
| **prediction_intervals** | `NULL` | (0, 1) | PI level | Batch |
| **cv_method** | `NULL` | method | Auto-select fraction | Batch |
| **chunk_size** | 5000 | \[10, ∞) | Points per chunk | Streaming |
| **overlap** | 500 | \[0, chunk) | Overlap between chunks | Streaming |
| **merge_strategy** | `"weighted_average"` | 4 | Chunk merge | Streaming |
| **window_capacity** | 1000 | \[3, ∞) | Max window size | Online |
| **min_points** | 2 | \[2, window\] | Min before output | Online |
| **update_mode** | `"incremental"` | 2 options | Update strategy | Online |

------------------------------------------------------------------------

## Parameter Options

- **weight_function**: `"tricube"` (default), `"epanechnikov"`,
  `"gaussian"`, `"biweight"`, `"cosine"`, `"triangle"`, `"uniform"`
- **robustness_method**: `"bisquare"` (default), `"huber"`, `"talwar"`
- **zero_weight_fallback**: `"use_local_mean"` (default),
  `"return_original"`, `"return_none"`
- **boundary_policy**: `"extend"` (default), `"reflect"`, `"zero"`,
  `"noboundary"`
- **scaling_method**: `"mad"` (default), `"mar"`, `"mean"`
- **merge_strategy**: `"weighted_average"` (default), `"average"`,
  `"take_first"`, `"take_last"`
- **update_mode**: `"incremental"` (default), `"full"`

------------------------------------------------------------------------

## Core Parameters

### fraction

The proportion of data used for each local fit. **Most important
parameter.**

| Value   | Effect          | Use Case                 |
|---------|-----------------|--------------------------|
| 0.1–0.3 | Fine detail     | Rapidly changing signals |
| 0.3–0.5 | Balanced        | General purpose          |
| 0.5–0.7 | Heavy smoothing | Noisy data               |
| 0.7–1.0 | Very smooth     | Trend extraction         |

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(fraction = 0.3)
result <- fit(model, x, y)
print(head(result$y))
#> [1] 0.4246043 0.4368821 0.4512604 0.4675346 0.4850950 0.5036147
```

### iterations

Number of robustness iterations. Each iteration downweights
high-residual points.

``` r

model <- Lowess(iterations = 3)
result <- fit(model, x, y)
print(head(result$y))
#> [1] 0.4862648 0.4942855 0.5026953 0.5114688 0.5205144 0.5296957
```

### delta

Distance threshold for skipping intermediate points (approximation).
Setting `delta > 0` speeds up computation on dense grids. Passed as a
fraction of the data range.

``` r

model <- Lowess(delta = 0.01)
result <- fit(model, x, y)
print(head(result$y))
#> [1] 0.4862648 0.4942855 0.5026953 0.5114688 0.5205144 0.5296957
```

### parallel

Enable parallel CPU execution (multiple cores).

``` r

model <- Lowess(parallel = TRUE)
result <- fit(model, x, y)
print(head(result$y))
#> [1] 0.4862648 0.4942855 0.5026953 0.5114688 0.5205144 0.5296957
```

### return_diagnostics

Return goodness-of-fit diagnostics (R², residuals, etc.).

``` r

model <- Lowess(return_diagnostics = TRUE)
result <- fit(model, x, y)
print(result$diagnostics$r_squared)
#> [1] 0.7296086
```

------------------------------------------------------------------------

## Streaming Parameters

### chunk_size and overlap

``` r

model <- StreamingLowess(
    chunk_size = 5000,
    overlap = 500,
    merge_strategy = "weighted_average"
)
```

------------------------------------------------------------------------

## Online Parameters

### window_capacity and min_points

``` r

model <- OnlineLowess(
    window_capacity = 50,
    min_points = 5,
    update_mode = "incremental"
)
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
