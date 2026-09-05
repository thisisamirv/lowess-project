# Choosing an Execution Mode

`rfastlowess` provides three execution modes for different data sizes
and processing needs:

![Execution mode
comparison](../reference/figures/adapter_comparison.svg)

Execution mode comparison

| Mode | Class | Use Case | Memory | Key Features |
|----|----|----|----|----|
| **Batch** | `Lowess` | Full dataset | Entire dataset | CI, PI, CV, GPU |
| **Streaming** | `StreamingLowess` | \>100K points | Per chunk | With overlap |
| **Online** | `OnlineLowess` | Live / real-time | Sliding window | Per-point |

## Quick Decision Guide

| Situation                                      | Mode      |
|------------------------------------------------|-----------|
| Data fits in memory; needs intervals or CV     | Batch     |
| Data too large for memory or arrives in chunks | Streaming |
| Data arrives point-by-point in real time       | Online    |

------------------------------------------------------------------------

## Batch Adapter

Standard mode for complete datasets. **Supports all features.**

### When to Use

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

### Example

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
    return_diagnostics = TRUE,
    parallel = TRUE
)
result <- fit(model, x, y)
cat("95% CI at midpoint: [", result$confidence_lower[50], ", ",
    result$confidence_upper[50], "]\n")
#> 95% CI at midpoint: [ -0.04604079 ,  0.1249563 ]
```

------------------------------------------------------------------------

## Streaming Adapter

Process large datasets in chunks with configurable overlap.

### When to Use

- Dataset \>100,000 points
- Memory-constrained environments
- Batch processing pipelines

### Parameters

| Parameter        | Default              | Description            |
|------------------|----------------------|------------------------|
| `chunk_size`     | 5000                 | Points per chunk       |
| `overlap`        | `chunk_size / 10`    | Overlap between chunks |
| `merge_strategy` | `"weighted_average"` | How to merge overlaps  |

### Merge Strategies

| Strategy             | Behavior                   |
|----------------------|----------------------------|
| `"average"`          | Average overlapping values |
| `"weighted_average"` | Distance-weighted blend    |
| `"take_first"`       | Keep left chunk values     |
| `"take_last"`        | Keep right chunk values    |

### Example

``` r

library(rfastlowess)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + 0.1

model <- StreamingLowess(
    fraction = 0.3,
    iterations = 2,
    chunk_size = 5000,
    overlap = 500,
    merge_strategy = "average"
)
process_chunk(model, x, y)
#> <LowessResult>
#>   Points:            0 
#>   Fraction Used:     0.3 
#>   Iterations Used:   0
result <- finalize(model)
cat("Smoothed y[0]:", result$y[1], "\n")
#> Smoothed y[0]: 0.2578452
```

> **Always call finalize():** The streaming adapter buffers overlap
> data. Call `finalize(model)` after the last chunk to retrieve the
> buffered tail.

------------------------------------------------------------------------

## Online Adapter

Incremental updates with a sliding window for real-time data.

### When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

### Parameters

| Parameter         | Default         | Description                 |
|-------------------|-----------------|-----------------------------|
| `window_capacity` | 1000            | Max points in window        |
| `min_points`      | 2               | Points before output starts |
| `update_mode`     | `"incremental"` | Update strategy             |

### Update Modes

| Mode            | Behavior                  | Speed         |
|-----------------|---------------------------|---------------|
| `"incremental"` | Update only affected fits | Faster        |
| `"full"`        | Recompute entire window   | More accurate |

### Example

``` r

library(rfastlowess)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + 0.1

model <- OnlineLowess(
    fraction = 0.2,
    iterations = 1,
    window_capacity = 100,
    min_points = 5,
    update_mode = "incremental"
)
shown <- 0
for (i in seq_along(x)) {
    result <- add_point(model, x[i], y[i])
    if (!is.null(result) && shown < 5) {
        cat("Current smoothed value:", result$y, "\n")
        shown <- shown + 1
    }
}
#> Current smoothed value: 0.351148 
#> Current smoothed value: 0.4120334 
#> Current smoothed value: 0.4716625 
#> Current smoothed value: 0.5297949 
#> Current smoothed value: 0.5861967
```

------------------------------------------------------------------------

## Feature Comparison

| Feature              | Batch | Streaming | Online |
|----------------------|-------|-----------|--------|
| Confidence intervals | ✓     | ✗         | ✗      |
| Prediction intervals | ✓     | ✗         | ✗      |
| Cross-validation     | ✓     | ✗         | ✗      |
| Diagnostics          | ✓     | ✓         | ✗      |
| Residuals            | ✓     | ✓         | ✓      |
| Robustness weights   | ✓     | ✓         | ✓      |
| Parallel execution   | ✓     | ✓         | ✗      |

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
