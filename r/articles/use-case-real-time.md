# Use Case: Real-Time Processing

Streaming and online LOWESS for live data.

## Overview

When data arrives continuously—from sensors, logs, or streaming
pipelines—you need incremental smoothing that doesn’t require
reprocessing the entire dataset.

------------------------------------------------------------------------

## Online Mode: Point-by-Point

For true real-time applications where each point must be processed
immediately.

`window_capacity = 25` limits the internal buffer to the 25 most recent
observations; each
[`add_point()`](https://thisisamirv.github.io/lowess-project/r/reference/add_point.md)
call costs O(window) rather than growing with total history.
`min_points = 5` suppresses output until the window holds enough points
for a stable fit — calls made before that threshold return `NULL`.
`update_mode = "incremental"` re-fits only the most recent point rather
than the full window, halving typical latency at a modest accuracy cost.

### Sensor Data Example

``` r

library(rfastlowess)

model <- OnlineLowess(
    fraction = 0.3,
    iterations = 1,
    window_capacity = 25,
    min_points = 5,
    update_mode = "incremental"
)

# Simulate real-time data arrival
count <- 0
for (i in 0:99) {
    yi <- 20.0 + 5.0 * sin(i / 10.0) + sin(i * 1.7) * 0.5

    result <- add_point(model, i, yi)
    if (!is.null(result)) {
        if (count < 5) cat(sprintf("Time %d: smoothed = %.4f\n", i, result$y))
        count <- count + 1
    }
}
#> Time 4: smoothed = 22.1941
#> Time 5: smoothed = 22.7964
#> Time 6: smoothed = 22.4733
#> Time 7: smoothed = 22.9120
#> Time 8: smoothed = 24.0164
cat(sprintf("... (%d more)\n", count - 5))
#> ... (91 more)
```

------------------------------------------------------------------------

## Streaming Mode: Chunk Processing

For large datasets that arrive in batches or files.

`chunk_size` controls how many data points are processed in one pass;
matching it to your file-read buffer or message-batch size avoids
unnecessary copying. `overlap` retains that many points from the
previous chunk as context so the neighbourhood at chunk boundaries is
not artificially truncated. `merge_strategy = "weighted_average"` blends
the overlapping region smoothly; use `"last"` if chunk boundaries are
guaranteed to be well separated and no blending is needed.

> **Always call finalize():** The streaming adapter buffers overlap
> data. Call
> [`finalize()`](https://thisisamirv.github.io/lowess-project/r/reference/finalize.md)
> after the last chunk to retrieve the buffered tail.

### Log File Processing

``` r

library(rfastlowess)

chunk1_x <- 0:49
chunk1_y <- sin(chunk1_x) + 0.1
chunk2_x <- 50:99
chunk2_y <- sin(chunk2_x) + 0.1

model <- StreamingLowess(
    fraction = 0.1,
    iterations = 2,
    chunk_size = 50,
    overlap = 10,
    merge_strategy = "weighted_average"
)

# Process chunks as they arrive
process_chunk(model, chunk1_x, chunk1_y)
#> <LowessResult>
#>   Points:            40 
#>   Fraction Used:     0.1 
#>   Iterations Used:   0
process_chunk(model, chunk2_x, chunk2_y)
#> <LowessResult>
#>   Points:            50 
#>   Fraction Used:     0.1 
#>   Iterations Used:   0

# CRITICAL: Get buffered overlap data
result <- finalize(model)
cat(sprintf("y[0]: %.6f\n", result$y[1]))
#> y[0]: 0.516484
```

------------------------------------------------------------------------

## Real-Time Dashboard Example

The dashboard pattern uses a plain LOWESS fit on a manually managed
sliding window rather than `OnlineLowess`. This is the simplest approach
when your UI framework already owns the data buffer and you only need
the most recent smoothed value per frame. The trade-off is a full
O(window^2) refit on every tick; for high-frequency streams prefer
`OnlineLowess` with `update_mode = "incremental"` to bound per-frame
cost.

``` r

library(rfastlowess)

n <- 100
x <- seq(0, 2 * pi, length.out = n)
y <- sin(x) + 0.1

window_x <- numeric(0)
window_y <- numeric(0)
last_smoothed <- 0

for (i in seq_along(x)) {
    window_x <- c(window_x, x[i])
    window_y <- c(window_y, y[i])

    if (length(window_x) > 50) {
        window_x <- window_x[-1]
        window_y <- window_y[-1]
    }

    if (length(window_x) < 2) next
    model <- Lowess(fraction = 0.4)
    result <- fit(model, window_x, window_y)
    last_smoothed <- result$y[length(result$y)]
}
cat(sprintf("Smoothed (dashboard, latest tick): %.4f\n", last_smoothed))
#> Smoothed (dashboard, latest tick): -0.0663
```

------------------------------------------------------------------------

## Choosing Parameters

### Online Mode

| Parameter         | Guidance                                         |
|-------------------|--------------------------------------------------|
| `window_capacity` | Enough history for `fraction` to work            |
| `min_points`      | 2–5 typically; higher for stability              |
| `update_mode`     | `"incremental"` for speed, `"full"` for accuracy |

### Streaming Mode

| Parameter        | Guidance                                                |
|------------------|---------------------------------------------------------|
| `chunk_size`     | Balance memory vs. processing overhead                  |
| `overlap`        | 10–20% of chunk_size for smooth transitions             |
| `merge_strategy` | `"weighted_average"` for quality, `"average"` otherwise |

------------------------------------------------------------------------

## Performance Considerations

| Mode          | Memory         | Latency      | Use Case            |
|---------------|----------------|--------------|---------------------|
| **Online**    | Fixed (window) | ~1ms/point   | Sensors, dashboards |
| **Streaming** | ~chunk_size    | ~100ms/chunk | Large files, ETL    |
| **Batch**     | Full dataset   | N/A          | Analysis, reports   |

------------------------------------------------------------------------

## See Also

- [Execution
  Modes](https://thisisamirv.github.io/lowess-project/r/articles/adapter-choice.md)
  — Detailed mode comparison
- [Merge
  Strategies](https://thisisamirv.github.io/lowess-project/r/articles/merge.md)
  — Chunk reconciliation in depth
- [Scaling
  Methods](https://thisisamirv.github.io/lowess-project/r/articles/scaling.md)
  — Robustness scale estimation
- [Time
  Series](https://thisisamirv.github.io/lowess-project/r/articles/use-case-time-series.md)
  — General time series analysis

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
