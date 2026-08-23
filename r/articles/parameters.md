# Parameters Reference

## All Parameters

| Parameter | Default | Range | Description | Mode |
|----|----|----|----|----|
| **fraction** | 0.67 | (0, 1\] | Neighbourhood size | All |
| **iterations** | 0 | \[0, ∞) | Robustness iterations | All |
| **degree** | 1 | 0, 1, 2 | Polynomial degree | All |
| **delta** | 0.0 | \[0, ∞) | Distance threshold for skipping | Batch |
| **weight_function** | `"tricube"` | 7 options | Distance kernel | All |
| **robustness_method** | `"bisquare"` | 3 options | Outlier weighting | All |
| **scaling_method** | `"mad"` | 3 options | Scale estimate | All |
| **zero_weight_fallback** | `"use_local_mean"` | 3 options | Zero-weight handling | All |
| **boundary_policy** | `"extend"` | 4 options | Edge padding | All |
| **backend** | `"cpu"` | `"cpu"`, `"gpu"` | Compute backend | Batch |
| **parallel** | `FALSE` | `TRUE`/`FALSE` | Parallel CPU execution | Batch |
| **confidence_intervals** | `NULL` | (0, 1) | CI coverage level | Batch |
| **prediction_intervals** | `NULL` | (0, 1) | PI level | Batch |
| **cv_method** | `NULL` | method | Auto-select fraction | Batch |
| **chunk_size** | 5000 | \[10, ∞) | Points per chunk | Streaming |
| **overlap** | 500 | \[0, chunk) | Overlap between chunks | Streaming |
| **merge_strategy** | `"weighted_average"` | 4 options | Merge overlaps | Streaming |
| **window_capacity** | 1000 | \[3, ∞) | Max window size | Online |
| **min_points** | 2 | \[2, window\] | Min before output | Online |
| **update_mode** | `"incremental"` | 2 options | Update strategy | Online |

------------------------------------------------------------------------

## Parameter Options

| Parameter | Available Options |
|----|----|
| **weight_function** | `"tricube"`, `"epanechnikov"`, `"gaussian"`, `"biweight"`, `"cosine"`, `"triangle"`, `"uniform"` |
| **robustness_method** | `"bisquare"`, `"huber"`, `"talwar"` |
| **zero_weight_fallback** | `"use_local_mean"`, `"return_original"`, `"return_none"` |
| **boundary_policy** | `"extend"`, `"reflect"`, `"zero"`, `"noboundary"` |
| **scaling_method** | `"mad"`, `"mar"`, `"mean"` |
| **merge_strategy** | `"average"`, `"weighted_average"`, `"take_first"`, `"take_last"` |
| **update_mode** | `"incremental"`, `"full"` |

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
```

### iterations

Number of robustness iterations. Each iteration downweights
high-residual points.

``` r

model <- Lowess(iterations = 3)
result <- fit(model, x, y)
```

### degree

Polynomial degree for local fits: 0 (constant), 1 (linear, default), or
2 (quadratic).

``` r

model <- Lowess(degree = 2L)
result <- fit(model, x, y)
```

### delta

Distance threshold for skipping intermediate points (approximation).
Setting `delta > 0` speeds up computation on dense grids. Passed as a
fraction of the data range.

``` r

model <- Lowess(delta = 0.01)
result <- fit(model, x, y)
```

### parallel

Enable parallel CPU execution (multiple cores).

``` r

model <- Lowess(parallel = TRUE)
result <- fit(model, x, y)
```

### return_diagnostics

Return goodness-of-fit diagnostics (R², residuals, etc.).

``` r

model <- Lowess(return_diagnostics = TRUE)
result <- fit(model, x, y)
print(result$diagnostics$r_squared)
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
