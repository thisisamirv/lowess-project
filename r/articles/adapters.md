# Execution Modes

------------------------------------------------------------------------

title: “Execution Modes” output: BiocStyle::html_document vignette: \> %
% % —

## Overview

![Execution mode
comparison](../reference/figures/adapter_comparison.svg)

Execution mode comparison

`rfastlowess` provides three execution modes for different data sizes
and processing needs:

| Mode | Use Case | Memory | Key Features |
|----|----|----|----|
| **Batch** | Complete datasets in memory | Entire dataset | All features: CI, PI, CV, GPU |
| **Streaming** | Large files (\>100K points) | One chunk at a time | Chunked processing with overlap |
| **Online** | Real-time / live data | Fixed sliding window | Point-by-point incremental updates |

------------------------------------------------------------------------

## Batch Mode

The default mode. Processes the entire dataset at once and supports all
features.

### When to Use

- Dataset fits comfortably in memory
- Need confidence/prediction intervals
- Need cross-validation
- One-shot analysis

### Parameters

| Parameter              | Default | Description             |
|------------------------|---------|-------------------------|
| `fraction`             | 0.67    | Neighbourhood size      |
| `iterations`           | 0       | Robustness iterations   |
| `confidence_intervals` | `NULL`  | CI coverage (e.g. 0.95) |
| `prediction_intervals` | `NULL`  | PI coverage (e.g. 0.95) |
| `parallel`             | `FALSE` | Enable CPU parallelism  |
| `backend`              | `"cpu"` | `"cpu"` or `"gpu"`      |

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
    return_diagnostics = TRUE
)
result <- fit(model, x, y)

print(result$confidence_lower)
print(result$confidence_upper)
print(result$diagnostics$r_squared)
```

------------------------------------------------------------------------

## Streaming Mode

Processes data in fixed-size chunks with configurable overlap. Results
for each chunk are returned after calling
[`process_chunk()`](https://thisisamirv.github.io/lowess-project/r/reference/process_chunk.md).
Call
[`finalize()`](https://thisisamirv.github.io/lowess-project/r/reference/finalize.md)
after the last chunk to flush remaining buffered points.

### When to Use

- Dataset is too large to fit in memory
- Processing data from a file or stream
- Batch pipeline with memory constraints

### Parameters

| Parameter        | Default              | Description                     |
|------------------|----------------------|---------------------------------|
| `chunk_size`     | 5000                 | Points per chunk                |
| `overlap`        | 500                  | Overlap between chunks          |
| `merge_strategy` | `"weighted_average"` | How to merge overlapping values |

### Merge Strategies

| Strategy             | Behavior                              |
|----------------------|---------------------------------------|
| `"average"`          | Arithmetic mean of both estimates     |
| `"weighted_average"` | Distance-weighted blend (recommended) |
| `"take_first"`       | Keep left-chunk estimate              |
| `"take_last"`        | Keep right-chunk estimate             |

### Example

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)
x_chunk <- x[seq_len(50)]
y_chunk <- y[seq_len(50)]

model <- StreamingLowess(
    fraction = 0.3,
    iterations = 2,
    chunk_size = 5000,
    overlap = 500,
    merge_strategy = "average"
)
result <- process_chunk(model, x_chunk, y_chunk)
final  <- finalize(model)
```

------------------------------------------------------------------------

## Online Mode

Maintains a sliding window and processes each incoming point
immediately.

### When to Use

- Real-time data streams (sensors, logs)
- Each point must be smoothed as it arrives
- Memory-bounded processing with a fixed window

### Parameters

| Parameter         | Default         | Description                       |
|-------------------|-----------------|-----------------------------------|
| `window_capacity` | 1000            | Max points in sliding window      |
| `min_points`      | 2               | Minimum points before output      |
| `update_mode`     | `"incremental"` | `"incremental"` or `"full"` refit |

### Example

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
```

> **Note:** `update_mode = "incremental"` refits only the most recent
> point for lower latency. `update_mode = "full"` refits the entire
> window for higher accuracy.
