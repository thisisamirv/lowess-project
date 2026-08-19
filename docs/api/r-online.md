# OnlineLowess — R API Reference

See also: [fastLowess R API Reference](r.md)

## Class

### `OnlineLowess`

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

```r
library(rfastlowess)

online <- OnlineLowess(fraction = 0.5, window_capacity = 50L)
print(online)
#> <OnlineLowess Model>
#>   Fraction:          0.5
#>   Window Capacity:   50
#>   Min Points:        3
```

**Methods:**

```r
library(rfastlowess)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + 0.1

online <- OnlineLowess(fraction = 0.5, window_capacity = 50L)

# Returns NULL until min_points (3) are reached
result <- add_point(online, x[[1L]], y[[1L]])  # NULL
result <- add_point(online, x[[2L]], y[[2L]])  # NULL

# Returns a named list once enough points are available
result <- add_point(online, x[[3L]], y[[3L]])
cat(result$smoothed)
#> 0.2266
```

* Adds a single point to the sliding window. Returns a named list (`$smoothed`, `$residual`, …) once the window has enough points, or `NULL` while still filling.

## Options Structure

### `OnlineOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `integer` | `1000L` | Max points in sliding window |
| `min_points` | `integer` | `3L` | Min points before smoothing starts |
| `update_mode` | `character` | `"full"` | Update mode (`"full"` or `"incremental"`) |
| `parallel` | `logical` | `FALSE` | Enable parallel execution (off by default; online LOWESS fits one point at a time) |

## Result Structure

### `OnlineOutput` (named list)

Returned by `add_point()` once the window has enough points (`NULL` until then).

| Field | Type | Description |
| --- | --- | --- |
| `smoothed` | `numeric` | Smoothed value for the latest point |
| `std_error` | `numeric` (optional) | Standard error (if requested) |
| `residual` | `numeric` (optional) | Residual y − smoothed (if requested) |
| `robustness_weight` | `numeric` (optional) | Robustness weight (if requested) |
| `iterations_used` | `integer` (optional) | Robustness iterations performed |

## Options

### update_mode

*See: [Execution Modes](../user-guide/adapters.md)*

* `"full"` (default; alias: `"resmooth"`)
* `"incremental"` (alias: `"single"`)
