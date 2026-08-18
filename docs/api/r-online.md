# OnlineLowess — R API Reference

See also: [fastLowess R API Reference](r.md)

## Class

### `OnlineLowess`

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

```r
library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

library(rfastlowess)
online <- OnlineLowess(fraction = 0.3, window_capacity = 50)
```

**Methods:**

```r
library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

online <- OnlineLowess(fraction = 0.3, window_capacity = 50)
result <- add_point(online, x[[1L]], y[[1L]])  # returns list or NULL
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
