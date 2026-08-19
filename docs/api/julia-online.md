# OnlineLowess — Julia API Reference

See also: [FastLOWESS Julia API Reference](julia.md)

## Struct

### `OnlineLowess`

The `OnlineLowess` struct updates the model incrementally with new data points.

**Constructor:**

```julia
using FastLOWESS

online = OnlineLowess(fraction=0.5, window_capacity=50)
```

**Methods:**

```julia
using FastLOWESS

x = collect(range(0, 2π, length=100))
y = sin.(x) .+ 0.1

online = OnlineLowess(fraction=0.5, window_capacity=50)

# Returns nothing until min_points (3) are reached
result = add_point(online, x[1], y[1])  # nothing
result = add_point(online, x[2], y[2])  # nothing

# Returns OnlineOutput once enough points are available
result = add_point(online, x[3], y[3])
println(result.y)
# 0.22659245357374927
```

* Adds a single point to the sliding window. Returns `nothing` while the window is still filling (fewer than `min_points` seen), and an `OnlineOutput` once smoothing begins.

## Options Structure

### `OnlineOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `Int` | `1000` | Max points in sliding window |
| `min_points` | `Int` | `3` | Min points before smoothing starts |
| `update_mode` | `String` | `"full"` | Update mode (`"full"` or `"incremental"`) |
| `parallel` | `Bool` | `false` | Enable parallel execution (off by default; online LOWESS fits one point at a time) |

## Result Structure

### `OnlineOutput`

Returned by `add_point()` once the window has enough points (`nothing` until then).

| Field | Type | Description |
| --- | --- | --- |
| `y` | `Float64` | Smoothed value for the latest point |
| `standard_error` | `Union{Float64, Nothing}` | Standard error (if requested) |
| `residual` | `Union{Float64, Nothing}` | Residual y − smoothed (if requested) |
| `robustness_weight` | `Union{Float64, Nothing}` | Robustness weight (if requested) |
| `iterations_used` | `Union{Int, Nothing}` | Robustness iterations performed |

## Options

### update_mode

*See: [Execution Modes](../user-guide/adapters.md)*

* `"full"` (default; alias: `"resmooth"`)
* `"incremental"` (alias: `"single"`)
