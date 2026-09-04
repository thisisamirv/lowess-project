# Online Adapter

Incremental updates with a sliding window for real-time data.

See also: [Batch Adapter](api.md)

## When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

![Online Adapter](../assets/online_comparison.svg)

## Class

### `OnlineLowess`

The `OnlineLowess` type updates the model incrementally with new data points.

**Constructor:**

```@example online
using FastLOWESS

model = OnlineLowess(; fraction=0.2, window_capacity=100, min_points=5)
println(typeof(model))
```

- Keyword arguments configure the `OnlineLowess` model; see [Options Structure](#options-structure) below.

**Methods:**

#### `add_point(model, x, y)`

Adds a single point to the sliding window and returns the smoothed value for that point, or `nothing` while the window is still filling up (fewer than `min_points` seen so far). Once the window reaches `window_capacity`, each new point evicts the oldest one, so memory stays bounded regardless of how much history has passed through. `update_mode` controls how much work each call does: `"incremental"` re-fits only the newest point, while `"full"` re-smooths the entire window for a more accurate but slower result.

```@example online
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

shown = 0
for i in eachindex(x)
    global shown
    result = add_point(model, x[i], y[i])
    if result !== nothing && shown < 5
        println("Current smoothed value: ", result.y)
        shown += 1
    end
end
```

## Options Structure

### `OnlineLowess` keyword arguments (mirrors `Lowess`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `Float64` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `Int` | `3` | Number of robustifying iterations |
| `delta` | `Float64` | `NaN` | Interpolation distance (`NaN` auto-sets it to 0.0 in Online, i.e. interpolation disabled) |
| `weight_function` | `String` | `"tricube"` | Weight function name |
| `robustness_method` | `String` | `"bisquare"` | Robustness method name |
| `scaling_method` | `String` | `"mad"` | Residual scaling method |
| `boundary_policy` | `String` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `String` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge` | `Float64` | `NaN` | Auto-convergence tolerance |
| `return_robustness_weights` | `Bool` | `false` | Include `robustness_weight` in result |
| `window_capacity` | `Int` | `1000` | Max points in sliding window |
| `min_points` | `Int` | `2` | Min points before smoothing starts |
| `update_mode` | `String` | `"incremental"` | Update mode (`"full"` or `"incremental"`) |

Confidence/prediction intervals, standard errors, cross-validation, GPU `backend`, `custom_weights`, `return_sorted`, `return_diagnostics`, `return_residuals`, and `parallel` are Batch-only (or Batch/Streaming-only) and not available here; see [Batch Adapter](api.md) for those.

## Options

### fraction

`fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

### iterations

`iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

### delta

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `NaN` (default) auto-sets it to `0` in Online mode, i.e. interpolation is disabled and every point is fit exactly.

### weight_function

*See: [Weight Functions](../weighting/kernels.md)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](../weighting/robustness.md)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### scaling_method

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### boundary_policy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### auto_converge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `NaN` (default) disables early stopping.

### window_capacity

Maximum number of most recent points kept in the sliding window; older points are discarded as new ones arrive. Each `add_point` call costs O(`window_capacity`) rather than growing with total history.

### min_points

Minimum number of points required before smoothing starts. `add_point` returns `nothing` until the window reaches this size.

### update_mode

*See: [Execution Modes](../guide/adapter-choice.md)*

| Mode | Behavior | Speed |
| --- | --- | --- |
| `"incremental"` (default) | Update only affected fits | Faster |
| `"full"` | Recompute entire window | More accurate |

## Result Structure

### `OnlineOutput`

Returned by `add_point` once the window has enough points (`nothing` until then).

| Field | Type | Description |
| --- | --- | --- |
| `y` | `Float64` | Smoothed value for the latest point |
| `standard_error` | `Union{Float64, Nothing}` | Always `nothing` — standard errors require `return_se`/confidence intervals, which are Batch-only |
| `residual` | `Union{Float64, Nothing}` | Residual y − smoothed; always present (there is no `return_residuals` option for Online) |
| `robustness_weight` | `Union{Float64, Nothing}` | Robustness weight, if `return_robustness_weights` was set |
| `iterations_used` | `Union{Int, Nothing}` | Robustness iterations performed |

There is no `Diagnostics` object or `return_diagnostics` option for `OnlineLowess`: `OnlineOutput` carries no diagnostics field, since diagnostics like RMSE/R2 need more than one point's worth of history to be meaningful.

## Example

```@example online
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

model = OnlineLowess(;
    fraction=0.2,
    iterations=1,
    window_capacity=100,
    min_points=5,
    update_mode="incremental"
)
shown = 0
for i in eachindex(x)
    global shown
    result = add_point(model, x[i], y[i])
    if result !== nothing && shown < 5
        println("Current smoothed value: ", result.y)
        shown += 1
    end
end
```

---
