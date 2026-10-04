\page api_online OnlineLowess API

# OnlineLowess API

See also: [fastLowess](api.md)

## When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

![Online Adapter](online_comparison.svg)

## Class

### fastlowess::OnlineLowess

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    fastlowess::OnlineOptions opts;
    opts.fraction = 0.5;
    opts.window_capacity = 50;
    opts.min_points = 3;
    fastlowess::OnlineLowess model(opts);

    auto r = model.add_point(x[0], y[0]).value();
    auto r2 = model.add_point(x[1], y[1]).value();
    auto r3 = model.add_point(x[2], y[2]).value();
    if (r3.has_value()) { std::cout << "y: " << r3.y() << "\n"; }
    return 0;
}
```

```output
y: 0.226592
```

- `options`: An `OnlineOptions` struct (inherits from `LowessOptions`) with `window_capacity`, `min_points`, and `update_mode`.

#### `add_point(x, y)`

Adds a single point to the sliding window and returns the smoothed value for that point, or a result with `has_value() == false` while the window is still filling up (fewer than `min_points` seen so far). Once the window reaches `window_capacity`, each new point evicts the oldest one, so memory stays bounded regardless of how much history has passed through. `update_mode` controls how much work each call does: `"incremental"` re-fits only the newest point, while `"full"` re-smooths the entire window for a more accurate but slower result.

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    fastlowess::OnlineOptions opts;
    opts.fraction = 0.5;
    opts.window_capacity = 50;
    opts.min_points = 3;
    fastlowess::OnlineLowess model(opts);

    // Returns OnlineOutput with has_value() == false until min_points (3) are reached
    auto r1 = model.add_point(x[0], y[0]).value();  // r1.has_value() == false
    auto r2 = model.add_point(x[1], y[1]).value();  // r2.has_value() == false

    // Returns OnlineOutput with has_value() == true once enough points are available
    auto r3 = model.add_point(x[2], y[2]).value();
    if (r3.has_value()) {
        std::cout << r3.y() << std::endl;  // 0.22659245357374927
    }

    return 0;
}
```

```output
0.226592
```

## Options Structure

### OnlineOptions

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `double` | 0.67 | Smoothing fraction (bandwidth) |
| `iterations` | `int` | 0 | Number of robustifying iterations (requires `update_mode = "full"`) |
| `weight_function` | `std::string` | "tricube" | Weight function name |
| `robustness_method` | `std::string` | "bisquare" | Robustness method name |
| `delta` | `double` | NaN | Interpolation distance (`NaN` disables interpolation); positive values require `update_mode = "full"` |
| `zero_weight_fallback` | `std::string` | "use_local_mean" | Zero-weight handling |
| `boundary_policy` | `std::string` | "extend" | Boundary handling policy |
| `scaling_method` | `std::string` | "mad" | Residual scaling method |
| `auto_converge` | `double` | NaN | Auto-convergence tolerance; requires `update_mode = "full"` and `iterations > 0` |
| `missing` | `std::string` | "error" | Policy for non-finite (NaN/Inf) values in each point |
| `window_capacity` | `int` | 1000 | Max points in sliding window |
| `min_points` | `int` | 2 | Min points before smoothing starts |
| `update_mode` | `std::string` | "incremental" | Update mode (`"full"` or `"incremental"`) |
| `outputs` | `std::vector<std::string>` | `{}` | Request `"se"`, `"weights"`, or `"derivative"` |
| `intervals` | `IntervalsOptions` | disabled | Confidence/prediction levels and per-window bootstrap refits (`update_mode = "full"` only) |
| `seed` | `std::optional<uint64_t>` | unset | Reproducible bootstrap draws for each full-update window |

Incremental mode fits only the newest point. Positive `delta` is rejected there, and `auto_converge` requires full mode with at least one robustness iteration.

Cross-validation, GPU `backend`, `custom_weights`, the `"sorted"`, `"diagnostics"`, and `"residuals"` outputs, and `parallel` are Batch-only (or Batch/Streaming-only) and not available here; see [fastLowess](api.md) for those.

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

`iterations` controls robustness to outliers, at the cost of speed. Requires `update_mode = "full"`; the default `"incremental"` mode performs a non-robust single-point fit.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

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

### delta

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `NaN` (default) auto-sets it to `0` in Online mode, i.e. interpolation is disabled and every point is fit exactly.

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### boundary_policy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### auto_converge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `NaN` (default) disables early stopping.

### missing

Policy for handling a non-finite (NaN/Inf) `x` or `y` value passed to `add_point`:

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Return an error result |
| `"drop"` | Silently ignore the point — `add_point` succeeds but returns an `OnlineOutput` whose `has_value()` is `false` instead of adding it to the window |

### window_capacity

Maximum number of most recent points kept in the sliding window; older points are discarded as new ones arrive. Each `add_point()` call costs O(`window_capacity`) rather than growing with total history.

### min_points

Minimum number of points required before smoothing starts. `add_point()` returns a result with `has_value() == false` until the window reaches this size.

### update_mode

*See: [Execution Modes](../guide/adapter-choice.md)*

| Mode | Alias | Behavior | Speed |
| --- | --- | --- | --- |
| `"incremental"` (default) | `"single"` | Update only affected fits | Faster |
| `"full"` | `"resmooth"` | Recompute entire window | More accurate |

### outputs

Request `"derivative"` for the latest point's slope or `"weights"` for its robustness weight. Request `"se"` to populate `standard_error()` with `update_mode = "full"`; incremental mode rejects it at construction.

### intervals

*See: [Intervals](../guide/intervals.md)*

Set `intervals.confidence` and/or `intervals.prediction` to a coverage level such as `0.95`, or leave them as `NaN` to disable. `intervals.bootstrap` (at least 2) refits the current full-update window for percentile intervals and standard errors. Set `seed` for reproducible draws. All interval settings require `update_mode = "full"`.

## Result Structure

### fastlowess::OnlineOutput

Returned (inside `Expected`) by `add_point()`. Check `has_value()` before reading fields.

| Method | Return Type | Description |
| --- | --- | --- |
| `has_value()` | `bool` | `false` while window fills; `true` when output is ready |
| `y()` | `double` | Smoothed value for the latest point |
| `standard_error()` | `double` | Populated when `"se"` or intervals are requested (full mode only); NaN otherwise |
| `confidence_lower()` / `confidence_upper()` | `double` | Bounds around the mean response if `intervals.confidence` was set (full mode only) |
| `prediction_lower()` / `prediction_upper()` | `double` | Bounds for a new observation if `intervals.prediction` was set (full mode only) |
| `residual()` | `double` | Residual y − smoothed; always populated (no output selection is needed) |
| `robustness_weight()` | `double` | Robustness weight, if `outputs` contains `"weights"` |
| `iterations_used()` | `int` | Robustness iterations performed (−1 if N/A) |
| `derivative()` | `double` | Local fit derivative/slope if `"derivative"` was requested (NaN otherwise) |

There is no `Diagnostics` object or `"diagnostics"` output for `OnlineLowess`: `OnlineOutput` carries no diagnostics field, since diagnostics like RMSE/R² need more than one point's worth of history to be meaningful.
