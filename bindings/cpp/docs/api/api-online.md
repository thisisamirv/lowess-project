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

### OnlineOptions (inherits LowessOptions)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `double` | 0.67 | Smoothing fraction (bandwidth) |
| `iterations` | `int` | 3 | Number of robustifying iterations |
| `delta` | `double` | NaN | Interpolation distance (`NaN` auto-sets it to 0.0 in Online, i.e. interpolation disabled) |
| `weight_function` | `std::string` | "tricube" | Weight function name |
| `robustness_method` | `std::string` | "bisquare" | Robustness method name |
| `scaling_method` | `std::string` | "mad" | Residual scaling method |
| `boundary_policy` | `std::string` | "extend" | Boundary handling policy |
| `zero_weight_fallback` | `std::string` | "use_local_mean" | Zero-weight handling |
| `auto_converge` | `double` | NaN | Auto-convergence tolerance |
| `return_robustness_weights` | `bool` | false | Include `robustness_weight()` in result |
| `window_capacity` | `int` | 1000 | Max points in sliding window |
| `min_points` | `int` | 2 | Min points before smoothing starts |
| `update_mode` | `std::string` | "incremental" | Update mode (`"full"` or `"incremental"`) |

Confidence/prediction intervals, standard errors, cross-validation, GPU `backend`, `custom_weights`, `return_sorted`, `return_diagnostics`, `return_residuals`, and `parallel` are Batch-only (or Batch/Streaming-only) and not available here; see [fastLowess](api.md) for those.

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

Maximum number of most recent points kept in the sliding window; older points are discarded as new ones arrive. Each `add_point()` call costs O(`window_capacity`) rather than growing with total history.

### min_points

Minimum number of points required before smoothing starts. `add_point()` returns a result with `has_value() == false` until the window reaches this size.

### update_mode

*See: [Execution Modes](../guide/adapter-choice.md)*

| Mode | Alias | Behavior | Speed |
| --- | --- | --- | --- |
| `"incremental"` (default) | `"single"` | Update only affected fits | Faster |
| `"full"` | `"resmooth"` | Recompute entire window | More accurate |

## Result Structure

### fastlowess::OnlineOutput

Returned (inside `Expected`) by `add_point()`. Check `has_value()` before reading fields.

| Method | Return Type | Description |
| --- | --- | --- |
| `has_value()` | `bool` | `false` while window fills; `true` when output is ready |
| `y()` | `double` | Smoothed value for the latest point |
| `standard_error()` | `double` | Always NaN — standard errors require `return_se`/confidence intervals, which are Batch-only |
| `residual()` | `double` | Residual y − smoothed; always populated (there is no `return_residuals` option for Online) |
| `robustness_weight()` | `double` | Robustness weight, if `return_robustness_weights` was set |
| `iterations_used()` | `int` | Robustness iterations performed (−1 if N/A) |

There is no `Diagnostics` object or `return_diagnostics` option for `OnlineLowess`: `OnlineOutput` carries no diagnostics field, since diagnostics like RMSE/R² need more than one point's worth of history to be meaningful.
