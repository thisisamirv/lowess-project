# OnlineLowess — C++ API Reference

See also: [fastLowess C++ API Reference](cpp.md)

## Class

### `fastlowess::OnlineLowess`

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
    opts.fraction = 0.3;
    opts.window_capacity = 50;
    opts.min_points = 3;
    fastlowess::OnlineLowess model(opts);

    return 0;
}
```

* `options`: An `OnlineOptions` struct (inherits from `LowessOptions`) with `window_capacity`, `min_points`, and `update_mode`.

**Methods:**

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
    opts.fraction = 0.3;
    opts.window_capacity = 50;
    opts.min_points = 3;
    fastlowess::OnlineLowess model(opts);

    // Returns OnlineOutput with has_value() == false until min_points (3) are reached
    auto r1 = model.add_point(x[0], y[0]).value();  // r1.has_value() == false
    auto r2 = model.add_point(x[1], y[1]).value();  // r2.has_value() == false

    // Returns OnlineOutput with has_value() == true once enough points are available
    auto r3 = model.add_point(x[2], y[2]).value();
    if (r3.has_value()) {
        std::cout << r3.smoothed() << std::endl;  // 0.22659245357374927
    }

    return 0;
}
```

* Adds a single point to the sliding window. Returns `Expected<OnlineOutput>` — check `has_value()` to see whether the window is ready.

## Options Structure

### `OnlineOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `int` | 1000 | Max points in sliding window |
| `min_points` | `int` | 3 | Min points before smoothing starts |
| `update_mode` | `std::string` | "full" | Update mode (`"full"` or `"incremental"`) |
| `parallel` | `bool` | `false` | Enable parallel execution (off by default; online LOWESS fits one point at a time) |

## Result Structure

### `fastlowess::OnlineOutput`

Returned (inside `Expected`) by `add_point()`. Check `has_value()` before reading fields.

| Method | Return Type | Description |
| --- | --- | --- |
| `has_value()` | `bool` | `false` while window fills; `true` when output is ready |
| `smoothed()` | `double` | Smoothed value for the latest point |
| `std_error()` | `double` | Standard error (NaN if not computed) |
| `residual()` | `double` | Residual y − smoothed (NaN if not computed) |
| `robustness_weight()` | `double` | Robustness weight (NaN if not computed) |
| `iterations_used()` | `int` | Robustness iterations performed (−1 if N/A) |

## Options

### update_mode

*See: [Execution Modes](../user-guide/adapters.md)*

* `"full"` (default; alias: `"resmooth"`)
* `"incremental"` (alias: `"single"`)
