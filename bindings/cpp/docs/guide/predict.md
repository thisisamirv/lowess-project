\page guide_predict Out-of-Sample Prediction

# Out-of-Sample Prediction

Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

Out-of-sample prediction is available in **Batch** mode only. Streaming and Online modes do not support it.

`fastlowess::PredictModel::predict(new_x, options)` evaluates the fit at arbitrary query points, like R's `predict(model, newdata)`.

It reuses `fit()`'s own (possibly `delta`-interpolated) smoothed curve for its `y` output — so predicting at a training `x` always exactly reproduces that point's `fit()` output, regardless of `delta`. A fresh local fit is only run when `"derivative"`, `"se"` (or an interval level), or `max_neighbor_distance` needs the actual regression slope or standard error.

Requires `retain_model = true` on `LowessOptions` before `fit()`; obtain the `PredictModel` via `LowessResult::predict_model()` (moves the retained state out — only valid once, check `PredictModel::valid()`).

---

## Options

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `return_se` | `bool` | `false` | Include standard errors in the output |
| `confidence_level` | `double` | `NaN` | Confidence interval coverage level (e.g. `0.95`; NaN to disable) |
| `prediction_level` | `double` | `NaN` | Prediction interval coverage level (e.g. `0.95`; NaN to disable) |
| `return_derivative` | `bool` | `false` | Include the local fit's derivative (slope) at each query point |
| `extrapolation` | `std::string` | `"clamp"` | Behavior for query points outside the training `x`-range |
| `max_extrapolation_distance` | `double` | `NaN` | Under `"linear"` extrapolation, the max allowed distance beyond the training boundary before erroring |
| `max_neighbor_distance` | `double` | `NaN` | Max allowed distance to the farthest training point in a query's local window before erroring |

### return_se

Computes standard errors for each query point, using the retained model's residual scale and per-point leverage. Required for `confidence_level`/`prediction_level` to be populated. `false` by default.

### confidence_level

Confidence level for the confidence interval around the mean response at each query point (e.g. `0.95`). Uses the same z-score convention as `fit()`'s own confidence intervals. `NaN` (default) disables it.

### prediction_level

Confidence level for the prediction interval for a new observation at each query point (e.g. `0.95`). Widens using the same MAD-based residual scale `fit()` uses for its own intervals. `NaN` (default) disables it.

### return_derivative

Includes the local fit's derivative (slope) at each query point in the output. `false` by default.

### extrapolation

Behavior for query points outside `[min(x_train), max(x_train)]`:

| Policy | Behavior |
| --- | --- |
| `"clamp"` (default) | Clamps the query to the nearest boundary window |
| `"linear"` | Linearly extrapolates from the nearest boundary point's local fit and slope |
| `"error"` | Fails the whole call with an error |

### max_extrapolation_distance

Under `"linear"` extrapolation, the maximum allowed distance beyond the training boundary before `predict()` errors, instead of returning an unbounded value. `NaN` (default, uncapped).

### max_neighbor_distance

Maximum allowed distance to the farthest training point in a query's local window before `predict()` errors. Guards against an "empty range" blind spot: a query point can fall within `[min(x_train), max(x_train)]` yet still be far from any real training point (e.g. training `x` in `[0,10]` and `[90,100]`, query at `x=50`). `NaN` (default, uncapped); applies regardless of `extrapolation`.

## Example

### Basic Usage

```cpp
#include <fastlowess.hpp>
#include <iostream>
#include <vector>

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2.1, 4.0, 6.2, 8.0, 10.1};

    fastlowess::LowessOptions opts;
    opts.fraction = 0.7;
    opts.retain_model = true;
    fastlowess::Lowess model(opts);
    auto result = model.fit(x, y).value();

    auto predict_model = result.predict_model();
    auto prediction = predict_model.predict({1.5, 4.5});
    for (double v : prediction.y()) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

```output
3.05 9.05
```

### Standard Errors and Derivative

```cpp
#include <fastlowess.hpp>
#include <iostream>
#include <vector>

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2.1, 4.0, 6.2, 8.0, 10.1};

    fastlowess::LowessOptions opts;
    opts.fraction = 0.7;
    opts.retain_model = true;
    fastlowess::Lowess model(opts);
    auto result = model.fit(x, y).value();

    auto predict_model = result.predict_model();
    fastlowess::PredictOptions popts;
        popts.outputs = {"se", "derivative"};
    auto prediction = predict_model.predict({2.5}, popts);
    return 0;
}
```

### Linear Extrapolation

```cpp
#include <fastlowess.hpp>
#include <iostream>
#include <vector>

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2.1, 4.0, 6.2, 8.0, 10.1};

    fastlowess::LowessOptions opts;
    opts.fraction = 0.7;
    opts.retain_model = true;
    fastlowess::Lowess model(opts);
    auto result = model.fit(x, y).value();

    auto predict_model = result.predict_model();
    fastlowess::PredictOptions popts;
    popts.extrapolation = "linear";
    auto prediction = predict_model.predict({10.0}, popts);
    return 0;
}
```
