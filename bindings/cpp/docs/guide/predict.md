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
| `outputs` | `std::vector<std::string>` | `{}` | Request `"se"` and/or `"derivative"` |
| `intervals` | `IntervalsOptions` | disabled | Confidence/prediction levels and optional residual-bootstrap refits |
| `seed` | `std::optional<uint64_t>` | unset | Reproducible prediction-time bootstrap draws; zero is valid |
| `extrapolation` | `std::string` | `"clamp"` | Behavior for query points outside the training `x`-range |
| `max_extrapolation_distance` | `double` | `NaN` | Under `"linear"` extrapolation, the max allowed distance beyond the training boundary before erroring |
| `max_neighbor_distance` | `double` | `NaN` | Max allowed distance to the farthest training point in a query's local window before erroring |

### outputs

Request `"se"` for standard errors and `"derivative"` for local slopes. Analytic intervals compute their required standard errors even when `"se"` was not requested separately.

### intervals

Set `intervals.confidence` for bounds around the mean response and `intervals.prediction` for bounds of a new observation (e.g. `0.95`); both default to `NaN` (disabled). Without bootstrap, these use normal-theory standard errors and the retained residual scale.

Set `intervals.bootstrap` to at least 2 to resample the retained Batch residuals, refit the model, and calculate query-point percentile bounds and standard errors. `seed` on `PredictOptions` controls these draws independently of the seed used for fitting or CV. Bootstrap prediction requires `LowessOptions::retain_model = true`.

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

### Bootstrap Intervals

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <vector>

int main() {
    std::vector<double> x(30), y(30);
    for (int i = 0; i < 30; ++i) {
        x[i] = i * 0.1;
        y[i] = std::sin(x[i]) + 0.1 * std::cos(7.0 * x[i]);
    }

    fastlowess::LowessOptions options;
    options.retain_model = true;
    fastlowess::Lowess model(options);
    auto result = model.fit(x, y).value();
    auto retained = result.predict_model();

    fastlowess::PredictOptions predict_options;
    predict_options.outputs = {"se", "derivative"};
    predict_options.intervals.confidence = 0.95;
    predict_options.intervals.prediction = 0.95;
    predict_options.intervals.bootstrap = 40;
    predict_options.seed = 42;
    auto predicted = retained.predict({0.75, 1.25}, predict_options);
    return predicted.valid() && predicted.confidence_lower().size() == 2 ? 0 : 1;
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
