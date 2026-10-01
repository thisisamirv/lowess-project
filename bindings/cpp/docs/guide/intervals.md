\page intervals Intervals

# Intervals

Confidence and prediction intervals for uncertainty quantification.

## Overview

![Confidence and Prediction Intervals](intervals_comparison.svg)

> **Adapter support:** Confidence and prediction intervals are available in **Batch** mode, **Streaming** mode (computed per chunk and merged across overlap boundaries via `merge_strategy`, like `y_vector()`/`derivative()`), and **Online** mode when `update_mode = "full"` is set (construction fails if combined with the default `"incremental"` mode).

| Type | Represents | Width | Use |
| --- | --- | --- | --- |
| **Confidence** | Uncertainty in mean curve | Narrow | Where is the true trend? |
| **Prediction** | Uncertainty for new points | Wide | Where will new data fall? |

---

## Confidence Intervals

Estimate uncertainty in the smoothed curve itself.

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


    fastlowess::LowessOptions options;
    options.fraction = 0.5;
    options.intervals.confidence = 0.95;
    fastlowess::Lowess model(options);
    auto result = model.fit(x, y).value();

    auto ci_lower = result.confidence_lower();
    auto ci_upper = result.confidence_upper();

    std::cout << "95% CI: [" << result.confidence_lower()[0] << ", " << result.confidence_upper()[0] << "]\n";
    return 0;
}
```

```output
95% CI: [0.292929, 0.372154]
```

---

## Prediction Intervals

Estimate where new observations might fall.

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

    fastlowess::LowessOptions options;
    options.fraction = 0.5;
    options.intervals.prediction = 0.95;
    fastlowess::Lowess model(options);
    auto result = model.fit(x, y).value();

    std::cout << "95% PI: [" << result.prediction_lower()[0] << ", " << result.prediction_upper()[0] << "]\n";
    return 0;
}
```

```output
95% PI: [-0.040546, 0.705629]
```

---

## Both Intervals

Request both types simultaneously:

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

    fastlowess::LowessOptions options;
    options.fraction = 0.5;
    options.intervals.confidence = 0.95;
    options.intervals.prediction = 0.95;
    fastlowess::Lowess model(options);
    auto result = model.fit(x, y).value();

    std::cout << "95% CI: [" << result.confidence_lower()[0] << ", " << result.confidence_upper()[0] << "]\n";
    return 0;
}
```

```output
95% CI: [0.292929, 0.372154]
```

---

## Confidence Levels

Common levels and their z-values:

| Level | z-value | Interpretation |
| --- | --- | --- |
| 0.90 | 1.645 | 90% of intervals contain true value |
| 0.95 | 1.960 | 95% of intervals contain true value |
| 0.99 | 2.576 | 99% of intervals contain true value |

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

    fastlowess::LowessOptions options;
    options.intervals.confidence = 0.99;
    fastlowess::Lowess model(options);
    auto result = model.fit(x, y).value();

    std::cout << "99% CI: [" << result.confidence_lower()[0] << ", " << result.confidence_upper()[0] << "]\n";
    return 0;
}
```

```output
99% CI: [0.317464, 0.444303]
```

---

## Residual Bootstrap

Set `intervals.bootstrap` to at least 2 to replace analytic intervals with residual-bootstrap refits. `seed` is an optional outer setting shared with CV for Batch; on Streaming and Online it seeds each chunk or full-update window. Online bootstrap requires `update_mode = "full"`.

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

    fastlowess::LowessOptions batch_options;
    batch_options.intervals.confidence = 0.95;
    batch_options.intervals.prediction = 0.95;
    batch_options.intervals.bootstrap = 20;
    batch_options.seed = 42;
    fastlowess::Lowess batch(batch_options);
    auto fitted = batch.fit(x, y).value();
    if (fitted.standard_errors().size() != x.size()) return 1;

    fastlowess::StreamingOptions streaming_options;
    streaming_options.chunk_size = 30;
    streaming_options.intervals.confidence = 0.95;
    streaming_options.intervals.bootstrap = 20;
    streaming_options.seed = 42;
    fastlowess::StreamingLowess streaming(streaming_options);
    auto chunk = streaming.process_chunk(x, y).value();
    if (chunk.confidence_lower().empty()) return 1;

    fastlowess::OnlineOptions online_options;
    online_options.min_points = 5;
    online_options.update_mode = "full";
    online_options.intervals.prediction = 0.95;
    online_options.intervals.bootstrap = 20;
    online_options.seed = 42;
    fastlowess::OnlineLowess online(online_options);
    bool has_bounds = false;
    for (int i = 0; i < 30; ++i) {
        auto output = online.add_point(x[i], y[i]).value();
        if (output.has_value()) has_bounds = std::isfinite(output.prediction_lower());
    }
    return has_bounds ? 0 : 1;
}
```

---

## Standard Errors

Access standard errors directly (available when intervals are computed):

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    fastlowess::LowessOptions opts;
    opts.outputs = {"se"};
    fastlowess::Lowess model(opts);
    auto result = model.fit(x, y).value();

    auto se = result.standard_errors();
    for (int i = 0; i < 3; ++i) {
        std::cout << "Point " << i << ": SE = " << std::fixed << std::setprecision(4) << se[i] << "\n";
    }
    return 0;
}
```

```output
Point 0: SE = 0.0246
Point 1: SE = 0.0252
Point 2: SE = 0.0258
```

---

## Availability

> **Supported In All Three Adapters:** Confidence and prediction intervals are available in **Batch**, **Streaming**, and **Online** mode (`update_mode = "full"` only).

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✓ | ✓ (`update_mode = "full"` only) |
| Prediction intervals | ✓ | ✓ | ✓ (`update_mode = "full"` only) |
| Standard errors | ✓ | ✓ | ✓ (`update_mode = "full"` only) |
