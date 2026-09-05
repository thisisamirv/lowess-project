\page quickstart Quick Start

# Quick Start

Get up and running with LOWESS in minutes.

## Basic Smoothing

Smooth a noisy sine wave — the kind of signal where LOWESS shines. Each example recovers the underlying trend from 100 points of Gaussian noise.

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    // 100-point noisy sine wave (deterministic)
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + ((i * 7 + 3) % 17 / 17.0 - 0.5) * 0.6;
    }

    fastlowess::Lowess model({ .fraction = 0.3, .iterations = 3 });
    auto result = model.fit(x, y).value();

    std::cout << "First smoothed value: " << result.y_vector()[0]
              << "  (true: " << std::sin(x[0]) << ")\n";
    return 0;
}
```

```output
First smoothed value: 0.02784  (true: 0)
```

---

## With Confidence Intervals

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
    options.iterations = 3;
    options.confidence_intervals = 0.95;
    options.prediction_intervals = 0.95;
    options.return_diagnostics = true;

    fastlowess::Lowess model(options);
    auto result = model.fit(x, y).value();

    // Access standard C++ vectors
    auto lower = result.confidence_lower();
    auto upper = result.confidence_upper();
    double r2 = result.diagnostics().r_squared();

    std::cout << "95% CI: [" << result.confidence_lower()[0] << ", " << result.confidence_upper()[0] << "]\n";
    return 0;
}
```

```output
95% CI: [0.29413, 0.374611]
```

---

## Handling Outliers

LOWESS can robustly handle outliers through iterative reweighting:

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    // Data with an outlier at index 3
    std::vector<double> x_out = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<double> y_outlier = {2.0, 4.0, 6.0, 50.0, 10.0, 12.0};

    fastlowess::LowessOptions options;
    options.fraction = 0.7;
    options.iterations = 5;
    options.robustness_method = "bisquare";
    options.return_robustness_weights = true;

    fastlowess::Lowess model(options);
    auto result = model.fit(x_out, y_outlier).value();

    // Check weights
    auto weights = result.robustness_weights();
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights[i] < 0.5) {
            std::cout << "Point " << i << " is outlier (weight: " << weights[i] << ")\n";
        }
    }

    return 0;
}
```

```output
Point 3 is outlier (weight: 0)
```

---

## Streaming Mode

For datasets too large to fit in memory, stream them in fixed-size chunks with overlap.

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 5000;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 10 * M_PI / (n - 1);
        y[i] = std::sin(x[i] / M_PI) * std::exp(-x[i] / 30.0)
             + ((i * 7 + 3) % 17 / 17.0 - 0.5) * 0.3;
    }

    fastlowess::StreamingOptions opts;
    opts.fraction   = 0.2;
    opts.chunk_size = 1000;
    opts.overlap    = 100;

    fastlowess::StreamingLowess model(opts);

    for (int start = 0; start <= 4000; start += 1000) {
        int end = std::min(start + 1000, n);
        model.process_chunk(
            std::vector<double>(x.begin() + start, x.begin() + end),
            std::vector<double>(y.begin() + start, y.begin() + end)
        );
    }
    auto result = model.finalize().value();
    std::cout << "Smoothed " << result.y_vector().size() << " points in streaming mode\n";
    return 0;
}
```

```output
Smoothed 100 points in streaming mode
```

---

## Next Steps

| Topic | Link |
| --- | --- |
| How LOWESS works | [Concepts](concepts.md) |
| All parameters explained | [API Reference](../api/api.md) |
| Batch vs Streaming vs Online | [Execution Modes](../guide/adapter-choice.md) |
| Edge handling | [Boundary](../advanced/boundary.md) |
| Outlier handling in depth | [Robustness](../weighting/robustness.md) |
| Full API per language | [API Reference](../api/api.md) |
