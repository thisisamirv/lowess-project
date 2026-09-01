\page use_case_time_series Time Series Analysis

# Time Series Analysis

LOWESS for trend extraction and temporal smoothing.

## Overview

Time series data often contains noise, seasonality, and trends. LOWESS provides flexible trend extraction without parametric assumptions.

---

## Basic Trend Extraction

`fraction = 0.1` sizes the neighbourhood as 10% of the data at each evaluation point — narrow enough to follow a slowly varying trend without smearing periodic variation. Three robustness `iterations` down-weight noise spikes so they cannot bias the fitted curve; this is especially important when the signal-to-noise ratio is low or when occasional outliers are expected.

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 500;
    std::vector<double> t(n), y(n);
    for (int i = 0; i < n; ++i) {
        t[i] = i * 100.0 / (n - 1);
        y[i] = 10.0 + 0.5 * t[i] + 3.0 * std::sin(t[i] / 10.0) + (std::fmod(i * 7 + 3, 1.7) - 0.85) * 3.0;
    }

    fastlowess::LowessOptions trend_opts;
    trend_opts.fraction = 0.1;
    trend_opts.iterations = 3;
    fastlowess::Lowess basic_model(trend_opts);
    auto basic_result = basic_model.fit(t, y).value();

    std::cout << "y[0]: " << basic_result.y_vector()[0] << "\n";
    return 0;
}
```

```output
y[0]: 11.3216
```

---

## Detrending

Remove trend to analyze residual patterns.

Setting `return_residuals = True` stores `observed − smoothed` alongside the smooth. A slightly wider `fraction = 0.3` produces a smoother baseline trend, so short-duration oscillations end up in the residuals rather than being absorbed into the trend component. The residual series is then ready for spectral analysis, seasonality detection, or change-point methods.

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> t(n), y(n);
    for (int i = 0; i < n; ++i) {
        t[i] = i * 2.0 * M_PI / (n - 1);
        y[i] = std::sin(t[i]) + 0.1;
    }

    fastlowess::Lowess model({
        .fraction = 0.3,
        .iterations = 3,
        .return_residuals = true
    });
    auto result = model.fit(t, y).value();

    auto trend = result.y_vector();
    auto detrended = result.residuals();

    std::cout << "residuals[0]: " << result.residuals()[0] << "\n";
    return 0;
}
```

```output
residuals[0]: -0.158244
```

---

## Forecasting with Prediction Intervals

Prediction intervals widen the uncertainty band to include both the uncertainty in the fitted curve (confidence interval) and the expected scatter of new observations around it. `fraction = 0.2` offers a balance between local detail and stable interval width — too small a fraction produces jagged interval edges; too large a fraction underestimates local variance near turning points.

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> t(n), y(n);
    for (int i = 0; i < n; ++i) {
        t[i] = i * 2.0 * M_PI / (n - 1);
        y[i] = std::sin(t[i]) + 0.1;
    }

    fastlowess::Lowess forecast_model({
        .fraction = 0.2,
        .iterations = 3,
        .confidence_intervals = 0.95,
        .prediction_intervals = 0.95
    });
    auto result = forecast_model.fit(t, y).value();

    std::cout << "95% PI: [" << result.prediction_lower()[0] << ", " << result.prediction_upper()[0] << "]\n";
    return 0;
}
```

```output
95% PI: [0.158010, 0.290883]
```

---

## Handling Missing Data

LOWESS naturally handles irregular time sampling:

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> tIrregular(n), yIrregular(n);
    for (int i = 0; i < n; ++i) {
        tIrregular[i] = i * 1.0 + (i * 31 % 10) * 0.1;
        yIrregular[i] = 10.0 + 0.3 * tIrregular[i] + 2.0 * std::sin(tIrregular[i] * 0.1);
    }

    fastlowess::Lowess missing_model({ .fraction = 0.2 });
    auto result = missing_model.fit(tIrregular, yIrregular).value();

    std::cout << "y[0]: " << result.y_vector()[0] << "\n";
    return 0;
}
```

```output
y[0]: 11.3273
```

---

## Multi-Scale Analysis

Use different fractions to extract features at different scales:

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> t(n), y(n);
    for (int i = 0; i < n; ++i) {
        t[i] = i * 2.0 * M_PI / (n - 1);
        y[i] = std::sin(t[i]) + 0.1;
    }

    std::vector<double> scales = {0.05, 0.2, 0.5};
    std::vector<std::vector<double>> trends;
    for (auto f : scales) {
        fastlowess::Lowess scale_model({ .fraction = f });
        auto result = scale_model.fit(t, y).value();
        trends.push_back(result.y_vector());
    }

    std::cout << "y[0]: " << trends[0][0] << "\n";
    return 0;
}
```

```output
y[0]: 0.131712
```

---

## Gene Expression Time Course

Biological application:

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 49;
    std::vector<double> hours(n), expression(n);
    for (int i = 0; i < n; ++i) {
        hours[i] = i * 0.5;
        expression[i] = 100.0 * (1.0 + 0.5 * std::sin(hours[i] * M_PI / 12.0)) + (std::fmod(i * 7 + 3, 1.7) - 0.85) * 10.0;
    }

    fastlowess::Lowess gene_model({
        .fraction = 0.3,
        .iterations = 3,
        .return_diagnostics = true
    });
    auto result = gene_model.fit(hours, expression).value();

    std::cout << "R2: " << result.diagnostics().r_squared() << std::endl;

    return 0;
}
```

```output
R2: 0.973054
```

---

## Choosing Fraction for Time Series

| Data Type | Recommended Fraction | Rationale |
| --- | --- | --- |
| Daily data (years) | 0.3–0.5 | Capture annual trends |
| Hourly data (days) | 0.1–0.2 | Capture daily patterns |
| Sensor data (minutes) | 0.05–0.1 | Preserve short-term features |
| Noisy data | Higher | Reduce noise impact |
| Clean data | Lower | Preserve detail |

---

## See Also

- [Real-Time Processing](use-case-real-time.md) — For streaming time series
- [Cross-Validation](cross-validation.md) — Optimal fraction selection
- [Boundary Handling](boundary.md) — Edge bias in trend extraction
- [API Reference](api.md) — Full parameter reference
