<!-- markdownlint-disable MD024 MD046 MD033 MD037 -->
# Time Series Analysis

LOWESS for trend extraction and temporal smoothing.

## Overview

Time series data often contains noise, seasonality, and trends. LOWESS provides flexible trend extraction without parametric assumptions.

---

## Basic Trend Extraction

`fraction = 0.1` sizes the neighbourhood as 10% of the data at each evaluation point — narrow enough to follow a slowly varying trend without smearing periodic variation. Three robustness `iterations` down-weight noise spikes so they cannot bias the fitted curve; this is especially important when the signal-to-noise ratio is low or when occasional outliers are expected.

=== "Python"
    ```python
    import fastlowess as fl
    import numpy as np
    import matplotlib.pyplot as plt

    # Simulate noisy time series with trend
    np.random.seed(42)
    t = np.linspace(0, 100, 500)
    trend = 10 + 0.5 * t + 3 * np.sin(t / 10)
    noise = np.random.normal(0, 3, len(t))
    y = trend + noise

    # Extract trend with LOWESS
    model = fl.Lowess(fraction=0.1, iterations=3)
    result = model.fit(t, y)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(t, y, "gray", alpha=0.5, label="Observed")
    plt.plot(t, result.y, "b-", linewidth=2, label="Trend (LOWESS)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Trend Extraction")
    plt.show()
    ```
=== "WebAssembly"
    ```javascript
    const { Lowess } = require('fastlowess-wasm');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const model = new Lowess({ 
        fraction: 0.1, 
        iterations: 3 
    });
    const result = model.fit(x, y);

    // Trend values in result.y
    ```

=== "C++"
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

        fastlowess::LowessOptions trend_opts;
        trend_opts.fraction = 0.1;
        trend_opts.iterations = 3;
        fastlowess::Lowess basic_model(trend_opts);
        auto basic_result = basic_model.fit(t, y).value();

        // Trend in basic_result.y_vector()

        return 0;
    }
    ```

---

## Detrending

Remove trend to analyze residual patterns.

Setting `return_residuals = True` stores `observed − smoothed` alongside the smooth. A slightly wider `fraction = 0.3` produces a smoother baseline trend, so short-duration oscillations end up in the residuals rather than being absorbed into the trend component. The residual series is then ready for spectral analysis, seasonality detection, or change-point methods.

=== "Python"
    ```python
    import fastlowess as fl
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    t = np.linspace(0, 100, 500)
    trend_true = 10 + 0.5 * t + 3 * np.sin(t / 10)
    y = trend_true + np.random.normal(0, 3, len(t))

    # Smooth to get trend
    model = fl.Lowess(fraction=0.3, iterations=3, return_residuals=True)
    result = model.fit(t, y)

    trend = result.y
    detrended = result.residuals

    # Analyze residuals for seasonality, etc.
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, trend)
    plt.title("Extracted Trend")

    plt.subplot(1, 2, 2)
    plt.plot(t, detrended)
    plt.title("Detrended (Residuals)")
    plt.tight_layout()
    ```
=== "WebAssembly"
    ```javascript
    const { Lowess } = require('fastlowess-wasm');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const model = new Lowess({ 
        fraction: 0.3, 
        iterations: 3, 
        return_residuals: true 
    });
    const result = model.fit(x, y);

    // Access result.y (trend) and result.residuals (detrended)
    ```

=== "C++"
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

        return 0;
    }
    ```

---

## Forecasting with Prediction Intervals

Prediction intervals widen the uncertainty band to include both the uncertainty in the fitted curve (confidence interval) and the expected scatter of new observations around it. `fraction = 0.2` offers a balance between local detail and stable interval width — too small a fraction produces jagged interval edges; too large a fraction underestimates local variance near turning points.

=== "Python"
    ```python
    import fastlowess as fl
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    t = np.linspace(0, 100, 500)
    trend_true = 10 + 0.5 * t + 3 * np.sin(t / 10)
    y = trend_true + np.random.normal(0, 3, len(t))

    model = fl.Lowess(
        fraction=0.2,
        iterations=3,
        confidence_intervals=0.95,
        prediction_intervals=0.95
    )
    result = model.fit(t, y)

    # Plot with uncertainty bands
    plt.figure(figsize=(12, 5))
    plt.plot(t, y, "gray", alpha=0.3)
    plt.plot(t, result.y, "b-", linewidth=2, label="Trend")
    plt.fill_between(
        t,
        result.prediction_lower,
        result.prediction_upper,
        alpha=0.2, color="blue", label="95% Prediction"
    )
    plt.legend()
    ```
=== "WebAssembly"
    ```javascript
    const { Lowess } = require('fastlowess-wasm');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const model = new Lowess({
        fraction: 0.2,
        iterations: 3,
        prediction_intervals: 0.95
    });
    const result = model.fit(x, y);

    // Access result.prediction_lower and result.prediction_upper
    ```

=== "C++"
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

        // Access result.prediction_lower() and result.prediction_upper()

        return 0;
    }
    ```

---

## Handling Missing Data

LOWESS naturally handles irregular time sampling:

=== "Python"
    ```python
    import fastlowess as fl
    import numpy as np

    rng = np.random.default_rng(42)
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x) + rng.normal(0, 0.3, 100)

    # Irregular time points (gaps in data)
    t_irregular = np.sort(np.random.uniform(0, 100, 200))
    y_irregular = 10 + t_irregular * 0.3 + np.random.normal(0, 2, 200)

    # LOWESS handles this seamlessly
    model = fl.Lowess(fraction=0.2)
    result = model.fit(t_irregular, y_irregular)
    ```
=== "WebAssembly"
    ```javascript
    const { Lowess } = require('fastlowess-wasm');

    const n = 100;
    const tIrregular = Float64Array.from({ length: n }, (_, i) => i * 1.0 + (i * 31 % 10) * 0.1).sort((a, b) => a - b);
    const yIrregular = Float64Array.from(tIrregular, t => 10 + 0.3 * t + 2.0 * Math.sin(t * 0.1));
    const model = new Lowess({ fraction: 0.2 });
    const result = model.fit(tIrregular, yIrregular);
    ```

=== "C++"
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

        return 0;
    }
    ```

---

## Multi-Scale Analysis

Use different fractions to extract features at different scales:

=== "Python"
    ```python
    import fastlowess as fl
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    t = np.linspace(0, 100, 500)
    trend_true = 10 + 0.5 * t + 3 * np.sin(t / 10)
    y = trend_true + np.random.normal(0, 3, len(t))

    # Multiple smoothing scales
    fractions = [0.05, 0.2, 0.5]

    plt.figure(figsize=(12, 5))
    plt.plot(t, y, "gray", alpha=0.3, label="Data")

    for f in fractions:
        model = fl.Lowess(fraction=f)
        result = model.fit(t, y)
        plt.plot(t, result.y, label=f"fraction={f}")

    plt.legend()
    plt.title("Multi-Scale LOWESS")
    ```
=== "WebAssembly"
    ```javascript
    const { Lowess } = require('fastlowess-wasm');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const trends = [0.05, 0.2, 0.5].map(f => {
        const model = new Lowess({ fraction: f });
        const result = model.fit(x, y);
        return result.y;
    });
    ```

=== "C++"
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

        return 0;
    }
    ```

---

## Gene Expression Time Course

Biological application:

=== "Python"
    ```python
    import numpy as np
    import fastlowess as fl

    # Gene expression over 24 hours
    hours = np.arange(0, 24.5, 0.5)
    expression = 100 * (1 + 0.5 * np.sin(hours * np.pi / 12)) + np.random.normal(0, 10, len(hours))

    model = fl.Lowess(
        fraction=0.3,
        iterations=3,
        confidence_intervals=0.95,
        return_diagnostics=True
    )
    result = model.fit(hours, expression)

    print(f"R²: {result.diagnostics.r_squared:.3f}")
    ```
=== "WebAssembly"
    ```javascript
    const { Lowess } = require('fastlowess-wasm');

    const n = 24;
    const hours = Float64Array.from({ length: n }, (_, i) => i);
    const expression = Float64Array.from(hours, h => 5 + 3 * Math.sin(h * Math.PI / 12) + (h % 3) * 0.2);
    const model = new Lowess({ fraction: 0.3, iterations: 3, return_diagnostics: true });
    const result = model.fit(hours, expression);

    console.log("R²:", result.diagnostics.r_squared);
    ```

=== "C++"
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
            expression[i] = 100.0 * (1.0 + 0.5 * std::sin(hours[i] * M_PI / 12.0));
        }

        fastlowess::Lowess gene_model({
            .fraction = 0.3,
            .iterations = 3,
            .return_diagnostics = true
        });
        auto result = gene_model.fit(hours, expression).value();

        std::cout << "R²: " << result.diagnostics().r_squared() << std::endl;

        return 0;
    }
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

- [Real-Time Processing](real-time.md) — For streaming time series
- [Cross-Validation](../user-guide/cross-validation.md) — Optimal fraction selection
- [Boundary Handling](../user-guide/boundary.md) — Edge bias in trend extraction
- [Parameters](../user-guide/parameters.md) — Full parameter reference
