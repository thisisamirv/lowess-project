<!-- markdownlint-disable MD024 MD046 MD033 -->
# Time Series Analysis

LOWESS for trend extraction and temporal smoothing.

## Overview

Time series data often contains noise, seasonality, and trends. LOWESS provides flexible trend extraction without parametric assumptions.

---

## Basic Trend Extraction

=== "R"
    ```r
    library(rfastlowess)

    set.seed(42)
    t <- seq(0, 100, length.out = 500)
    trend <- 10 + 0.5 * t + 3 * sin(t / 10)
    noise <- rnorm(500, sd = 3)
    y <- trend + noise

    result <- fastlowess(t, y, fraction = 0.1, iterations = 3)

    plot(t, y, col = "gray", pch = ".",
         xlab = "Time", ylab = "Value", main = "Trend Extraction")
    lines(result$x, result$y, col = "blue", lwd = 2)
    ```

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
    result = fl.smooth(t, y, fraction=0.1, iterations=3)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(t, y, "gray", alpha=0.5, label="Observed")
    plt.plot(t, result["y"], "b-", lwd=2, label="Trend (LOWESS)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Trend Extraction")
    plt.show()
    ```

=== "Rust"
    ```rust
    use fastLowess::prelude::*;
    use ndarray::Array1;

    let t: Array1<f64> = Array1::linspace(0.0, 100.0, 500);
    let y: Array1<f64> = /* your data */;

    let model = Lowess::new()
        .fraction(0.1)
        .iterations(3)
        .adapter(Batch)
        .build()?;

    let result = model.fit(&t, &y)?;
    // result.y contains the trend
    ```

=== "Julia"
    ```julia
    using fastLowess

    t = collect(range(0, 100, length=500))
    trend_true = 10.0 .+ 0.5 .* t .+ 3.0 .* sin.(t ./ 10.0)
    y = trend_true .+ randn(500) .* 3.0

    # Extract trend
    result = smooth(t, y, fraction=0.1, iterations=3)

    println("Extracted trend points: ", length(result.y))
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastlowess');

    // t and y are your time series arrays (Float64Array)
    const result = fl.smooth(t, y, { 
        fraction: 0.1, 
        iterations: 3 
    });

    console.log("Extracted trend:", result.y);
    ```

=== "WebAssembly"
    ```javascript
    import { smooth } from 'fastlowess-wasm';

    const result = smooth(t, y, { 
        fraction: 0.1, 
        iterations: 3 
    });

    // Trend values in result.y
    ```

=== "C++"
    ```cpp
    #include "fastlowess.hpp"

    std::vector<double> t = /* time points */;
    std::vector<double> y = /* values */;

    auto result = fastlowess::smooth(t, y, {
        .fraction = 0.1,
        .iterations = 3
    });

    // Trend in result.y_vector()
    ```

---

## Detrending

Remove trend to analyze residual patterns:

=== "R"
    ```r
    result <- fastlowess(t, y, fraction = 0.3, iterations = 3, return_residuals = TRUE)

    trend <- result$y
    detrended <- result$residuals

    par(mfrow = c(1, 2))
    plot(t, trend, type = "l", main = "Trend")
    plot(t, detrended, type = "l", main = "Detrended")
    ```

=== "Python"
    ```python
    # Smooth to get trend
    result = fl.smooth(t, y, fraction=0.3, iterations=3, return_residuals=True)

    trend = result["y"]
    detrended = result["residuals"]

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

=== "Rust"
    ```rust
    let model = Lowess::new()
        .fraction(0.3)
        .iterations(3)
        .return_residuals()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&t, &y)?;
    let trend = &result.y;
    let detrended = result.residuals.as_ref().unwrap();
    ```

=== "Julia"
    ```julia
    # Smooth to get trend and residuals
    result = smooth(t, y, fraction=0.3, iterations=3, return_residuals=true)

    trend = result.y
    detrended = result.residuals

    println("Detrended variance: ", var(detrended))
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastlowess');

    const result = fl.smooth(t, y, { 
        fraction: 0.3, 
        iterations: 3, 
        returnResiduals: true 
    });

    const trend = result.y;
    const detrended = result.residuals;
    ```

=== "WebAssembly"
    ```javascript
    import { smooth } from 'fastlowess-wasm';

    const result = smooth(t, y, { 
        fraction: 0.3, 
        iterations: 3, 
        returnResiduals: true 
    });

    // Access result.y (trend) and result.residuals (detrended)
    ```

=== "C++"
    ```cpp
    #include "fastlowess.hpp"

    auto result = fastlowess::smooth(t, y, {
        .fraction = 0.3,
        .iterations = 3,
        .return_residuals = true
    });

    auto trend = result.y_vector();
    auto detrended = result.residuals();
    ```

---

## Forecasting with Prediction Intervals

=== "R"
    ```r
    result <- fastlowess(
        t, y,
        fraction = 0.2,
        iterations = 3,
        confidence_intervals = 0.95,
        prediction_intervals = 0.95
    )

    plot(t, y, col = "gray", pch = 16)
    lines(result$x, result$y, col = "blue", lwd = 2)
    lines(result$x, result$prediction_lower, col = "blue", lty = 2)
    lines(result$x, result$prediction_upper, col = "blue", lty = 2)
    ```

=== "Python"
    ```python
    result = fl.smooth(
        t, y,
        fraction=0.2,
        iterations=3,
        confidence_intervals=0.95,
        prediction_intervals=0.95
    )

    # Plot with uncertainty bands
    plt.figure(figsize=(12, 5))
    plt.plot(t, y, "gray", alpha=0.3)
    plt.plot(t, result["y"], "b-", lwd=2, label="Trend")
    plt.fill_between(
        t,
        result["prediction_lower"],
        result["prediction_upper"],
        alpha=0.2, color="blue", label="95% Prediction"
    )
    plt.legend()
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .fraction(0.2)
        .iterations(3)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .adapter(Batch)
        .build()?;

    let result = model.fit(&t, &y)?;
    // Access result.prediction_lower and result.prediction_upper
    ```

=== "Julia"
    ```julia
    result = smooth(
        t, y,
        fraction=0.2,
        iterations=3,
        confidence_intervals=0.95,
        prediction_intervals=0.95
    )

    # Intervals are available in result.prediction_lower/upper
    println("First point 95% PI: [$(result.prediction_lower[1]), $(result.prediction_upper[1])]")
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastlowess');

    const result = fl.smooth(t, y, {
        fraction: 0.2,
        iterations: 3,
        predictionIntervals: 0.95
    });

    console.log(`95% PI: [${result.predictionLower[0]}, ${result.predictionUpper[0]}]`);
    ```

=== "WebAssembly"
    ```javascript
    import { smooth } from 'fastlowess-wasm';

    const result = smooth(t, y, {
        fraction: 0.2,
        iterations: 3,
        predictionIntervals: 0.95
    });

    // Access result.predictionLower and result.predictionUpper
    ```

=== "C++"
    ```cpp
    #include "fastlowess.hpp"

    auto result = fastlowess::smooth(t, y, {
        .fraction = 0.2,
        .iterations = 3,
        .confidence_intervals = 0.95,
        .prediction_intervals = 0.95
    });

    // Access result.predictionLower and result.predictionUpper
    ```

---

## Handling Missing Data

LOWESS naturally handles irregular time sampling:

=== "R"
    ```r
    t_irregular <- sort(runif(200, 0, 100))
    y_irregular <- 10 + 0.3 * t_irregular + rnorm(200, sd = 2)

    result <- fastlowess(t_irregular, y_irregular, fraction = 0.2)
    ```

=== "Python"
    ```python
    # Irregular time points (gaps in data)
    t_irregular = np.sort(np.random.uniform(0, 100, 200))
    y_irregular = 10 + t_irregular * 0.3 + np.random.normal(0, 2, 200)

    # LOWESS handles this seamlessly
    result = fl.smooth(t_irregular, y_irregular, fraction=0.2)
    ```

=== "Rust"
    ```rust
    // Irregular sampling - no special handling needed
    let t_irregular: Array1<f64> = /*sorted irregular times */;
    let y_irregular: Array1<f64> = /* corresponding values*/;

    let model = Lowess::new()
        .fraction(0.2)
        .adapter(Batch)
        .build()?;

    let result = model.fit(&t_irregular, &y_irregular)?;
    ```

=== "Julia"
    ```julia
    # Irregular time points (gaps in data)
    t_irregular = sort(rand(200) .*100.0)
    y_irregular = 10.0 .+ t_irregular .* 0.3 .+ randn(200) .* 2.0

    # LOWESS handles this seamlessly
    result = smooth(t_irregular, y_irregular, fraction=0.2)
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastlowess');

    // No special handling needed for irregular spacing
    const result = fl.smooth(tIrregular, yIrregular, { fraction: 0.2 });
    ```

=== "WebAssembly"
    ```javascript
    import { smooth } from 'fastlowess-wasm';

    const result = smooth(tIrregular, yIrregular, { fraction: 0.2 });
    ```

=== "C++"
    ```cpp
    #include "fastlowess.hpp"

    auto result = fastlowess::smooth(tIrregular, yIrregular, {
        .fraction = 0.2,
    });
    ```

---

## Multi-Scale Analysis

Use different fractions to extract features at different scales:

=== "R"
    ```r
    fractions <- c(0.05, 0.2, 0.5)

    plot(t, y, col = "gray", pch = ".", main = "Multi-Scale LOWESS")
    colors <- c("red", "blue", "green")
    for (i in seq_along(fractions)) {
        result <- fastlowess(t, y, fraction = fractions[i])
        lines(result$x, result$y, col = colors[i], lwd = 2)
    }
    legend("topleft", legend = paste("f =", fractions), col = colors, lwd = 2)
    ```

=== "Python"
    ```python
    # Multiple smoothing scales
    fractions = [0.05, 0.2, 0.5]

    plt.figure(figsize=(12, 5))
    plt.plot(t, y, "gray", alpha=0.3, label="Data")
    
    for f in fractions:
        result = fl.smooth(t, y, fraction=f)
        plt.plot(t, result["y"], label=f"fraction={f}")
    
    plt.legend()
    plt.title("Multi-Scale LOWESS")
    ```

=== "Rust"
    ```rust
    let fractions = [0.05, 0.2, 0.5];

    for f in fractions {
        let model = Lowess::new()
            .fraction(f)
            .adapter(Batch)
            .build()?;
        let result = model.fit(&t, &y)?;
        // Store or plot result.y for each scale
    }
    ```

=== "Julia"
    ```julia
    fractions = [0.05, 0.2, 0.5]

    results = [smooth(t, y, fraction=f) for f in fractions]
    # results[i].y contains smoothed values for each fraction
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastlowess');

    const scales = [0.05, 0.2, 0.5];
    const trends = scales.map(f => {
        return fl.smooth(t, y, { fraction: f }).y;
    });
    ```

=== "WebAssembly"
    ```javascript
    import { smooth } from 'fastlowess-wasm';

    const trends = [0.05, 0.2, 0.5].map(f => {
        const result = smooth(t, y, { fraction: f });
        return result.y;
    });
    ```

=== "C++"
    ```cpp
    #include "fastlowess.hpp"

    std::vector<double> scales = {0.05, 0.2, 0.5};
    std::vector<std::vector<double>> trends;
    for (auto f : scales) {
        auto result = fastlowess::smooth(t, y, { .fraction = f });
        trends.push_back(result.y_vector());
    }
    ```

---

## Gene Expression Time Course

Biological application:

=== "R"
    ```r
    # Gene expression over 24 hours
    hours <- seq(0, 24, by = 0.5)

    # Circadian pattern with measurement noise
    expression <- 100 * (1 + 0.5 * sin(hours * pi / 12)) + rnorm(49, sd = 10)

    result <- fastlowess(
        hours, expression,
        fraction = 0.3,
        iterations = 3,
        confidence_intervals = 0.95,
        return_diagnostics = TRUE
    )

    # Plot
    plot(hours, expression, pch = 16, col = "gray",
         xlab = "Time (hours)", ylab = "Expression Level",
         main = "Gene Expression Time Course")
    lines(result$x, result$y, col = "red", lwd = 2)
    lines(result$x, result$confidence_lower, col = "red", lty = 2)
    lines(result$x, result$confidence_upper, col = "red", lty = 2)

    cat("R²:", result$diagnostics$r_squared, "\n")
    ```

=== "Python"
    ```python
    import numpy as np
    import fastlowess as fl

    # Gene expression over 24 hours
    hours = np.arange(0, 24.5, 0.5)
    expression = 100 * (1 + 0.5 * np.sin(hours * np.pi / 12)) + np.random.normal(0, 10, len(hours))

    result = fl.smooth(
        hours, expression,
        fraction=0.3,
        iterations=3,
        confidence_intervals=0.95,
        return_diagnostics=True
    )

    print(f"R²: {result['diagnostics']['r_squared']:.3f}")
    ```

=== "Rust"
    ```rust
    let hours: Array1<f64> = Array1::range(0.0, 24.5, 0.5);
    let expression: Array1<f64> = /*circadian data*/;

    let model = Lowess::new()
        .fraction(0.3)
        .iterations(3)
        .confidence_intervals(0.95)
        .return_diagnostics()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&hours, &expression)?;
    if let Some(diag) = &result.diagnostics {
        println!("R²: {:.3}", diag.r_squared);
    }
    ```

=== "Julia"
    ```julia
    using fastLowess

    hours = collect(range(0, 24, step=0.5))
    expression = 100 .*(1.0 .+ 0.5 .* sin.(hours .*pi ./ 12.0)) .+ randn(length(hours)) .* 10.0

    result = smooth(
        hours, expression,
        fraction=0.3,
        iterations=3,
        confidence_intervals=0.95,
        return_diagnostics=true
    )

    println("R²: ", result.diagnostics.r_squared)
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastlowess');

    const result = fl.smooth(hours, expression, {
        fraction: 0.3,
        iterations: 3,
        returnDiagnostics: true
    });

    console.log(`R²: ${result.diagnostics.rSquared.toFixed(3)}`);
    ```

=== "WebAssembly"
    ```javascript
    import { smooth } from 'fastlowess-wasm';

    const result = smooth(hours, expression, {
        fraction: 0.3,
        iterations: 3,
        returnDiagnostics: true
    });

    console.log("R²:", result.diagnostics.rSquared);
    ```

=== "C++"
    ```cpp
    #include "fastlowess.hpp"

    auto result = fastlowess::smooth(hours, expression, {
        .fraction = 0.3,
        .iterations = 3,
        .return_diagnostics = true
    });

    std::cout << "R²: " << result.diagnostics.r_squared << std::endl;
    ```

---

## Choosing Fraction for Time Series

| Data Type             | Recommended Fraction | Rationale                    |
|-----------------------|----------------------|------------------------------|
| Daily data (years)    | 0.3–0.5              | Capture annual trends        |
| Hourly data (days)    | 0.1–0.2              | Capture daily patterns       |
| Sensor data (minutes) | 0.05–0.1             | Preserve short-term features |
| Noisy data            | Higher               | Reduce noise impact          |
| Clean data            | Lower                | Preserve detail              |

---

## See Also

- [Real-Time Processing](real-time.md) — For streaming time series
- [Cross-Validation](../user-guide/cross-validation.md) — Optimal fraction selection
