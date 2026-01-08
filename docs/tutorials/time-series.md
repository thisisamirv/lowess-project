# Time Series Analysis

LOWESS for trend extraction and temporal smoothing.

## Overview

Time series data often contains noise, seasonality, and trends. LOWESS provides flexible trend extraction without parametric assumptions.

---

## Basic Trend Extraction

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

---

## Detrending

Remove trend to analyze residual patterns:

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

=== "R"
    ```r
    result <- fastlowess(t, y, fraction = 0.3, iterations = 3, return_residuals = TRUE)

    trend <- result$y
    detrended <- result$residuals

    par(mfrow = c(1, 2))
    plot(t, trend, type = "l", main = "Trend")
    plot(t, detrended, type = "l", main = "Detrended")
    ```

---

## Forecasting with Prediction Intervals

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

---

## Handling Missing Data

LOWESS naturally handles irregular time sampling:

=== "Python"
    ```python
    # Irregular time points (gaps in data)
    t_irregular = np.sort(np.random.uniform(0, 100, 200))
    y_irregular = 10 + t_irregular * 0.3 + np.random.normal(0, 2, 200)

    # LOWESS handles this seamlessly
    result = fl.smooth(t_irregular, y_irregular, fraction=0.2)
    ```

=== "R"
    ```r
    t_irregular <- sort(runif(200, 0, 100))
    y_irregular <- 10 + 0.3 * t_irregular + rnorm(200, sd = 2)

    result <- fastlowess(t_irregular, y_irregular, fraction = 0.2)
    ```

---

## Multi-Scale Analysis

Use different fractions to extract features at different scales:

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
