<!-- markdownlint-disable MD024 -->
# Intervals

Confidence and prediction intervals for uncertainty quantification.

## Overview

![Intervals](../assets/diagrams/confidence_vs_prediction_intervals.svg)

| Type           | Represents                 | Width  | Use                       |
|----------------|----------------------------|--------|---------------------------|
| **Confidence** | Uncertainty in mean curve  | Narrow | Where is the true trend?  |
| **Prediction** | Uncertainty for new points | Wide   | Where will new data fall? |

---

## Confidence Intervals

Estimate uncertainty in the smoothed curve itself.

=== "R"
    ```r
    result <- fastlowess(x, y, fraction = 0.5, confidence_intervals = 0.95)

    # Plot with bands
    plot(x, y, pch = 16, col = "gray")
    lines(result$x, result$y, col = "blue", lwd = 2)
    lines(result$x, result$confidence_lower, col = "blue", lty = 2)
    lines(result$x, result$confidence_upper, col = "blue", lty = 2)
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, fraction=0.5, confidence_intervals=0.95)

    print("Smoothed:", result["y"])
    print("CI Lower:", result["confidence_lower"])
    print("CI Upper:", result["confidence_upper"])
    ```

=== "Rust"
    ```rust
    use fastLowess::prelude::*;

    let model = Lowess::new()
        .fraction(0.5)
        .confidence_intervals(0.95)  // 95% CI
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    
    // Access intervals
    if let (Some(lower), Some(upper)) = (&result.confidence_lower, &result.confidence_upper) {
        for i in 0..result.y.len() {
            println!("x={:.2}: y={:.2} [{:.2}, {:.2}]", 
                result.x[i], result.y[i], lower[i], upper[i]);
        }
    }
    ```

=== "Julia"
    ```julia
    using fastlowess

    result = smooth(x, y, fraction=0.5, confidence_intervals=0.95)

    for i in 1:length(result.y)
        println("x=$(result.x[i]): y=$(result.y[i]) [$(result.confidence_lower[i]), $(result.confidence_upper[i])]")
    end
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { fraction: 0.5, confidenceIntervals: 0.95 });

    result.y.forEach((y, i) => {
        console.log(`x=${result.x[i]}: y=${y} [${result.confidenceLower[i]}, ${result.confidenceUpper[i]}]`);
    });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { fraction: 0.5, confidenceIntervals: 0.95 });

    result.y.forEach((y, i) => {
        console.log(`x=${result.x[i]}: y=${y} [${result.confidenceLower[i]}, ${result.confidenceUpper[i]}]`);
    });
    ```

=== "C++"
    ```cpp
    #include "fastlowess.hpp"

    auto result = fastlowess::smooth(x, y, {
        .fraction = 0.5,
        .confidence_intervals = 0.95
    });

    auto ci_lower = result.confidence_lower();
    auto ci_upper = result.confidence_upper();
    ```

---

## Prediction Intervals

Estimate where new observations might fall.

=== "R"
    ```r
    result <- fastlowess(x, y, fraction = 0.5, prediction_intervals = 0.95)

    # Wider than confidence intervals
    polygon(
        c(result$x, rev(result$x)),
        c(result$prediction_lower, rev(result$prediction_upper)),
        col = rgb(1, 0, 0, 0.2), border = NA
    )
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, fraction=0.5, prediction_intervals=0.95)

    print("PI Lower:", result["prediction_lower"])
    print("PI Upper:", result["prediction_upper"])
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .fraction(0.5)
        .prediction_intervals(0.95)  // 95% PI
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    
    if let (Some(lower), Some(upper)) = (&result.prediction_lower, &result.prediction_upper) {
        println!("Prediction bounds: [{:.2}, {:.2}]", lower[0], upper[0]);
    }
    ```

=== "Julia"
    ```julia
    result = smooth(x, y, fraction=0.5, prediction_intervals=0.95)

    println("Prediction bounds: [$(result.prediction_lower[1]), $(result.prediction_upper[1])]")
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { fraction: 0.5, predictionIntervals: 0.95 });
    console.log(`Prediction bounds: [${result.predictionLower[0]}, ${result.predictionUpper[0]}]`);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { fraction: 0.5, predictionIntervals: 0.95 });
    console.log(`Prediction bounds: [${result.predictionLower[0]}, ${result.predictionUpper[0]}]`);
    ```

=== "C++"
    ```cpp
    auto result = fastlowess::smooth(x, y, { .fraction = 0.5, .prediction_intervals = 0.95 });
    ```

---

## Both Intervals

Request both types simultaneously:

=== "R"
    ```r
    result <- fastlowess(
        x, y,
        fraction = 0.5,
        confidence_intervals = 0.95,
        prediction_intervals = 0.95
    )
    ```

=== "Python"
    ```python
    result = fl.smooth(
        x, y,
        fraction=0.5,
        confidence_intervals=0.95,
        prediction_intervals=0.95
    )
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .fraction(0.5)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = smooth(
        x, y,
        fraction=0.5,
        confidence_intervals=0.95,
        prediction_intervals=0.95
    )
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, {
        fraction: 0.5,
        confidenceIntervals: 0.95,
        predictionIntervals: 0.95
    });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, {
        fraction: 0.5,
        confidenceIntervals: 0.95,
        predictionIntervals: 0.95
    });
    ```

=== "C++"
    ```cpp
    auto result = fastlowess::smooth(x, y, { .fraction = 0.5, .confidence_intervals = 0.95, .prediction_intervals = 0.95 });
    ```

---

## Confidence Levels

Common levels and their z-values:

| Level | z-value | Interpretation                      |
|-------|---------|-------------------------------------|
| 0.90  | 1.645   | 90% of intervals contain true value |
| 0.95  | 1.960   | 95% of intervals contain true value |
| 0.99  | 2.576   | 99% of intervals contain true value |

=== "R"
    ```r
    # 99% confidence interval (wider)
    result <- fastlowess(x, y, confidence_intervals = 0.99)
    ```

=== "Python"
    ```python
    # 90% confidence interval (narrower)
    result = fl.smooth(x, y, confidence_intervals=0.90)
    ```

=== "Rust"
    ```rust
    // 99% confidence interval
    let model = Lowess::new()
        .confidence_intervals(0.99)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    # 99% confidence interval
    result = smooth(x, y, confidence_intervals=0.99)
    ```

=== "Node.js"
    ```javascript
    // 99% confidence interval
    const result = smooth(x, y, { confidenceIntervals: 0.99 });
    ```

=== "WebAssembly"
    ```javascript
    // 99% confidence interval
    const result = smooth(x, y, { confidenceIntervals: 0.99 });
    ```

=== "C++"
    ```cpp
    auto result = fastlowess::smooth(x, y, { .confidence_intervals = 0.99 });
    ```

---

## Standard Errors

Access standard errors directly (available when intervals are computed):

=== "R"
    ```r
    result <- fastlowess(x, y, confidence_intervals = 0.95)
    print(result$std_err)
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, confidence_intervals=0.95)
    print("Standard errors:", result["std_err"])
    ```

=== "Rust"
    ```rust
    let result = model.fit(&x, &y)?;

    if let Some(std_err) = &result.std_err {
        for (i, &se) in std_err.iter().enumerate() {
            println!("Point {}: SE = {:.4}", i, se);
        }
    }
    ```

=== "Julia"
    ```julia
    result = smooth(x, y, confidence_intervals=0.95)

    for (i, se) in enumerate(result.std_err)
        println("Point $i: SE = $se")
    end
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { confidenceIntervals: 0.95 });

    result.standardErrors.forEach((se, i) => {
        console.log(`Point ${i}: SE = ${se.toFixed(4)}`);
    });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { confidenceIntervals: 0.95 });

    result.standardErrors.forEach((se, i) => {
        console.log(`Point ${i}: SE = ${se.toFixed(4)}`);
    });
    ```

=== "C++"
    ```cpp
    auto result = fastlowess::smooth(x, y, { .confidence_intervals = 0.95 });
    ```

---

## Availability

!!! warning "Batch Mode Only"
    Confidence and prediction intervals are only available in **Batch** mode. Streaming and Online modes do not support intervals.

| Feature              | Batch | Streaming | Online |
|----------------------|-------|-----------|--------|
| Confidence intervals | ✓     | ✗         | ✗      |
| Prediction intervals | ✓     | ✗         | ✗      |
| Standard errors      | ✓     | ✗         | ✗      |
