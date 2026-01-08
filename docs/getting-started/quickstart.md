# Quick Start

Get up and running with LOWESS in minutes.

## Basic Smoothing

=== "Rust"

    ```rust
    use lowess::prelude::*;

    fn main() -> Result<(), LowessError> {
        // Sample data
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7];

        // Build and fit the model
        let model = Lowess::new()
            .fraction(0.5)      // Use 50% of data for each fit
            .iterations(3)      // 3 robustness iterations
            .adapter(Batch)
            .build()?;

        let result = model.fit(&x, &y)?;
        
        println!("{}", result);
        Ok(())
    }
    ```

=== "Python"

    ```python
    import fastlowess as fl
    import numpy as np

    # Sample data
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y = np.array([2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7])

    # Smooth the data
    result = fl.smooth(x, y, fraction=0.5, iterations=3)

    print("Smoothed values:", result["y"])
    ```

=== "R"

    ```r
    library(rfastlowess)

    # Sample data
    x <- c(1, 2, 3, 4, 5, 6, 7, 8)
    y <- c(2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7)

    # Smooth the data
    result <- fastlowess(x, y, fraction = 0.5, iterations = 3)

    print(result$y)
    ```

**Output:**

```text
Smoothed values: [2.02, 4.00, 6.00, 8.10, 10.04, 12.03, 13.90, 15.78]
```

---

## With Confidence Intervals

=== "Rust"

    ```rust
    use lowess::prelude::*;

    let model = Lowess::new()
        .fraction(0.5)
        .iterations(3)
        .confidence_intervals(0.95)  // 95% CI
        .prediction_intervals(0.95)  // 95% PI
        .return_diagnostics()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    
    // Access intervals
    if let Some(ci_lower) = &result.confidence_lower {
        println!("CI Lower: {:?}", ci_lower);
    }
    ```

=== "Python"

    ```python
    result = fl.smooth(
        x, y,
        fraction=0.5,
        iterations=3,
        confidence_intervals=0.95,
        prediction_intervals=0.95,
        return_diagnostics=True
    )

    print("Smoothed:", result["y"])
    print("CI Lower:", result["confidence_lower"])
    print("CI Upper:", result["confidence_upper"])
    print("R²:", result["diagnostics"]["r_squared"])
    ```

=== "R"

    ```r
    result <- fastlowess(
        x, y,
        fraction = 0.5,
        iterations = 3,
        confidence_intervals = 0.95,
        prediction_intervals = 0.95,
        return_diagnostics = TRUE
    )

    print(result$confidence_lower)
    print(result$confidence_upper)
    print(result$diagnostics$r_squared)
    ```

---

## Handling Outliers

LOWESS can robustly handle outliers through iterative reweighting:

=== "Rust"

    ```rust
    // Data with an outlier at position 3
    let y_with_outlier = vec![2.0, 4.0, 6.0, 50.0, 10.0, 12.0];  // 50.0 is outlier

    let model = Lowess::new()
        .fraction(0.5)
        .iterations(5)                    // More iterations for outliers
        .robustness_method(Bisquare)      // Default, smooth downweighting
        .return_robustness_weights()      // See which points were downweighted
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y_with_outlier)?;
    
    // Outliers will have low robustness weights
    if let Some(weights) = &result.robustness_weights {
        for (i, w) in weights.iter().enumerate() {
            if *w < 0.5 {
                println!("Point {} is likely an outlier (weight: {:.3})", i, w);
            }
        }
    }
    ```

=== "Python"

    ```python
    y_with_outlier = np.array([2.0, 4.0, 6.0, 50.0, 10.0, 12.0])

    result = fl.smooth(
        x, y_with_outlier,
        fraction=0.5,
        iterations=5,
        robustness_method="bisquare",
        return_robustness_weights=True
    )

    # Check which points were downweighted
    for i, w in enumerate(result["robustness_weights"]):
        if w < 0.5:
            print(f"Point {i} is likely an outlier (weight: {w:.3f})")
    ```

=== "R"

    ```r
    y_with_outlier <- c(2, 4, 6, 50, 10, 12)

    result <- fastlowess(
        x, y_with_outlier,
        fraction = 0.5,
        iterations = 5,
        robustness_method = "bisquare",
        return_robustness_weights = TRUE
    )

    # Check downweighted points
    weights <- result$robustness_weights
    for (i in seq_along(weights)) {
        if (weights[i] < 0.5) {
            cat(sprintf("Point %d is likely an outlier (weight: %.3f)\n", i, weights[i]))
        }
    }
    ```

---

## Next Steps

- [Concepts](concepts.md) — Understand how LOWESS works
- [Parameters](../user-guide/parameters.md) — All configuration options
- [Execution Modes](../user-guide/adapters.md) — Batch, Streaming, Online
