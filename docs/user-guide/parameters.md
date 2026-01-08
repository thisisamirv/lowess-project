# Parameters

Complete reference for all LOWESS configuration options.

## Quick Reference

| Parameter                     | Default        | Range/Options | Description             | Adapter          |
|-------------------------------|----------------|---------------|-------------------------|------------------|
| **fraction**                  | 0.67           | (0, 1]        | Smoothing span          | All              |
| **iterations**                | 3              | [0, 1000]     | Robustness iterations   | All              |
| **delta**                     | 1% x-range     | [0, ∞)        | Interpolation threshold | All              |
| **weight_function**           | `Tricube`      | 7 options     | Distance kernel         | All              |
| **robustness_method**         | `Bisquare`     | 3 options     | Outlier weighting       | All              |
| **zero_weight_fallback**      | `UseLocalMean` | 3 options     | Zero-weight behavior    | All              |
| **boundary_policy**           | `Extend`       | 4 options     | Edge handling           | All              |
| **scaling_method**            | `MAD`          | 2 options     | Scale estimation        | All              |
| **auto_converge**             | None           | tolerance     | Early stopping          | All              |
| **return_residuals**          | false          | bool          | Include residuals       | All              |
| **return_robustness_weights** | false          | bool          | Include weights         | All              |
| **return_diagnostics**        | false          | bool          | Include metrics         | Batch, Streaming |
| **confidence_intervals**      | None           | (0, 1)        | CI level                | Batch            |
| **prediction_intervals**      | None           | (0, 1)        | PI level                | Batch            |
| **cross_validate**            | None           | method        | Auto-select fraction    | Batch            |
| **chunk_size**                | 5000           | [10, ∞)       | Points per chunk        | Streaming        |
| **overlap**                   | 500            | [0, chunk)    | Overlap between chunks  | Streaming        |
| **merge_strategy**            | `Average`      | 4 options     | Merge overlaps          | Streaming        |
| **window_capacity**           | 1000           | [3, ∞)        | Max window size         | Online           |
| **min_points**                | 2              | [2, window]   | Min before output       | Online           |
| **update_mode**               | `Incremental`  | 2 options     | Update strategy         | Online           |

---

## Parameter Options Summary

=== "Rust"

    | Parameter                | Available Options                                                                  |
    |--------------------------|------------------------------------------------------------------------------------|
    | **weight_function**      | `Tricube`, `Epanechnikov`, `Gaussian`, `Biweight`, `Cosine`, `Triangle`, `Uniform` |
    | **robustness_method**    | `Bisquare`, `Huber`, `Talwar`                                                      |
    | **zero_weight_fallback** | `UseLocalMean`, `ReturnOriginal`, `ReturnNone`                                     |
    | **boundary_policy**      | `Extend`, `Reflect`, `Zero`, `NoBoundary`                                          |
    | **scaling_method**       | `MAD`, `MAR`                                                                       |
    | **merge_strategy**       | `Average`, `Left`, `Right`, `Weighted`                                             |
    | **update_mode**          | `Incremental`, `Full`                                                              |

=== "Python"

    | Parameter                | Available Options                                                                            |
    |--------------------------|----------------------------------------------------------------------------------------------|
    | **weight_function**      | `"tricube"`, `"epanechnikov"`, `"gaussian"`, `"biweight"`, `"cosine"`, `"triangle"`, `"uniform"` |
    | **robustness_method**    | `"bisquare"`, `"huber"`, `"talwar"`                                                          |
    | **zero_weight_fallback** | `"use_local_mean"`, `"return_original"`, `"return_none"`                                     |
    | **boundary_policy**      | `"extend"`, `"reflect"`, `"zero"`, `"no_boundary"`                                           |
    | **scaling_method**       | `"mad"`, `"mar"`                                                                             |
    | **merge_strategy**       | `"average"`, `"left"`, `"right"`, `"weighted"`                                               |
    | **update_mode**          | `"incremental"`, `"full"`                                                                    |

=== "R"

    | Parameter                | Available Options                                                                            |
    |--------------------------|----------------------------------------------------------------------------------------------|
    | **weight_function**      | `"tricube"`, `"epanechnikov"`, `"gaussian"`, `"biweight"`, `"cosine"`, `"triangle"`, `"uniform"` |
    | **robustness_method**    | `"bisquare"`, `"huber"`, `"talwar"`                                                          |
    | **zero_weight_fallback** | `"use_local_mean"`, `"return_original"`, `"return_none"`                                     |
    | **boundary_policy**      | `"extend"`, `"reflect"`, `"zero"`, `"no_boundary"`                                           |
    | **scaling_method**       | `"mad"`, `"mar"`                                                                             |
    | **merge_strategy**       | `"average"`, `"left"`, `"right"`, `"weighted"`                                               |
    | **update_mode**          | `"incremental"`, `"full"`                                                                    |

---

## Core Parameters

### fraction

The proportion of data used for each local fit. **Most important parameter.**

| Value   | Effect          | Use Case                 |
|---------|-----------------|--------------------------|
| 0.1–0.3 | Fine detail     | Rapidly changing signals |
| 0.3–0.5 | Balanced        | General purpose          |
| 0.5–0.7 | Heavy smoothing | Noisy data               |
| 0.7–1.0 | Very smooth     | Trend extraction         |

=== "Rust"
    ```rust
    let model = Lowess::new()
        .fraction(0.3)
        .adapter(Batch)
        .build()?;
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, fraction=0.3)
    ```

=== "R"
    ```r
    result <- fastlowess(x, y, fraction = 0.3)
    ```

---

### iterations

Number of robustness iterations for outlier resistance.

| Value | Effect        | Performance       |
|-------|---------------|-------------------|
| 0     | No robustness | Fastest           |
| 1–3   | Moderate      | Recommended       |
| 4–6   | Strong        | Contaminated data |
| 7+    | Very strong   | Heavy outliers    |

=== "Rust"
    ```rust
    let model = Lowess::new()
        .iterations(5)
        .adapter(Batch)
        .build()?;
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, iterations=5)
    ```

=== "R"
    ```r
    result <- fastlowess(x, y, iterations = 5)
    ```

---

### delta

Interpolation optimization threshold. Points within `delta` distance reuse the previous fit.

- **Default**: 1% of x-range (Batch), 0.0 (Streaming/Online)
- **Effect**: Higher values = faster but less accurate

=== "Rust"
    ```rust
    let model = Lowess::new()
        .delta(0.05)
        .adapter(Batch)
        .build()?;
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, delta=0.05)
    ```

=== "R"
    ```r
    result <- fastlowess(x, y, delta = 0.05)
    ```

---

### weight_function

Distance weighting kernel for local fits.

| Kernel         | Efficiency | Smoothness    |
|----------------|:----------:|:-------------:|
| `Tricube`      | 0.998      | Very smooth   |
| `Epanechnikov` | 1.000      | Smooth        |
| `Gaussian`     | 0.961      | Infinite      |
| `Biweight`     | 0.995      | Very smooth   |
| `Cosine`       | 0.999      | Smooth        |
| `Triangle`     | 0.989      | Moderate      |
| `Uniform`      | 0.943      | None          |

See [Weight Functions](kernels.md) for detailed comparison.

=== "Rust"
    ```rust
    let model = Lowess::new()
        .weight_function(Epanechnikov)
        .adapter(Batch)
        .build()?;
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, weight_function="epanechnikov")
    ```

=== "R"
    ```r
    result <- fastlowess(x, y, weight_function = "epanechnikov")
    ```

---

### robustness_method

Method for downweighting outliers during iterative refinement.

| Method     | Behavior                | Use Case              |
|------------|-------------------------|-----------------------|
| `Bisquare` | Smooth downweighting    | General-purpose       |
| `Huber`    | Linear beyond threshold | Moderate outliers     |
| `Talwar`   | Hard threshold (0 or 1) | Extreme contamination |

See [Robustness](robustness.md) for detailed comparison.

=== "Rust"
    ```rust
    let model = Lowess::new()
        .robustness_method(Talwar)
        .adapter(Batch)
        .build()?;
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, robustness_method="talwar")
    ```

=== "R"
    ```r
    result <- fastlowess(x, y, robustness_method = "talwar")
    ```

---

### boundary_policy

Edge handling strategy to reduce boundary bias.

![Boundary Policy](../assets/diagrams/boundary_handling.svg)

| Policy       | Behavior                   | Use Case                    |
|--------------|----------------------------|-----------------------------|
| `Extend`     | Pad with first/last values | Most cases (default)        |
| `Reflect`    | Mirror data at boundaries  | Periodic/symmetric data     |
| `Zero`       | Pad with zeros             | Data approaches zero        |
| `NoBoundary` | No padding                 | Original Cleveland behavior |

=== "Rust"
    ```rust
    let model = Lowess::new()
        .boundary_policy(Reflect)
        .adapter(Batch)
        .build()?;
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, boundary_policy="reflect")
    ```

=== "R"
    ```r
    result <- fastlowess(x, y, boundary_policy = "reflect")
    ```

---

### scaling_method

Method for estimating residual scale during robustness iterations.

| Method | Description               | Robustness          |
|--------|---------------------------|---------------------|
| `MAD`  | Median Absolute Deviation | Very robust         |
| `MAR`  | Mean Absolute Residual    | Less robust, faster |

=== "Rust"
    ```rust
    let model = Lowess::new()
        .scaling_method(MAD)
        .adapter(Batch)
        .build()?;
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, scaling_method="mad")
    ```

=== "R"
    ```r
    result <- fastlowess(x, y, scaling_method = "mad")
    ```

---

### zero_weight_fallback

Behavior when all neighborhood weights are zero.

| Option           | Behavior                                  |
|------------------|-------------------------------------------|
| `UseLocalMean`   | Use mean of neighborhood (default)        |
| `ReturnOriginal` | Return original y value                   |
| `ReturnNone`     | Return NaN                                |

=== "Rust"
    ```rust
    let model = Lowess::new()
        .zero_weight_fallback(UseLocalMean)
        .adapter(Batch)
        .build()?;
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, zero_weight_fallback="use_local_mean")
    ```

=== "R"
    ```r
    result <- fastlowess(x, y, zero_weight_fallback = "use_local_mean")
    ```

---

### auto_converge

Enable early stopping when robustness weights stabilize.

![Auto-Convergence](../assets/diagrams/auto_converge.svg)

=== "Rust"
    ```rust
    let model = Lowess::new()
        .iterations(20)           // Maximum
        .auto_converge(1e-6)      // Stop when change < 1e-6
        .adapter(Batch)
        .build()?;
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, iterations=20, auto_converge=1e-6)
    ```

=== "R"
    ```r
    result <- fastlowess(x, y, iterations = 20, auto_converge = 1e-6)
    ```

---

## Output Options

### return_residuals

Include residuals (`y - smoothed`) in the output.

=== "Rust"
    ```rust
    let model = Lowess::new()
        .return_residuals()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    if let Some(residuals) = result.residuals {
        println!("Residuals: {:?}", residuals);
    }
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, return_residuals=True)
    print(result["residuals"])
    ```

=== "R"
    ```r
    result <- fastlowess(x, y, return_residuals = TRUE)
    print(result$residuals)
    ```

---

### return_diagnostics

Include fit quality metrics (Batch and Streaming only).

| Metric         | Description                  |
|----------------|------------------------------|
| `rmse`         | Root Mean Square Error       |
| `mae`          | Mean Absolute Error          |
| `r_squared`    | R² coefficient               |
| `residual_sd`  | Residual standard deviation  |
| `effective_df` | Effective degrees of freedom |
| `aic`          | Akaike Information Criterion |
| `aicc`         | Corrected AIC                |

=== "Rust"
    ```rust
    let model = Lowess::new()
        .return_diagnostics()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    if let Some(diag) = result.diagnostics {
        println!("R²: {:.4}", diag.r_squared);
        println!("RMSE: {:.4}", diag.rmse);
    }
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, return_diagnostics=True)
    print(f"R²: {result['diagnostics']['r_squared']:.4f}")
    ```

=== "R"
    ```r
    result <- fastlowess(x, y, return_diagnostics = TRUE)
    cat(sprintf("R²: %.4f\n", result$diagnostics$r_squared))
    ```

---

### return_robustness_weights

Include final robustness weights (useful for outlier detection).

=== "Rust"
    ```rust
    let model = Lowess::new()
        .iterations(3)
        .return_robustness_weights()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    // Points with weight < 0.5 are likely outliers
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, iterations=3, return_robustness_weights=True)
    outliers = [i for i, w in enumerate(result["robustness_weights"]) if w < 0.5]
    ```

=== "R"
    ```r
    result <- fastlowess(x, y, iterations = 3, return_robustness_weights = TRUE)
    outliers <- which(result$robustness_weights < 0.5)
    ```

---

### confidence_intervals / prediction_intervals

Request uncertainty estimates (Batch only).

See [Intervals](intervals.md) for detailed usage.

=== "Rust"
    ```rust
    let model = Lowess::new()
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .adapter(Batch)
        .build()?;
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, confidence_intervals=0.95, prediction_intervals=0.95)
    ```

=== "R"
    ```r
    result <- fastlowess(x, y, confidence_intervals = 0.95, prediction_intervals = 0.95)
    ```

---

### cross_validate

Automated fraction selection via cross-validation (Batch only).

See [Cross-Validation](cross-validation.md) for detailed usage.

=== "Rust"
    ```rust
    let model = Lowess::new()
        .cross_validate(KFold(5, &[0.2, 0.3, 0.5, 0.7]).seed(42))
        .adapter(Batch)
        .build()?;
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, cv_method="kfold", cv_k=5, cv_fractions=[0.2, 0.3, 0.5, 0.7])
    ```

=== "R"
    ```r
    result <- fastlowess(x, y, cv_method = "kfold", cv_k = 5, cv_fractions = c(0.2, 0.3, 0.5, 0.7))
    ```

---

## Streaming Parameters

Parameters specific to the Streaming adapter.

### chunk_size

Number of points processed per chunk.

- **Default**: 5000
- **Range**: [10, ∞)

### overlap

Number of overlapping points between consecutive chunks.

- **Default**: 500
- **Range**: [0, chunk_size)

### merge_strategy

How to combine overlapping regions.

| Strategy   | Behavior                   |
|------------|----------------------------|
| `Average`  | Average overlapping values |
| `Left`     | Keep left chunk values     |
| `Right`    | Keep right chunk values    |
| `Weighted` | Distance-weighted blend    |

See [Execution Modes](adapters.md) for usage examples.

---

## Online Parameters

Parameters specific to the Online adapter.

### window_capacity

Maximum number of points in the sliding window.

- **Default**: 1000
- **Range**: [3, ∞)

### min_points

Minimum points before smoothing output starts.

- **Default**: 2
- **Range**: [2, window_capacity]

### update_mode

Strategy for updating the smooth when new points arrive.

| Mode          | Behavior                  | Speed         |
|---------------|---------------------------|:-------------:|
| `Incremental` | Update only affected fits | Faster        |
| `Full`        | Recompute entire window   | More accurate |

See [Execution Modes](adapters.md) for usage examples.

---

## Next Steps

- [Weight Functions](kernels.md) — Kernel comparison
- [Robustness](robustness.md) — Outlier handling
- [Intervals](intervals.md) — Uncertainty quantification
- [Cross-Validation](cross-validation.md) — Automated parameter selection
- [Execution Modes](adapters.md) — Batch/Streaming/Online
