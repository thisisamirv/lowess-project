<!-- markdownlint-disable MD033 -->
# Custom Weights

Per-observation weights that encode data quality directly into the LOWESS fit.

## How Custom Weights Work

Standard LOWESS assigns equal prior trust to all observations. Custom weights
let you override this assumption point by point — before any distance or
robustness weighting is applied.

The effective weight of observation $j$ in a local fit centred at $x_i$ is:

$$w_{ij} = \text{custom\_weights}[j] \times K\!\left(\frac{d_{ij}}{h_i}\right) \times r_j$$

where $K$ is the distance kernel, $h_i$ is the local bandwidth, and $r_j$ is
the robustness weight from the current iteration.

!!! note "Batch adapter only"
    `custom_weights` applies in **Batch** mode. It is silently ignored in
    Streaming and Online adapters.

---

## When to Use Custom Weights

| Situation | Recommended weight |
| --- | --- |
| Point known to be erroneous | `0.0` — fully excluded |
| Unreliable sensor / low precision | `0.1 – 0.5` |
| Standard observation | `1.0` (default) |
| Carefully calibrated measurement | `> 1.0` |
| Measurement uncertainty $\sigma_i$ | $1 / \sigma_i^2$ |

### Custom Weights vs. Robustness Iterations

Both mechanisms handle unreliable data, but they serve different purposes:

| | Custom Weights | Robustness Iterations |
| --- | --- | --- |
| **When known** | Before fitting | Computed from residuals |
| **Knowledge required** | Prior knowledge of quality | None — data-driven |
| **Effect** | Fixed throughout fit | Adapts each iteration |
| **Use case** | Known bad sensors, calibration | Unknown outlier contamination |

They compose: you can use both simultaneously. Custom weights suppress
*a priori* bad points; robustness iterations then handle any *residual*
outliers that remain.

---

## Basic Usage

### Suppress a Known Outlier

Set the weight to `0` at the bad point — it is excluded from every local fit
that would otherwise include it.

=== "R"
    ```r
    x <- 1:10
    y <- x * 2.0
    y[6] <- 100.0              # spike at index 6

    weights <- rep(1.0, 10)
    weights[6] <- 0.0          # exclude the spike

    model <- Lowess(fraction = 0.5, iterations = 0L)
    result <- model$fit(x, y, custom_weights = weights)
    ```

=== "Python"
    ```python
    import numpy as np
    from fastlowess import Lowess

    x = np.arange(10, dtype=float)
    y = x * 2.0
    y[5] = 100.0               # spike at index 5

    weights = np.ones(10)
    weights[5] = 0.0           # exclude the spike

    result = Lowess(fraction=0.5, iterations=0, custom_weights=weights.tolist()).fit(x, y)
    ```

=== "Rust"
    ```rust
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let mut y: Vec<f64> = x.iter().map(|v| v * 2.0).collect();
    y[5] = 100.0; // spike

    let mut weights = vec![1.0_f64; 10];
    weights[5] = 0.0; // exclude the spike

    let model = Lowess::new()
        .fraction(0.5)
        .iterations(0)
        .custom_weights(weights)
        .build()?;

    let result = model.fit(&x, &y)?;
    ```

=== "Julia"
    ```julia
    x = collect(1.0:10.0)
    y = x .* 2.0
    y[6] = 100.0               # spike at index 6 (1-indexed)

    weights = ones(10)
    weights[6] = 0.0           # exclude the spike

    model = Lowess(fraction = 0.5, iterations = 0)
    result = fit(model, x, y; custom_weights = weights)
    ```

=== "Node.js"
    ```javascript
    const fastlowess = require('fastlowess');

    const x = Float64Array.from({length: 10}, (_, i) => i);
    const y = Float64Array.from(x, v => v * 2);
    y[5] = 100.0; // spike

    const weights = new Float64Array(10).fill(1.0);
    weights[5] = 0.0; // exclude the spike

    const result = new fastlowess.Lowess({
        fraction: 0.5, iterations: 0, custom_weights: weights
    }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    import init, { smooth } from 'fastlowess_wasm.js';
    await init();

    const x = Float64Array.from({length: 10}, (_, i) => i);
    const y = Float64Array.from(x, v => v * 2);
    y[5] = 100.0; // spike

    const weights = new Float64Array(10).fill(1.0);
    weights[5] = 0.0; // exclude the spike

    const result = smooth(x, y, { fraction: 0.5, iterations: 0, custom_weights: weights });
    ```

=== "C++"
    ```cpp
    std::vector<double> x(10), y(10);
    for (std::size_t i = 0; i < 10; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = x[i] * 2.0;
    }
    y[5] = 100.0; // spike

    fastlowess::LowessOptions opts;
    opts.fraction = 0.5;
    opts.iterations = 0;

    std::vector<double> weights(10, 1.0);
    weights[5] = 0.0; // exclude the spike

    auto result = fastlowess::Lowess(opts).fit(x, y, weights).value();
    ```

---

### Emphasize Important Points

Assign high weights to measurements you trust most — calibration standards,
reference instruments, or low-noise observations.

=== "R"
    ```r
    weights <- rep(1.0, length(x))
    weights[calibration_indices] <- 10.0   # trust calibration 10× more

    result <- Lowess(fraction = 0.5)$fit(x, y, custom_weights = weights)
    ```

=== "Python"
    ```python
    weights = [1.0] * len(x)
    for i in calibration_indices:
        weights[i] = 10.0      # trust calibration 10× more

    result = Lowess(fraction=0.5, custom_weights=weights).fit(x, y)
    ```

=== "Rust"
    ```rust
    let mut weights = vec![1.0_f64; x.len()];
    for &i in &calibration_indices {
        weights[i] = 10.0; // trust calibration 10× more
    }

    let model = Lowess::new()
        .fraction(0.5)
        .custom_weights(weights)
        .build()?;
    ```

=== "Julia"
    ```julia
    weights = ones(length(x))
    weights[calibration_indices] .= 10.0   # trust calibration 10× more

    result = fit(Lowess(fraction = 0.5), x, y; custom_weights = weights)
    ```

=== "Node.js"
    ```javascript
    const weights = new Float64Array(x.length).fill(1.0);
    for (const i of calibrationIndices) weights[i] = 10.0;

    const result = new fastlowess.Lowess({
        fraction: 0.5, custom_weights: weights
    }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const weights = new Float64Array(x.length).fill(1.0);
    for (const i of calibrationIndices) weights[i] = 10.0;

    const result = smooth(x, y, { fraction: 0.5, custom_weights: weights });
    ```

=== "C++"
    ```cpp
    std::vector<double> weights(x.size(), 1.0);
    for (std::size_t idx : calibration_indices) weights[idx] = 10.0;

    auto result = fastlowess::Lowess(opts).fit(x, y, weights).value();
    ```

---

### Propagate Measurement Uncertainty

If each observation has a known standard deviation $\sigma_i$, set
$w_i = 1 / \sigma_i^2$ to give the fit information-theoretically optimal
weighting.

=== "R"
    ```r
    # sigma is a vector of measurement uncertainties
    weights <- 1.0 / sigma^2

    result <- Lowess(fraction = 0.5)$fit(x, y, custom_weights = weights)
    ```

=== "Python"
    ```python
    weights = (1.0 / sigma**2).tolist()
    result = Lowess(fraction=0.5, custom_weights=weights).fit(x, y)
    ```

=== "Rust"
    ```rust
    let weights: Vec<f64> = sigma.iter().map(|s| 1.0 / (s * s)).collect();

    let model = Lowess::new()
        .fraction(0.5)
        .custom_weights(weights)
        .build()?;
    ```

=== "Julia"
    ```julia
    weights = 1.0 ./ sigma .^ 2
    result = fit(Lowess(fraction = 0.5), x, y; custom_weights = weights)
    ```

=== "Node.js"
    ```javascript
    const weights = Float64Array.from(sigma, s => 1.0 / (s * s));
    const result = new fastlowess.Lowess({
        fraction: 0.5, custom_weights: weights
    }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const weights = Float64Array.from(sigma, s => 1.0 / (s * s));
    const result = smooth(x, y, { fraction: 0.5, custom_weights: weights });
    ```

=== "C++"
    ```cpp
    std::vector<double> weights(sigma.size());
    std::transform(sigma.begin(), sigma.end(), weights.begin(),
                   [](double s) { return 1.0 / (s * s); });

    auto result = fastlowess::Lowess(opts).fit(x, y, weights).value();
    ```

---

## Combined with Robustness Iterations

Custom weights and robustness iterations compose naturally: use custom weights
for *known* bad points and robustness for *unknown* contamination.

=== "R"
    ```r
    x <- 0:19
    y <- x * 1.5
    y[4]  <- -50.0   # known bad — zero out
    y[13] <- 80.0    # unknown outlier — let robustness handle it

    weights <- rep(1.0, 20)
    weights[4] <- 0.0

    result <- Lowess(fraction = 0.4, iterations = 3L)$fit(
        x, y, custom_weights = weights
    )
    ```

=== "Python"
    ```python
    x = list(range(20))
    y = [xi * 1.5 for xi in x]
    y[3]  = -50.0   # known bad
    y[12] = 80.0    # unknown outlier

    weights = [1.0] * 20
    weights[3] = 0.0

    result = Lowess(fraction=0.4, iterations=3, custom_weights=weights).fit(x, y)
    ```

=== "Rust"
    ```rust
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let mut y: Vec<f64> = x.iter().map(|v| v * 1.5).collect();
    y[3]  = -50.0; // known bad
    y[12] = 80.0;  // unknown outlier

    let mut weights = vec![1.0_f64; 20];
    weights[3] = 0.0;

    let model = Lowess::new()
        .fraction(0.4)
        .iterations(3)
        .custom_weights(weights)
        .build()?;
    ```

=== "Julia"
    ```julia
    x = collect(0.0:19.0)
    y = x .* 1.5
    y[4]  = -50.0   # known bad (1-indexed)
    y[13] = 80.0    # unknown outlier (1-indexed)

    weights = ones(20)
    weights[4] = 0.0

    result = fit(Lowess(fraction = 0.4, iterations = 3), x, y; custom_weights = weights)
    ```

=== "Node.js"
    ```javascript
    const y = Float64Array.from({length: 20}, (_, i) => i * 1.5);
    y[3]  = -50.0;  // known bad
    y[12] = 80.0;   // unknown outlier

    const weights = new Float64Array(20).fill(1.0);
    weights[3] = 0.0;

    const result = new fastlowess.Lowess({
        fraction: 0.4, iterations: 3, custom_weights: weights
    }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const weights = new Float64Array(20).fill(1.0);
    weights[3] = 0.0;

    const result = smooth(x, y, {
        fraction: 0.4, iterations: 3, custom_weights: weights
    });
    ```

=== "C++"
    ```cpp
    std::vector<double> weights(20, 1.0);
    weights[3] = 0.0;

    fastlowess::LowessOptions opts;
    opts.fraction = 0.4;
    opts.iterations = 3;

    auto result = fastlowess::Lowess(opts).fit(x, y, weights).value();
    ```

---

## Validation Rules

| Rule | Effect |
| --- | --- |
| Length must equal `n` | Error at fit time if mismatched |
| All values must be ≥ 0 | Negative weights are rejected |
| All-zero weight vector | Error: no points remain for any local fit |
| Uniform weights (`1.0` everywhere) | Identical result to omitting weights |

!!! warning "Zero-weight windows"
    If a local neighbourhood contains only zero-weight points, the fit at
    that centre point falls back to the behaviour specified by
    `zero_weight_fallback` (default: `"use_local_mean"`).

---

## See Also

- [Robustness](robustness.md) — adaptive outlier downweighting via IRLS
- [Parameters](parameters.md#custom_weights) — full parameter reference
