# Robustness

Outlier handling through iterative reweighting.

## How Robustness Works

Standard LOWESS can be biased by outliers. Robustness iterations downweight points with large residuals:

1. Fit initial LOWESS
2. Compute residuals
3. Assign robustness weights (large residuals → low weight)
4. Refit using combined distance × robustness weights
5. Repeat steps 2–4

![Robustness Methods](../assets/diagrams/robust_method_comparison.svg)

![Robustness Iterations](../assets/diagrams/robust_iter_comparison.svg)

---

## Robustness Methods

### Bisquare (Default)

Smooth downweighting. Points transition gradually from full weight to zero.

$$w(u) = \begin{cases} (1 - u^2)^2 & |u| < 1 \\ 0 & |u| \geq 1 \end{cases}$$

**Use when**: General purpose, balanced approach.

=== "R"
    ```r
    model <- Lowess(iterations = 3, robustness_method = "bisquare")
    result <- model$fit(x, y)
    ```

=== "Python"
    ```python
    model = fl.Lowess(iterations=3, robustness_method="bisquare")
    result = model.fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .iterations(3)
        .robustness_method("bisquare")
        .build()?;
    let result = model.fit(&x, &y)?;
    ```

=== "Julia"
    ```julia
    model = Lowess(; iterations=3, robustness_method="bisquare")
    result = fit(model, x, y)
    ```

=== "Node.js"
    ```javascript
    const model = new Lowess({ iterations: 3, robustness_method: "bisquare" });
    const result = model.fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const model = new Lowess({ iterations: 3, robustness_method: "bisquare" });
    const result = model.fit(x, y);
    ```

=== "C++"
    ```cpp
    fastlowess::Lowess model({ .iterations = 3,
        .robustness_method = "bisquare"
     });
    auto result = model.fit(x, y).value();
    ```

---

### Huber

Linear penalty beyond threshold. Less aggressive than Bisquare.

$$w(u) = \begin{cases} 1 & |u| \leq k \\ k/|u| & |u| > k \end{cases}$$

**Use when**: Moderate outliers, want to retain some influence.

=== "R"
    ```r
    model <- Lowess(iterations = 3, robustness_method = "huber")
    result <- model$fit(x, y)
    ```

=== "Python"
    ```python
    model = fl.Lowess(iterations=3, robustness_method="huber")
    result = model.fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .iterations(3)
        .robustness_method("huber")
        .build()?;
    let result = model.fit(&x, &y)?;
    ```

=== "Julia"
    ```julia
    model = Lowess(; iterations=3, robustness_method="huber")
    result = fit(model, x, y)
    ```

=== "Node.js"
    ```javascript
    const model = new Lowess({ iterations: 3, robustness_method: "huber" });
    const result = model.fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const model = new Lowess({ iterations: 3, robustness_method: "huber" });
    const result = model.fit(x, y);
    ```

=== "C++"
    ```cpp
    fastlowess::Lowess model({ .iterations = 3,
        .robustness_method = "huber"
     });
    auto result = model.fit(x, y).value();
    ```

---

### Talwar

Hard threshold. Points are either fully weighted or completely excluded.

$$w(u) = \begin{cases} 1 & |u| \leq k \\ 0 & |u| > k \end{cases}$$

**Use when**: Extreme outliers, want binary exclusion.

=== "R"
    ```r
    model <- Lowess(iterations = 3, robustness_method = "talwar")
    result <- model$fit(x, y)
    ```

=== "Python"
    ```python
    model = fl.Lowess(iterations=3, robustness_method="talwar")
    result = model.fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .iterations(3)
        .robustness_method("talwar")
        .build()?;
    let result = model.fit(&x, &y)?;
    ```

=== "Julia"
    ```julia
    model = Lowess(; iterations=3, robustness_method="talwar")
    result = fit(model, x, y)
    ```

=== "Node.js"
    ```javascript
    const model = new Lowess({ iterations: 3, robustness_method: "talwar" });
    const result = model.fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const model = new Lowess({ iterations: 3, robustness_method: "talwar" });
    const result = model.fit(x, y);
    ```

=== "C++"
    ```cpp
    fastlowess::Lowess model({ .iterations = 3,
        .robustness_method = "talwar"
     });
    auto result = model.fit(x, y).value();
    ```

---

## Comparison

| Method | Transition | Aggressiveness | Use Case |
| --- | --- | --- | --- |
| **Bisquare** | Smooth | Moderate | General purpose |
| **Huber** | Gradual | Mild | Preserve influence |
| **Talwar** | Hard | Strong | Extreme contamination |

---

## Detecting Outliers

Use robustness weights to identify potential outliers:

=== "R"
    ```r
    model <- Lowess(iterations = 5, return_robustness_weights = TRUE)
    result <- model$fit(x, y)

    weights <- result$robustness_weights
    outliers <- which(weights < 0.5)
    cat("Potential outliers at indices:", outliers, "\n")
    ```

=== "Python"
    ```python
    model = fl.Lowess(iterations=5, return_robustness_weights=True)
    result = model.fit(x, y)

    for i, w in enumerate(result.robustness_weights):
        if w < 0.5:
            print(f"Potential outlier at index {i}: weight = {w:.3f}")
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .iterations(5)
        .return_robustness_weights()
        .build()?;

    let result = model.fit(&x, &y)?;
    
    if let Some(weights) = &result.robustness_weights {
        for (i, &w) in weights.iter().enumerate() {
            if w < 0.5 {
                println!("Potential outlier at index {}: weight = {:.3}", i, w);
            }
        }
    }
    ```

=== "Julia"
    ```julia
    model = Lowess(; iterations=5, return_robustness_weights=true)
    result = fit(model, x, y)

    for (i, w) in enumerate(result.robustness_weights)
        if w < 0.5
            println("Potential outlier at index $i: weight = $w")
        end
    end
    ```

=== "Node.js"
    ```javascript
    const model = new Lowess({ iterations: 5, return_robustness_weights: true });
    const result = model.fit(x, y);

    result.robustness_weights.forEach((w, i) => {
        if (w < 0.5) {
            console.log(`Potential outlier at index ${i}: weight = ${w.toFixed(3)}`);
        }
    });
    ```

=== "WebAssembly"
    ```javascript
    const model = new Lowess({ iterations: 5, return_robustness_weights: true });
    const result = model.fit(x, y);

    result.robustness_weights.forEach((w, i) => {
        if (w < 0.5) {
            console.log(`Potential outlier at index ${i}: weight = ${w.toFixed(3)}`);
        }
    });
    ```

=== "C++"
    ```cpp
    fastlowess::Lowess model({ .iterations = 5,
        .return_robustness_weights = true
     });
    auto result = model.fit(x, y).value();

    auto weights = result.robustness_weights();
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights[i] < 0.5) {
            std::cout << "Potential outlier at " << i << std::endl;
        }
    }
    ```

---

## Scale Estimation

Residuals are scaled before computing robustness weights. Two methods:

| Method | Formula | Robustness |
| --- | --- | --- |
| **MAD** | `median(\|r − median(r)\|)` | Very robust (default) |
| **MAR** | `median(\|r\|)` | Robust, uncentered |
| **Mean** | `mean(\|r\|)` | Less robust, fastest |

![Scaling Methods Comparison](../assets/diagrams/scaling_comparison.svg)

=== "R"
    ```r
    model <- Lowess(iterations = 3, scaling_method = "mad")
    result <- model$fit(x, y)
    ```

=== "Python"
    ```python
    model = fl.Lowess(iterations=3, scaling_method="mad")
    result = model.fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .iterations(3)
        .scaling_method("mad")  // Default
        .build()?;
    let result = model.fit(&x, &y)?;
    ```

=== "Julia"
    ```julia
    model = Lowess(; iterations=3, scaling_method="mad")
    result = fit(model, x, y)
    ```

=== "Node.js"
    ```javascript
    const model = new Lowess({ iterations: 3, scaling_method: "mad" });
    const result = model.fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const model = new Lowess({ iterations: 3, scaling_method: "mad" });
    const result = model.fit(x, y);
    ```

=== "C++"
    ```cpp
    fastlowess::Lowess model({ .iterations = 3,
        .scaling_method = "mad"
     });
    auto result = model.fit(x, y).value();
    ```

---

## Auto-Convergence

Stop iterations early when weights stabilize:

!!! tip "Performance"
    Auto-convergence can significantly reduce computation when weights stabilize before reaching max iterations.

=== "R"
    ```r
    model <- Lowess(iterations = 10, auto_converge = 1e-6)
    result <- model$fit(x, y)
    ```

=== "Python"
    ```python
    model = fl.Lowess(iterations=10, auto_converge=1e-6)
    result = model.fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .iterations(10)           // Maximum iterations
        .auto_converge(1e-6)      // Stop when change < 1e-6
        .build()?;
    let result = model.fit(&x, &y)?;
    ```

=== "Julia"
    ```julia
    model = Lowess(; iterations=10, auto_converge=1e-6)
    result = fit(model, x, y)
    ```

=== "Node.js"
    ```javascript
    const model = new Lowess({ iterations: 10, auto_converge: 1e-6 });
    const result = model.fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const model = new Lowess({ iterations: 10, auto_converge: 1e-6 });
    const result = model.fit(x, y);
    ```

=== "C++"
    ```cpp
    fastlowess::Lowess model({ .iterations = 10,
        .auto_converge = 1e-6
     });
    auto result = model.fit(x, y).value();
    ```
