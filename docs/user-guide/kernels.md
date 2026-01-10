<!-- markdownlint-disable MD024 -->
# Weight Functions

Kernel functions for distance weighting.

## Overview

Weight functions (kernels) determine how neighboring points contribute to each local fit. Points closer to the target receive higher weights.

![Weight Functions](../assets/diagrams/ketnels.svg)

---

## Available Kernels

| Kernel           | Efficiency | Smoothness  | Support   |
|------------------|:----------:|:-----------:|:---------:|
| **Tricube**      | 0.998      | Very smooth | Compact   |
| **Epanechnikov** | 1.000      | Smooth      | Compact   |
| **Gaussian**     | 0.961      | Infinite    | Unbounded |
| **Biweight**     | 0.995      | Very smooth | Compact   |
| **Cosine**       | 0.999      | Smooth      | Compact   |
| **Triangle**     | 0.989      | Moderate    | Compact   |
| **Uniform**      | 0.943      | None        | Compact   |

**Efficiency** = AMISE relative to Epanechnikov (1.0 = optimal)

---

## Tricube (Default)

Cleveland's original choice. Best all-around performance.

$$w(u) = (1 - |u|^3)^3$$

**Use when**: Default choice for most applications.

=== "R"
    ```r
    result <- fastlowess(x, y, weight_function = "tricube")
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, weight_function="tricube")
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .weight_function(Tricube)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = smooth(x, y, weight_function="tricube")
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { weightFunction: "tricube" });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { weightFunction: "tricube" });
    ```

=== "C++"
    ```cpp
    auto result = fastlowess::smooth(x, y, { .weight_function = "tricube" });
    ```

---

## Epanechnikov

Theoretically optimal for kernel density estimation.

$$w(u) = \frac{3}{4}(1 - u^2)$$

**Use when**: Optimal MSE properties desired.

=== "R"
    ```r
    result <- fastlowess(x, y, weight_function = "epanechnikov")
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, weight_function="epanechnikov")
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .weight_function(Epanechnikov)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = smooth(x, y, weight_function="epanechnikov")
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { weightFunction: "epanechnikov" });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { weightFunction: "epanechnikov" });
    ```

=== "C++"
    ```cpp
    auto result = fastlowess::smooth(x, y, { .weight_function = "epanechnikov" });
    ```

---

## Gaussian

Infinitely smooth. No boundary effects.

$$w(u) = \exp(-u^2/2)$$

**Use when**: Maximum smoothness needed, computational cost acceptable.

=== "R"
    ```r
    result <- fastlowess(x, y, weight_function = "gaussian")
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, weight_function="gaussian")
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .weight_function(Gaussian)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = smooth(x, y, weight_function="gaussian")
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { weightFunction: "gaussian" });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { weightFunction: "gaussian" });
    ```

=== "C++"
    ```cpp
    auto result = fastlowess::smooth(x, y, { .weight_function = "gaussian" });
    ```

---

## Biweight

Good balance of efficiency and smoothness.

$$w(u) = (1 - u^2)^2$$

**Use when**: Alternative to Tricube with slightly different properties.

=== "R"
    ```r
    result <- fastlowess(x, y, weight_function = "biweight")
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, weight_function="biweight")
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .weight_function(Biweight)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = smooth(x, y, weight_function="biweight")
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { weightFunction: "biweight" });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { weightFunction: "biweight" });
    ```

=== "C++"
    ```cpp
    auto result = fastlowess::smooth(x, y, { .weight_function = "biweight" });
    ```

---

## Cosine

Smooth and computationally efficient.

$$w(u) = \cos(\pi u / 2)$$

**Use when**: Want smooth kernel with simple form.

=== "R"
    ```r
    result <- fastlowess(x, y, weight_function = "cosine")
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, weight_function="cosine")
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .weight_function(Cosine)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = smooth(x, y, weight_function="cosine")
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { weightFunction: "cosine" });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { weightFunction: "cosine" });
    ```

=== "C++"
    ```cpp
    auto result = fastlowess::smooth(x, y, { .weight_function = "cosine" });
    ```

---

## Triangle

Simple linear taper.

$$w(u) = 1 - |u|$$

**Use when**: Simple, interpretable weights.

=== "R"
    ```r
    result <- fastlowess(x, y, weight_function = "triangle")
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, weight_function="triangle")
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .weight_function(Triangle)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = smooth(x, y, weight_function="triangle")
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { weightFunction: "triangle" });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { weightFunction: "triangle" });
    ```

=== "C++"
    ```cpp
    auto result = fastlowess::smooth(x, y, { .weight_function = "triangle" });
    ```

---

## Uniform

Equal weights within window. Fastest but least smooth.

$$w(u) = 1$$

**Use when**: Speed is critical, smoothness less important.

=== "R"
    ```r
    result <- fastlowess(x, y, weight_function = "uniform")
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, weight_function="uniform")
    ```

=== "Rust"
    ```rust
    let model = Lowess::new()
        .weight_function(Uniform)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = smooth(x, y, weight_function="uniform")
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { weightFunction: "uniform" });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { weightFunction: "uniform" });
    ```

=== "C++"
    ```cpp
    auto result = fastlowess::smooth(x, y, { .weight_function = "uniform" });
    ```

---

## Choosing a Kernel

```mermaid
flowchart TD
    A[Choose Kernel] --> B{Need maximum smooth}
    B -- Yes --> C[Gaussian]
    B -- No --> D{Default acceptable}
    D -- Yes --> E[Tricube]
    D -- No --> F{Optimal MSE}
    F -- Yes --> G[Epanechnikov]
    F -- No --> H{Speed critical}
    H -- Yes --> I[Uniform]
    H -- No --> J[Biweight]
```

!!! tip "Recommendation"
    Stick with **Tricube** (default) unless you have specific requirements. The differences between kernels are usually small in practice.
