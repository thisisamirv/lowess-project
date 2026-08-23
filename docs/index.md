<!-- markdownlint-disable MD033 -->
# LOWESS Project

The fastest, most robust, and most feature-complete language-agnostic LOWESS (Locally Weighted Scatterplot Smoothing) implementation for **Rust**, **Python**, **R**, **Julia**, **JavaScript**, **C++**, and **WebAssembly**.

## What is LOWESS?

LOWESS is a nonparametric regression method that fits smooth curves through scatter plots. At each point, it fits a weighted polynomial using nearby data, with weights decreasing smoothly with distance. This creates flexible, data-adaptive curves without assuming a global functional form.

![LOWESS Smoothing Concept](assets/diagrams/lowess_concept.svg)

**Key advantages:**

- **No parametric assumptions** — Adapts to local data structure
- **Robust to outliers** — With robustness iterations enabled
- **Uncertainty quantification** — Confidence and prediction intervals
- **Handles irregular sampling** — Works with missing regions gracefully

## Why this package?

### Speed

The `lowess` project beats the competition in terms of speed, whether in single-threaded or multi-threaded parallel execution.

![LOWESS Benchmarks](assets/diagrams/benchmark_comparison.svg)

For detailed benchmark comparisons, see the [Benchmarks](https://lowess.readthedocs.io/benchmarks/) page.

### Robustness

This implementation is *more robust* than R's `lowess` and Python's `statsmodels` due to two key design choices:

**MAD-Based Scale Estimation:**

For robustness weight calculations, this crate uses *Median Absolute Deviation (MAD)* for scale estimation:

$$s = \text{median}(|r_i - \text{median}(r)|)$$

In contrast, `statsmodels` and R's `lowess` uses the median of absolute residuals (MAR):

$$s = \text{median}(|r_i|)$$

- MAD is a *breakdown-point-optimal* estimator—it remains valid even when up to 50% of data are outliers.
- The median-centering step removes asymmetric bias from residual distributions.
- MAD provides consistent outlier detection regardless of whether residuals are centered around zero.

**Boundary Padding:**

This crate applies a range of different *boundary policies* at dataset edges:

- **Extend**: Repeats edge values to maintain local neighborhood size.
- **Reflect**: Mirrors data symmetrically around boundaries.
- **Zero**: Pads with zeros (useful for signal processing).
- **NoBoundary**: Original Cleveland behavior

`statsmodels` and R's `lowess` do not apply boundary padding, which can lead to:

- Biased estimates near boundaries due to asymmetric local neighborhoods.
- Increased variance at the edges of the smoothed curve.

### Features

A variety of features, supporting a range of use cases:

| Feature | This package | statsmodels | R (stats) |
| --- | --- | --- | --- |
| Kernel | 7 options | only Tricube | only Tricube |
| Robustness Weighting | 3 options | only Huber | only Huber |
| Scale Estimation | 3 options | only MAR | only MAR |
| Boundary Padding | 4 options | no padding | no padding |
| Zero Weight Fallback | 3 options | no | no |
| Auto Convergence | yes | no | no |
| Online Mode | yes | no | no |
| Streaming Mode | yes | no | no |
| Confidence Intervals | yes | no | no |
| Prediction Intervals | yes | no | no |
| Cross-Validation | 2 options | no | no |
| Parallel Execution | yes | no | no |
| GPU Acceleration | yes* | no | no |
| `no-std` Support | yes | no | no |

\* GPU acceleration is currently in beta and may not be available on all platforms.

## Installation

Currently available for R, Python, Rust, Julia, Node.js, and WebAssembly.

=== "Python"

    Install from PyPI:

    ```bash
    pip install fastlowess
    ```

    Or from conda-forge:

    ```bash
    conda install -c conda-forge fastlowess
    ```
See the [Installation Guide](getting-started/installation.md) for more options and details.

## Quick Example

=== "Python"

    ```python
    import fastlowess as fl
    import numpy as np

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])

    model = fl.Lowess(fraction=0.5, iterations=3)
    result = model.fit(x, y)
    print(result.y)
    ```
## Getting Started

1. [Installation](getting-started/installation.md) — Set up the library for your language
2. [Quick Start](getting-started/quickstart.md) — Basic usage examples
3. [Concepts](getting-started/concepts.md) — Understand how LOWESS works

<!-- markdownlint-disable MD033 -->
<div class="grid cards" markdown>

- :fontawesome-brands-rust: **Rust**

    ---

    Pure Rust crates with zero-copy ndarray support, parallel execution, and GPU acceleration.

    [:octicons-arrow-right-24: Rust API](api/rust.md)

- :fontawesome-brands-python: **Python**

    ---

    Native Python bindings via PyO3 with NumPy integration and pip installation.

    [:octicons-arrow-right-24: Python API](api/python.md)

- :simple-r: **R**

    ---

    R package with Bioconductor-style documentation and seamless integration.

    [:octicons-arrow-right-24: R API](api/r.md)

- :simple-julia: **Julia**

    ---

    Native Julia package with C FFI, supporting parallel execution and JLL dependencies.

    [:octicons-arrow-right-24: Julia API](api/julia.md)

</div>

<!-- markdownlint-disable MD033 -->
<div class="grid cards" markdown>

- :simple-nodedotjs: **Node.js**

    ---

    Native Node.js bindings with high-performance C++ core and support for asynchronous streaming.

    [:octicons-arrow-right-24: Node.js API](api/nodejs.md)

- :simple-webassembly: **WebAssembly**

    ---

    Optimized WebAssembly build for browsers and Node.js with zero-overhead data transfer.

    [:octicons-arrow-right-24: WebAssembly API](api/wasm.md)

</div>

<!-- markdownlint-disable MD033 -->
<div class="grid cards" markdown>

- :material-language-cpp: **C++**

    ---

    Native C++ bindings with RAII memory management and STL container support.

    [:octicons-arrow-right-24: C++ API](api/cpp.md)

</div>

## License

Dual-licensed under [MIT](https://opensource.org/licenses/MIT) or [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).
