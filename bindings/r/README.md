# LOWESS Project

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://lowess.readthedocs.io/)
[![Crates.io](https://img.shields.io/crates/v/lowess.svg)](https://crates.io/crates/lowess)
[![PyPI](https://img.shields.io/pypi/v/fastlowess.svg)](https://pypi.org/project/fastlowess/)
[![Conda](https://anaconda.org/conda-forge/fastlowess/badges/version.svg)](https://anaconda.org/conda-forge/fastlowess)
[![R-universe](https://thisisamirv.r-universe.dev/badges/rfastlowess)](https://thisisamirv.r-universe.dev/rfastlowess)

The fastest, most robust, and most feature-complete language-agnostic LOWESS (Locally Weighted Scatterplot Smoothing) implementation for **Rust**, **Python**, and **R**.

> [!IMPORTANT]
>
> The `lowess-project` contains a complete ecosystem for LOWESS smoothing:
>
> - **[`lowess`](https://github.com/thisisamirv/lowess-project/crates/lowess)** - Core single-threaded Rust implementation with `no_std` support
> - **[`fastLowess`](https://github.com/thisisamirv/lowess-project/crates/fastLowess)** - Parallel CPU and GPU-accelerated Rust wrapper with ndarray integration  
> - **[`Python bindings`](https://github.com/thisisamirv/lowess-project/bindings/python)** - PyO3-based Python package
> - **[`R bindings`](https://github.com/thisisamirv/lowess-project/bindings/r)** - extendr-based R package

## LOESS vs. LOWESS

| Feature               | LOESS (This Crate)                | LOWESS                         |
|-----------------------|-----------------------------------|--------------------------------|
| **Polynomial Degree** | Linear, Quadratic, Cubic, Quartic | Linear (Degree 1)              |
| **Dimensions**        | Multivariate (n-D support)        | Univariate (1-D only)          |
| **Flexibility**       | High (Distance metrics)           | Standard                       |
| **Complexity**        | Higher (Matrix inversion)         | Lower (Weighted average/slope) |

> [!TIP]
> **Note:** For a **LOESS** implementation, use [`loess-project`](https://github.com/thisisamirv/loess-project).

---

## Documentation

> [!NOTE]
>
> ### 📚 [View the full documentation](https://lowess.readthedocs.io/)

## Why this package?

### Speed

The `lowess` project crushes the competition in terms of speed, wether in single-threaded or multi-threaded parallel execution.

Speedup relative to Python's `statsmodels.lowess` (higher is better):

| Category                        | statsmodels | R (stats) | Serial | Parallel | GPU     |
|---------------------------------|-------------|-----------|--------|----------|---------|
| **Clustered**                   | 163ms       | 83×       | 203×   | **433×** | 32×     |
| **Constant Y**                  | 134ms       | 92×       | 212×   | **410×** | 18×     |
| **Delta** (large–none)          | 105ms       | 2×        | 4×     | 6×       | **16×** |
| **Extreme Outliers**            | 489ms       | 106×      | 201×   | **388×** | 29×     |
| **Financial** (500–10K)         | 106ms       | 105×      | 252×   | **293×** | 12×     |
| **Fraction** (0.05–0.67)        | 221ms       | 104×      | 228×   | **391×** | 22×     |
| **Genomic** (1K–50K)            | 1833ms      | 7×        | 9×     | 20×      | **95×** |
| **High Noise**                  | 435ms       | 133×      | 134×   | **375×** | 32×     |
| **Iterations** (0–10)           | 204ms       | 115×      | 224×   | **386×** | 18×     |
| **Scale** (1K–50K)              | 1841ms      | 264×      | 487×   | **581×** | 98×     |
| **Scientific** (500–10K)        | 167ms       | 109×      | 205×   | **314×** | 15×     |
| **Scale Large**\* (100K–2M)     | —           | —         | 1×     | **1.4×** | 0.3×    |

\*Scale Large benchmarks are relative to Serial (statsmodels cannot handle these sizes)

*The numbers are the average across a range of scenarios for each category (e.g., Delta from none, to small, medium, and large).*

### Robustness

This implementation is *more robust* than R's `lowess` and Python's `statsmodels` due to two key design choices:

**MAD-Based Scale Estimation:**

For robustness weight calculations, this crate uses *Median Absolute Deviation (MAD)* for scale estimation:

```text
s = median(|r_i - median(r)|)
```

In contrast, `statsmodels` and R's `lowess` uses the median of absolute residuals (MAR):

```text
s = median(|r_i|)
```

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

| Feature              | This package  | statsmodels  | R (stats)    |
|----------------------|:-------------:|:------------:|:------------:|
| Kernel               | 7 options     | only Tricube | only Tricube |
| Robustness Weighting | 3 options     | only Huber   | only Huber   |
| Scale Estimation     | 2 options     | only MAR     | only MAR     |
| Boundary Padding     | 4 options     | no padding   | no padding   |
| Zero Weight Fallback | 3 options     | no           | no           |
| Auto Convergence     | yes           | no           | no           |
| Online Mode          | yes           | no           | no           |
| Streaming Mode       | yes           | no           | no           |
| Confidence Intervals | yes           | no           | no           |
| Prediction Intervals | yes           | no           | no           |
| Cross-Validation     | 2 options     | no           | no           |
| Parallel Execution   | yes           | no           | no           |
| GPU Acceleration     | yes*          | no           | no           |
| `no-std` Support     | yes           | no           | no           |

\* GPU acceleration is currently in beta and may not be available on all platforms.

## Validation

All implementations are **numerical twins** of R's `lowess`:

| Aspect          | Status         | Details                                       |
|-----------------|----------------|-----------------------------------------------|
| **Accuracy**    | ✅ EXACT MATCH | Max diff < 1e-12 across all scenarios         |
| **Consistency** | ✅ PERFECT     | Multiple scenarios pass with strict tolerance |
| **Robustness**  | ✅ VERIFIED    | Robust smoothing matches R exactly            |

## Installation

Currently available for R, Python, and Rust:

**R** (from R-universe, recommended):

```r
install.packages("rfastlowess", repos = "https://thisisamirv.r-universe.dev")
```

**Python** (from PyPI):

```bash
pip install fastlowess
```

Or from conda-forge:

```bash
conda install -c conda-forge fastlowess
```

**Rust** (lowess, no_std compatible):

```toml
[dependencies]
lowess = "0.99"
```

**Rust** (fastLowess, parallel + GPU):

```toml
[dependencies]
fastLowess = { version = "0.99", features = ["cpu"] }
```

## Quick Example

**R:**

```r
library(rfastlowess)

x <- c(1, 2, 3, 4, 5)
y <- c(2.0, 4.1, 5.9, 8.2, 9.8)

result <- fastlowess(x, y, fraction = 0.5, iterations = 3)
print(result$y)
```

**Python:**

```python
import fastlowess as fl
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])

result = fl.smooth(x, y, fraction=0.5, iterations=3)
print(result["y"])
```

**Rust:**

```rust
use lowess::prelude::*;

let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];

let model = Lowess::new()
    .fraction(0.5)
    .iterations(3)
    .adapter(Batch)
    .build()?;

let result = model.fit(&x, &y)?;
println!("{}", result);
```

---

## API Reference

**R:**

```r
fastlowess(
    x, y,
    fraction = 0.5,
    iterations = 3L,
    delta = 0.01,
    weight_function = "tricube",
    robustness_method = "bisquare",
    zero_weight_fallback = "use_local_mean",
    boundary_policy = "extend",
    confidence_intervals = 0.95,
    prediction_intervals = 0.95,
    return_diagnostics = TRUE,
    return_residuals = TRUE,
    return_robustness_weights = TRUE,
    cv_fractions = c(0.3, 0.5, 0.7),
    cv_method = "kfold",
    cv_k = 5L,
    auto_converge = 1e-4,
    parallel = TRUE
)
```

**Python:**

```python
fastlowess.smooth(
    x, y,
    fraction=0.5,
    iterations=3,
    delta=0.01,
    weight_function="tricube",
    robustness_method="bisquare",
    zero_weight_fallback="use_local_mean",
    boundary_policy="extend",
    confidence_intervals=0.95,
    prediction_intervals=0.95,
    return_diagnostics=True,
    return_residuals=True,
    return_robustness_weights=True,
    cv_fractions=[0.3, 0.5, 0.7],
    cv_method="kfold",
    cv_k=5,
    auto_converge=1e-4,
    parallel=True
)
```

**Rust:**

```rust
Lowess::new()
    .fraction(0.5)              // Smoothing span (0, 1]
    .iterations(3)              // Robustness iterations
    .delta(0.01)                // Interpolation threshold
    .weight_function(Tricube)   // Kernel selection
    .robustness_method(Bisquare)
    .zero_weight_fallback(UseLocalMean)
    .boundary_policy(Extend)
    .confidence_intervals(0.95)
    .prediction_intervals(0.95)
    .return_diagnostics()
    .return_residuals()
    .return_robustness_weights()
    .cross_validate(KFold(5, &[0.3, 0.5, 0.7]).seed(123))
    .auto_converge(1e-4)
    .adapter(Batch)             // or Streaming, Online
    .parallel(true)             // fastLowess only
    .backend(CPU)               // fastLowess only: CPU or GPU
    .build()?;
```

---

## Result Structure

**R:**

```r
result$x, result$y, result$standard_errors
result$confidence_lower, result$confidence_upper
result$prediction_lower, result$prediction_upper
result$residuals, result$robustness_weights
result$diagnostics, result$iterations_used
result$fraction_used, result$cv_scores
```

**Python:**

```python
result.x, result.y, result.standard_errors
result.confidence_lower, result.confidence_upper
result.prediction_lower, result.prediction_upper
result.residuals, result.robustness_weights
result.diagnostics, result.iterations_used
result.fraction_used, result.cv_scores
```

**Rust:**

```rust
pub struct LowessResult<T> {
    pub x: Vec<T>,                           // Sorted x values
    pub y: Vec<T>,                           // Smoothed y values
    pub standard_errors: Option<Vec<T>>,
    pub confidence_lower: Option<Vec<T>>,
    pub confidence_upper: Option<Vec<T>>,
    pub prediction_lower: Option<Vec<T>>,
    pub prediction_upper: Option<Vec<T>>,
    pub residuals: Option<Vec<T>>,
    pub robustness_weights: Option<Vec<T>>,
    pub diagnostics: Option<Diagnostics<T>>,
    pub iterations_used: Option<usize>,
    pub fraction_used: T,
    pub cv_scores: Option<Vec<T>>,
}
```

---

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

## References

- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". *JASA*.
- Cleveland, W.S. (1981). "LOWESS: A Program for Smoothing Scatterplots". *The American Statistician*.

## Citation

If you use this software in your research, please cite it using the [CITATION.cff](CITATION.cff) file or the BibTeX entry below:

```bibtex
@software{lowess_project,
  author = {Valizadeh, Amir},
  title = {LOWESS Project: High-Performance Locally Weighted Scatterplot Smoothing},
  year = {2026},
  url = {https://github.com/thisisamirv/lowess-project},
  license = {MIT OR Apache-2.0}
}
```
