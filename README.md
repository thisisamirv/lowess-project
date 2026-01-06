# LOWESS Project

[![Rust](https://img.shields.io/badge/rust-1.89%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Crates.io](https://img.shields.io/crates/v/lowess.svg)](https://crates.io/crates/lowess)
[![PyPI](https://img.shields.io/pypi/v/fastlowess.svg)](https://pypi.org/project/fastlowess/)
[![Conda](https://anaconda.org/conda-forge/fastlowess/badges/version.svg)](https://anaconda.org/conda-forge/fastlowess)
[![R-universe](https://thisisamirv.r-universe.dev/badges/rfastlowess)](https://thisisamirv.r-universe.dev/rfastlowess)

**High-performance LOWESS (Locally Weighted Scatterplot Smoothing) implementations** — A comprehensive monorepo providing core algorithms, parallel/GPU acceleration, and language bindings for Python and R.

## Overview

This monorepo contains a complete ecosystem for LOWESS smoothing:

- **[`lowess`](crates/lowess)** - Core single-threaded implementation with `no_std` support
- **[`fastLowess`](crates/fastLowess)** - Parallel CPU and GPU-accelerated wrapper with ndarray integration  
- **[Python bindings](bindings/python)** - PyO3-based Python package
- **[R bindings](bindings/r)** - extendr-based R package

## Features

### Core Capabilities

- **Robust Statistics**: MAD-based scale estimation, IRLS with Bisquare/Huber/Talwar weighting
- **Uncertainty Quantification**: Standard errors, confidence intervals, prediction intervals
- **Optimized Performance**: Delta optimization, streaming/online modes
- **Parameter Selection**: Built-in cross-validation
- **Flexibility**: Multiple kernels (Tricube, Epanechnikov, Gaussian, Uniform)

### Performance

- **Rust vs R**: 1.3x - 4.7x faster than R's `stats::lowess`
- **Rust vs Python**: 100x - 1000x faster than `statsmodels`
- **Parallel Scaling**: Multi-core speedups for large datasets
- **GPU Acceleration**: Optional WGPU backend for massive datasets

### Robustness Advantages

This implementation is **more robust** than both R and Python alternatives:

1. **MAD-Based Scale**: Breakdown-point-optimal estimator (50% outlier tolerance)

   ```plaintext
   s = median(|r_i - median(r)|)
   ```

2. **Boundary Padding**: Prevents edge bias with Extend/Reflect/Zero policies

3. **Gaussian Consistency**: Proper σ = 1.4826 × MAD for interval estimation

---

## Installation & Quick Start

### Rust (Core)

```toml
[dependencies]
lowess = "0.99.3"
```

```rust
use lowess::prelude::*;

let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];

let result = Lowess::new()
    .fraction(0.5)
    .adapter(Batch)
    .build()?
    .fit(&x, &y)?;
```

### Rust (Parallel/GPU)

```toml
[dependencies]
fastLowess = "0.99.3"
```

```rust
use fastLowess::prelude::*;
use ndarray::Array1;

let x = Array1::linspace(0.0, 10.0, 100);
let y = x.mapv(|v| v.sin() + 0.1 * v);

let result = Lowess::new()
    .fraction(0.5)
    .parallel(true)  // Multi-core by default
    .build()?
    .fit(&x, &y)?;
```

### Python

```bash
pip install fastlowess
```

```python
import numpy as np
import fastlowess

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)

result = fastlowess.smooth(x, y, fraction=0.3)
```

### R

```r
install.packages("rfastlowess", repos = "https://thisisamirv.r-universe.dev")
```

```r
library(rfastlowess)

x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.2)

result <- fastlowess(x, y, fraction = 0.3)
```

---

## API Reference

### Rust API

#### Builder Methods

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

#### Result Structure

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

### Python API

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

**Result attributes:**

```python
result.x, result.y, result.standard_errors
result.confidence_lower, result.confidence_upper
result.prediction_lower, result.prediction_upper
result.residuals, result.robustness_weights
result.diagnostics, result.iterations_used
result.fraction_used, result.cv_scores
```

### R API

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

**Result structure:**

```r
result$x, result$y, result$standard_errors
result$confidence_lower, result$confidence_upper
result$prediction_lower, result$prediction_upper
result$residuals, result$robustness_weights
result$diagnostics, result$iterations_used
result$fraction_used, result$cv_scores
```

---

## Advanced Usage

### Streaming Processing

For datasets that don't fit in memory:

**Rust:**

```rust
let mut processor = Lowess::new()
    .fraction(0.3)
    .adapter(Streaming)
    .chunk_size(1000)
    .overlap(100)
    .build()?;

for chunk in data_chunks {
    processor.process_chunk(&chunk.x, &chunk.y)?;
}
let result = processor.finalize()?;
```

**Python:**

```python
result = fastlowess.smooth_streaming(
    x, y, fraction=0.3, chunk_size=5000, overlap=500
)
```

**R:**

```r
result <- fastlowess_streaming(
    x, y, fraction = 0.3, chunk_size = 5000L, overlap = 500L
)
```

### Online Processing

For real-time data streams:

**Rust:**

```rust
let mut processor = Lowess::new()
    .fraction(0.2)
    .adapter(Online)
    .window_capacity(100)
    .build()?;

for (x, y) in data_stream {
    if let Some(output) = processor.add_point(x, y)? {
        println!("Smoothed: {}", output.smoothed);
    }
}
```

**Python:**

```python
result = fastlowess.smooth_online(
    x, y, fraction=0.2, window_capacity=100
)
```

**R:**

```r
result <- fastlowess_online(
    x, y, fraction = 0.2, window_capacity = 100L
)
```

---

## Parameter Selection Guide

### Fraction (Smoothing Span)

- **0.1-0.3**: Local, captures rapid changes (wiggly)
- **0.4-0.6**: Balanced, general-purpose
- **0.7-1.0**: Global, smooth trends only
- **Default: 0.67** (2/3, Cleveland's choice)
- **Use CV** when uncertain

### Robustness Iterations

- **0**: Clean data, speed critical
- **1-2**: Light contamination
- **3**: Default, good balance (recommended)
- **4-5**: Heavy outliers
- **>5**: Diminishing returns

### Kernel Function

- **Tricube** (default): Best all-around, smooth, efficient
- **Epanechnikov**: Theoretically optimal MSE
- **Gaussian**: Very smooth, no compact support
- **Uniform**: Fastest, least smooth (moving average)

### Delta Optimization

- **None**: Small datasets (n < 1000)
- **0.01 × range(x)**: Good starting point for dense data
- **Manual tuning**: Adjust based on data density

---

## Performance Benchmarks

### Rust vs R

| Category         | Matched | Median Speedup | Mean Speedup |
|------------------|---------|----------------|--------------|
| **Delta**        | 4       | **3.37x**      | 3.25x        |
| **Financial**    | 3       | **2.16x**      | 2.22x        |
| **Pathological** | 4       | **2.01x**      | 1.87x        |
| **Scalability**  | 3       | **1.97x**      | 1.94x        |
| **Iterations**   | 6       | **1.94x**      | 2.02x        |

**Top Performance Wins:**

- delta_medium: **4.72x**
- delta_large: **3.97x**
- iterations_0: **2.54x**

### Rust vs Python (statsmodels)

Average speedup: **280x** | Maximum speedup: **1169x**

| Benchmark        | statsmodels | fastLowess    |
|------------------|-------------|---------------|
| scale_50000      | 6798.58ms   | **987-1169x** |
| scale_10000      | 408.13ms    | **378-270x**  |
| scientific_10000 | 479.57ms    | **316-461x**  |
| fraction_0.05    | 122.00ms    | **376-274x**  |

### R Package Performance

`rfastlowess` vs `stats::lowess`: **1.1x - 6.8x** speedup (average 2.3x)

| Benchmark    | R         | rfastlowess |
|--------------|-----------|-------------|
| delta_none   | 164.35ms  | **1.5-6.8x**|
| genomic_50000| 2818.61ms | **1.6-5.5x**|
| clustered    | 2.16ms    | **2.4-4.7x**|

---

## Validation

All implementations are **numerical twins** of R's `lowess`:

| Aspect          | Status         | Details                                   |
|-----------------|----------------|-------------------------------------------|
| **Accuracy**    | ✅ EXACT MATCH | Max diff < 1e-12 across all scenarios     |
| **Consistency** | ✅ PERFECT     | 15/15 scenarios pass with strict tolerance|
| **Robustness**  | ✅ VERIFIED    | Robust smoothing matches R exactly        |

---

## Development

### Workspace Structure

This monorepo uses **Cargo workspace inheritance** for centralized configuration:

```toml
# Root Cargo.toml
[workspace.package]
version = "0.99.3"
authors = ["Amir Valizadeh <thisisamirv@gmail.com>"]
edition = "2024"
license = "MIT OR Apache-2.0"
rust-version = "1.89.0"
readme = "README.md"
# ... and more

[workspace.dependencies]
lowess = { version = "0.99.3", path = "crates/lowess" }
fastLowess = { version = "0.99.3", path = "crates/fastLowess" }
# ... shared dependencies
```

All member crates inherit from workspace:

```toml
# Individual crate Cargo.toml
[package]
name = "lowess"
version = { workspace = true }
authors = { workspace = true }
readme = { workspace = true }
# ...
```

**Benefits:**

- Single source of truth for versions and metadata
- Update once → all crates bump together
- Consistent MSRV across all packages

### Building from Source

```bash
# Build all crates
cargo build --all

# Run tests
cargo test --all

# Build Python bindings
cd bindings/python
maturin develop

# Build R package
cd bindings/r
make check
```

---

## Documentation

- **Rust**: [docs.rs/lowess](https://docs.rs/lowess) | [docs.rs/fastLowess](https://docs.rs/fastLowess)
- **Python**: [fastlowess-py.readthedocs.io](https://fastlowess-py.readthedocs.io/)
- **R**: [thisisamirv.github.io/rfastlowess](https://thisisamirv.github.io/rfastlowess/)

## Contributing

Contributions are welcome! Please see individual crate CONTRIBUTING.md files for guidelines.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## References

- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". *JASA*.
- Cleveland, W.S. (1981). "LOWESS: A Program for Smoothing Scatterplots". *The American Statistician*.

## Citation

```bibtex
@software{lowess_rust,
  author = {Valizadeh, Amir},
  title = {LOWESS: High-Performance Locally Weighted Scatterplot Smoothing},
  year = {2024},
  url = {https://github.com/thisisamirv/lowess-project}
}
```
