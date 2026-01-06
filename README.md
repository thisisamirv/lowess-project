# LOWESS Project

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
- **[`Python bindings`](bindings/python)** - PyO3-based Python package
- **[`R bindings`](bindings/r)** - extendr-based R package

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

## Installation

**Rust (Core):**

```toml
[dependencies]
lowess = "0.99.3"
```

**Rust (Parallel/GPU):**

```toml
[dependencies]
fastLowess = "0.99.3"
```

**Python:**

Install via PyPI:

```bash
pip install fastlowess
```

Or install from conda-forge:

```bash
conda install -c conda-forge fastlowess
```

**R:**

```r
install.packages("rfastlowess", repos = "https://thisisamirv.r-universe.dev")
```

---

## API Reference

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

---

## Result Structure

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

**Python:**

```python
result.x, result.y, result.standard_errors
result.confidence_lower, result.confidence_upper
result.prediction_lower, result.prediction_upper
result.residuals, result.robustness_weights
result.diagnostics, result.iterations_used
result.fraction_used, result.cv_scores
```

**R:**

```r
result$x, result$y, result$standard_errors
result$confidence_lower, result$confidence_upper
result$prediction_lower, result$prediction_upper
result$residuals, result$robustness_weights
result$diagnostics, result$iterations_used
result$fraction_used, result$cv_scores
```

## Streaming Processing

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

## Online Processing

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

## Performance Advantages

The `fastLowess` crate achieves massive performance gains over Python's `statsmodels` and R's `stats::lowess`:

### Performance Summary by Category

| Category         | statsmodels   | R        | Rust (Serial) | Rust (Parallel) | Rust (GPU) |
|------------------|---------------|----------|---------------|-----------------|------------|
| **Delta**        | 0.5-414.9ms   | 1-3x     | 4-5x          | 2-13x           | 0.1-64x    |
| **Financial**    | 6.50-302.2ms  | 58-168x  | 113-296x      | 58-611x         | 3-26x      |
| **Fraction**     | 122.0-362.6ms | 81-178x  | 129-421x      | 350-428x        | 15-32x     |
| **Genomic**      | 17.8-6776.6ms | 2-16x    | 4-19x         | 11-33x          | 7-269x     |
| **Iterations**   | 45.2-495.7ms  | 109-128x | 204-266x      | 345-405x        | 11-27x     |
| **Pathological** | 133.6-489.0ms | 83-133x  | 134-212x      | 372-433x        | 8-32x      |
| **Scale**        | 18.0-6798.6ms | 83-661x  | 150-1077x     | 107-1264x       | 8-277x     |
| **Scientific**   | 8.6-479.6ms   | 50-191x  | 91-370x       | 52-663x         | 3-35x      |

**Key Takeaways**:

1. **Rust (Parallel CPU)** is the dominant performer for general-purpose workloads, consistently achieving the highest speedups (often 300x-500x over statsmodels).
2. **R (stats::lowess)** is a very strong runner-up, frequently outperforming statsmodels by ~80-150x, but generally trailing Rust Parallel.
3. **Rust (GPU)** excels in specific high-compute scenarios (e.g., genomic with large datasets or delta_none where interpolation is skipped), but carries overhead that makes it slower than the highly optimized CPU backend for smaller datasets.
4. **Large Scale Scaling**: At very large scales (100k - 2M points), the parallel CPU backend maintains a modest lead (1.3x - 1.5x) over the sequential CPU backend, likely bottlenecked by memory bandwidth rather than compute.
5. **Small vs Large Delta**: Setting `delta=0` (no interpolation) allows the GPU to shine (63.5x speedup), outperforming both CPU variants due to the massive O(N²) interaction workload being parallelized across thousands of GPU cores.

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
  year = {2026},
  url = {https://github.com/thisisamirv/lowess-project}
}
```
