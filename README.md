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

| Name                  | statsmodels |      R      |  Rust (CPU)*  | Rust (GPU)|
|-----------------------|-------------|-------------|---------------|-----------|
| clustered             |  162.77ms   |  [82.8x]²   |  [203-433x]¹  |   32.4x   |
| constant_y            |  133.63ms   |  [92.3x]²   |  [212-410x]¹  |   17.5x   |
| delta_large           |   0.51ms    |   [0.8x]²   |  [3.8-2.2x]¹  |   0.1x    |
| delta_medium          |   0.79ms    |   [1.3x]²   |  [4.4-3.4x]¹  |   0.1x    |
| delta_none            |  414.86ms   |    2.5x     |  [3.8-13x]²   | [63.5x]¹  |
| delta_small           |   1.45ms    |   [1.7x]²   |  [4.3-4.5x]¹  |   0.2x    |
| extreme_outliers      |  488.96ms   |  [106.4x]²  |  [201-388x]¹  |   28.9x   |
| financial_1000        |   13.55ms   |  [76.6x]²   |  [145-108x]¹  |   4.7x    |
| financial_10000       |  302.20ms   |  [168.3x]²  |  [453-611x]¹  |   26.3x   |
| financial_500         |   6.49ms    |  [58.0x]¹   |  [113-58x]²   |   2.7x    |
| financial_5000        |  103.94ms   |  [117.3x]²  |  [296-395x]¹  |   14.1x   |
| fraction_0.05         |  122.00ms   |  [177.6x]²  |  [421-350x]¹  |   14.5x   |
| fraction_0.1          |  140.59ms   |  [112.8x]²  |  [291-366x]¹  |   15.9x   |
| fraction_0.2          |  181.57ms   |  [85.3x]²   |  [210-419x]¹  |   19.3x   |
| fraction_0.3          |  220.98ms   |  [84.8x]²   |  [168-380x]¹  |   22.4x   |
| fraction_0.5          |  296.47ms   |  [80.9x]²   |  [146-415x]¹  |   27.3x   |
| fraction_0.67         |  362.59ms   |  [83.1x]²   |  [129-413x]¹  |   32.0x   |
| genomic_1000          |   17.82ms   |  [15.9x]²   |   [19-33x]¹   |   6.5x    |
| genomic_10000         |  399.90ms   |    3.6x     |  [5.3-16x]²   | [70.3x]¹  |
| genomic_5000          |  138.49ms   |    5.0x     |  [7.0-19x]²   | [34.8x]¹  |
| genomic_50000         |  6776.57ms  |    2.4x     |  [3.5-11x]²   | [269.2x]¹ |
| high_noise            |  435.85ms   |  [132.6x]²  |  [134-375x]¹  |   32.3x   |
| iterations_0          |   45.18ms   |  [128.4x]²  |  [266-405x]¹  |   10.6x   |
| iterations_1          |   94.10ms   |  [114.3x]²  |  [236-384x]¹  |   14.4x   |
| iterations_10         |  495.65ms   |  [116.0x]²  |  [204-369x]¹  |   27.0x   |
| iterations_2          |  135.48ms   |  [109.0x]²  |  [219-432x]¹  |   16.6x   |
| iterations_3          |  181.56ms   |  [108.8x]²  |  [213-382x]¹  |   18.7x   |
| iterations_5          |  270.58ms   |  [110.4x]²  |  [208-345x]¹  |   22.7x   |
| scale_1000            |   17.95ms   |  [82.6x]²   |  [150-107x]¹  |   8.1x    |
| scale_10000           |  408.13ms   |  [178.1x]²  |  [433-552x]¹  |   76.3x   |
| scale_5000            |  139.81ms   |  [133.6x]²  |  [289-401x]¹  |   28.8x   |
| scale_50000           |  6798.58ms  |  [661.0x]²  | [1077-1264x]¹ |  277.2x   |
| scientific_1000       |   19.04ms   |  [70.1x]²   |  [113-115x]¹  |   5.4x    |
| scientific_10000      |  479.57ms   |  [190.7x]²  |  [370-663x]¹  |   35.2x   |
| scientific_500        |   8.59ms    |  [49.6x]²   |   [91-52x]¹   |   3.2x    |
| scientific_5000       |  161.42ms   |  [124.9x]²  |  [244-427x]¹  |   17.9x   |
| scale_100000**        |      -      |      -      |    1-1.3x     |   0.3x    |
| scale_1000000**       |      -      |      -      |    1-1.3x     |   0.3x    |
| scale_2000000**       |      -      |      -      |    1-1.5x     |   0.3x    |
| scale_250000**        |      -      |      -      |    1-1.4x     |   0.3x    |
| scale_500000**        |      -      |      -      |    1-1.3x     |   0.3x    |

\* **Rust (CPU)**: Shows range `Seq - Par`. E.g., `12-48x` means 12x speedup (Sequential) and 48x speedup (Parallel). Rank determined by Parallel speedup.
\*\* **Large Scale**: `Rust (Serial)` is the baseline (1x).

¹ Winner (Fastest implementation)

² Runner-up (Second fastest implementation)

**Key Takeaways**:

1. **Rust (Parallel CPU)** is the dominant performer for general-purpose workloads, consistently achieving the highest speedups (often 300x-500x over statsmodels).
2. **R (stats::lowess)** is a very strong runner-up, frequently outperforming statsmodels by ~80-150x, but generally trailing Rust Parallel.
3. **Rust (GPU)** excels in specific high-compute scenarios (e.g., `genomic` with large datasets or `delta_none` where interpolation is skipped), but carries overhead that makes it slower than the highly optimized CPU backend for smaller datasets.
4. **Large Scale Scaling**: At very large scales (100k - 2M points), the parallel CPU backend maintains a modest lead (1.3x - 1.5x) over the sequential CPU backend, likely bottlenecked by memory bandwidth rather than compute.
5. **Small vs Large Delta**: Setting `delta=0` (no interpolation, `delta_none`) allows the GPU to shine (63.5x speedup), outperforming both CPU variants due to the massive O(N²) interaction workload being parallelized across thousands of GPU cores.

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
