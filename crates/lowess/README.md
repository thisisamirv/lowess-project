# lowess

[![Crates.io](https://img.shields.io/crates/v/lowess.svg)](https://crates.io/crates/lowess)
[![Documentation](https://docs.rs/lowess/badge.svg)](https://docs.rs/lowess)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

A high-performance implementation of LOWESS (Locally Weighted Scatterplot Smoothing) in Rust. This crate provides a robust, production-ready implementation with support for confidence intervals, multiple kernel functions, and optimized execution modes.

> [!IMPORTANT]
> For parallelization or `ndarray` support, use [`fastLowess`](https://github.com/av746/fastLowess).

## Features

- **Robust Statistics**: IRLS with Bisquare, Huber, or Talwar weighting for outlier handling.
- **Uncertainty Quantification**: Point-wise standard errors, confidence intervals, and prediction intervals.
- **Optimized Performance**: Delta optimization for skipping dense regions and streaming/online modes for large or real-time datasets.
- **Parameter Selection**: Built-in cross-validation for automatic smoothing fraction selection.
- **Flexibility**: Multiple weight kernels (Tricube, Epanechnikov, etc.) and `no_std` support (requires `alloc`).
- **Validated**: Numerical agreement with R's `stats::lowess`.

## Robustness Advantages

This implementation is **more robust than R's `lowess`** due to two key design choices:

### MAD-Based Scale Estimation

For robustness weight calculations, this crate uses **Median Absolute Deviation (MAD)** for scale estimation:

```text
s = median(|r_i - median(r)|)
```

In contrast, R's `lowess` uses median of absolute residuals:

```text
s = median(|r_i|)
```

**Why MAD is more robust:**

- MAD is a **breakdown-point-optimal** estimator—it remains valid even when up to 50% of data are outliers.
- The median-centering step removes asymmetric bias from residual distributions.
- MAD provides consistent outlier detection regardless of whether residuals are centered around zero.

### Boundary Padding

This crate applies **boundary policies** (Extend, Reflect, Zero) at dataset edges:

- **Extend**: Repeats edge values to maintain local neighborhood size.
- **Reflect**: Mirrors data symmetrically around boundaries.
- **Zero**: Pads with zeros (useful for signal processing).
- **NoBoundary**: Original Cleveland behavior

R's `lowess` does not apply boundary padding, which can lead to:

- Biased estimates near boundaries due to asymmetric local neighborhoods.
- Increased variance at the edges of the smoothed curve.

### Gaussian Consistency Factor

For interval estimation (confidence/prediction), residual scale is computed using:

```text
sigma = 1.4826 * MAD
```

The factor 1.4826 = 1/Phi^-1(3/4) ensures consistency with the standard deviation under Gaussian assumptions.

## Performance Advantages

The Rust `lowess` crate demonstrates consistent performance improvements over R's `stats::lowess` across all tested scenarios. Median speedups range from **1.3x to 3.4x** across different categories, with peak speedups reaching **4.7x** in specific configurations. No regressions were observed; Rust was faster in all matched benchmarks.

### Category Comparison

| Category         | Matched | Median Speedup | Mean Speedup |
|------------------|---------|----------------|--------------|
| **Delta**        | 4       | **3.37x**      | 3.25x        |
| **Financial**    | 3       | **2.16x**      | 2.22x        |
| **Pathological** | 4       | **2.01x**      | 1.87x        |
| **Scalability**  | 3       | **1.97x**      | 1.94x        |
| **Iterations**   | 6       | **1.94x**      | 2.02x        |
| **Fraction**     | 6       | **1.85x**      | 1.93x        |
| **Scientific**   | 3       | **1.84x**      | 1.80x        |
| **Genomic**      | 2       | **1.31x**      | 1.31x        |

### Top 10 Performance Wins

| Benchmark       | Rust   | R      | Speedup   |
|-----------------|--------|--------|-----------|
| delta_medium    | 0.18ms | 0.85ms | **4.72x** |
| delta_large     | 0.14ms | 0.54ms | **3.97x** |
| delta_small     | 0.34ms | 0.95ms | **2.76x** |
| iterations_0    | 0.17ms | 0.43ms | **2.54x** |
| financial_5000  | 0.36ms | 0.89ms | **2.46x** |
| clustered       | 0.82ms | 1.97ms | **2.39x** |
| fraction_0.05   | 0.31ms | 0.72ms | **2.36x** |
| iterations_2    | 0.64ms | 1.43ms | **2.23x** |
| scale_10000     | 0.96ms | 2.08ms | **2.17x** |
| constant_y      | 0.65ms | 1.40ms | **2.17x** |

**Regressions: None identified.** Rust outperforms R in all matched benchmarks. Check [Benchmarks](https://github.com/thisisamirv/lowess/tree/bench/benchmarks) for detailed results and reproducible benchmarking code.

## Validation

The Rust `lowess` crate is a **numerical twin** of R's `lowess` implementation:

| Aspect          | Status         | Details                                    |
|-----------------|----------------|--------------------------------------------|
| **Accuracy**    | ✅ EXACT MATCH | Max diff < 1e-12 across all scenarios      |
| **Consistency** | ✅ PERFECT     | 15/15 scenarios pass with strict tolerance |
| **Robustness**  | ✅ VERIFIED    | Robust smoothing matches R exactly         |

Check [Validation](https://github.com/thisisamirv/lowess/tree/bench/validation) for detailed scenario results.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
lowess = "0.7"
```

For `no_std` environments:

```toml
[dependencies]
lowess = { version = "0.7", default-features = false }
```

## Quick Start

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];

    // Basic smoothing
    let result = Lowess::new()
        .fraction(0.5)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;

    println!("Smoothed values: {:?}", result.y);
    Ok(())
}
```

## Builder Methods

```rust
use lowess::prelude::*;

Lowess::new()
    // Smoothing span (0, 1]
    .fraction(0.5)

    // Robustness iterations
    .iterations(3)

    // Interpolation threshold
    .delta(0.01)

    // Kernel selection
    .weight_function(Tricube)

    // Robustness method
    .robustness_method(Bisquare)

    // Zero-weight fallback behavior
    .zero_weight_fallback(UseLocalMean)

    // Boundary handling (for edge effects)
    .boundary_policy(Extend)

    // Confidence intervals
    .confidence_intervals(0.95)

    // Prediction intervals
    .prediction_intervals(0.95)

    // Diagnostics
    .return_diagnostics()
    .return_residuals()
    .return_robustness_weights()

    // Cross-validation (for parameter selection)
    .cross_validate(KFold(5, &[0.3, 0.5, 0.7]).seed(123))

    // Convergence
    .auto_converge(1e-4)

    // Execution mode
    .adapter(Batch)

    // Build the model
    .build()?;
```

### Result Structure

```rust
pub struct LowessResult<T> {
    // Sorted x values
    pub x: Vec<T>,

    // Smoothed y values
    pub y: Vec<T>,

    // Point-wise standard errors
    pub standard_errors: Option<Vec<T>>,

    // Confidence intervals
    pub confidence_lower: Option<Vec<T>>,
    pub confidence_upper: Option<Vec<T>>,

    // Prediction intervals
    pub prediction_lower: Option<Vec<T>>,
    pub prediction_upper: Option<Vec<T>>,

    // Residuals
    pub residuals: Option<Vec<T>>,

    // Final IRLS weights
    pub robustness_weights: Option<Vec<T>>,

    // Diagnostics
    pub diagnostics: Option<Diagnostics<T>>,

    // Actual iterations used
    pub iterations_used: Option<usize>,

    // Selected fraction
    pub fraction_used: T,

    // CV RMSE per fraction
    pub cv_scores: Option<Vec<T>>,
}
```

## Streaming Processing

For datasets that don't fit in memory:

```rust
let mut processor = Lowess::new()
    .fraction(0.3)
    .iterations(2)
    .adapter(Streaming)
    .chunk_size(1000)
    .overlap(100)
    .build()?;

// Process data in chunks
for chunk in data_chunks {
    let result = processor.process_chunk(&chunk.x, &chunk.y)?;
}

// Finalize processing
let final_result = processor.finalize()?;
```

## Online Processing

For real-time data streams:

```rust
let mut processor = Lowess::new()
    .fraction(0.2)
    .iterations(1)
    .adapter(Online)
    .window_capacity(100)
    .build()?;

// Process points as they arrive
for (x, y) in data_stream {
    if let Some(output) = processor.add_point(x, y)? {
        println!("Smoothed: {}", output.smoothed);
    }
}
```

## Parameter Selection Guide

### Fraction (Smoothing Span)

- **0.1-0.3**: Local, captures rapid changes (wiggly)
- **0.4-0.6**: Balanced, general-purpose
- **0.7-1.0**: Global, smooth trends only
- **Default: 0.67** (2/3, Cleveland's choice)
- **Use CV** when uncertain

### Robustness Method

- **Bisquare** (default): Best all-around, smooth, efficient
- **Huber**: Theoretically optimal MSE

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

## Examples

Check the `examples` directory for more complex scenarios:

```bash
cargo run --example batch_smoothing
cargo run --example online_smoothing
cargo run --example streaming_smoothing
```

## MSRV

Rust **1.85.0** or later (2024 Edition).

## Related Work

- [fastLowess (Rust)](https://github.com/thisisamirv/fastlowess)
- [fastLowess (Python wrapper)](https://github.com/thisisamirv/fastlowess-py)
- [fastLowess (R wrapper)](https://github.com/thisisamirv/fastlowess-R)

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

## License

Licensed under either of

- Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
   ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## References

- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". *JASA*.
- Cleveland, W.S. (1981). "LOWESS: A Program for Smoothing Scatterplots". *The American Statistician*.
