# fastLowess

[![Crates.io](https://img.shields.io/crates/v/fastLowess.svg)](https://crates.io/crates/fastLowess)
[![Documentation](https://docs.rs/fastLowess/badge.svg)](https://docs.rs/fastLowess)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

**High-performance parallel and GPU-accelerated LOWESS (Locally Weighted Scatterplot Smoothing) for Rust** — A high-level wrapper around the [`lowess`](https://github.com/thisisamirv/lowess) crate that adds rayon-based parallelism, GPU acceleration, and seamless ndarray integration.

> [!IMPORTANT]
> For a minimal, single-threaded, and `no_std` version, use base [`lowess`](https://github.com/thisisamirv/lowess).

## Features

- **Parallel by Default**: Multi-core regression fits via [rayon](https://crates.io/crates/rayon), achieving multiple orders of magnitude speedups on large datasets.
- **ndarray Integration**: Native support for `Array1<T>` and `ArrayView1<T>`.
- **Robust Statistics**: MAD-based scale estimation and IRLS with Bisquare, Huber, or Talwar weighting.
- **Uncertainty Quantification**: Point-wise standard errors, confidence intervals, and prediction intervals.
- **Optimized Performance**: Delta optimization for skipping dense regions and streaming/online modes.
- **Parameter Selection**: Built-in cross-validation for automatic smoothing fraction selection.

## Robustness Advantages

Built on the same core as `lowess`, this implementation is **more robust than statsmodels** due to:

### MAD-Based Scale Estimation

We use **Median Absolute Deviation (MAD)** for scale estimation, which is breakdown-point-optimal:

```text
s = median(|r_i - median(r)|)
```

### Boundary Padding

We apply **boundary policies** (Extend, Reflect, Zero) at dataset edges to maintain symmetric local neighborhoods, preventing the edge bias common in other implementations.

### Gaussian Consistency Factor

For precision in intervals, residual scale is computed using:

```text
sigma = 1.4826 * MAD
```

## Performance Advantages

The `fastLowess` crate demonstrates massive performance gains over Python's `statsmodels`. The Rust CPU backend is the decisive winner across almost all standard benchmarks, often achieving **multi-hundred-fold speedups**.

The table below shows speedups relative to the **baseline**.

- **Standard Benchmarks**: Baseline is `statsmodels` (Python).
- **Large Scale Benchmarks**: Baseline is `Rust (Serial)` (1x), as `statsmodels` times out.

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

## Installation

### CPU Backend (Default)

The default installation includes rayon-based parallelism and ndarray support:

```toml
[dependencies]
fastLowess = "0.4"
```

Or explicitly enable the `cpu` feature:

```toml
[dependencies]
fastLowess = { version = "0.4", features = ["cpu"] }
```

### GPU Backend

For GPU acceleration using `wgpu`, enable the `gpu` feature:

```toml
[dependencies]
fastLowess = { version = "0.4", features = ["gpu"] }
```

> [!NOTE]
> The GPU backend requires compatible GPU hardware and drivers. See the [Backend Comparison](#backend-comparison) section below for feature limitations.

## Quick Start

```rust
use fastLowess::prelude::*;
use ndarray::Array1;

fn main() -> Result<(), LowessError> {
    // Data as ndarray Array1
    let x = Array1::linspace(0.0, 10.0, 100);
    let y = x.mapv(|v| v.sin() + 0.1 * v);

    // Build the model (parallel by default)
    let result = Lowess::new()
        .fraction(0.5)
        .adapter(Batch)
        .parallel(true)
        .build()?
        .fit(&x, &y)?;

    println!("Smoothed values: {:?}", result.y);
    Ok(())
}
```

## Builder Methods

```rust
use fastLowess::prelude::*;

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
    .cross_validate(KFold(5).with_fractions(&[0.3, 0.5, 0.7]).seed(123))

    // Convergence
    .auto_converge(1e-4)

    // Execution mode
    .adapter(Batch)

    // Backend (CPU or GPU)
    .backend(CPU)

    // Parallelism
    .parallel(true)

    // Build the model
    .build()?;
```

### Backend Comparison

| Backend    | Use Case         | Features              | Limitations         |
|------------|------------------|-----------------------|---------------------|
| CPU        | General          | All features          | None                |
| GPU (beta) | High-performance | Special circumstances | Only vanilla LOWESS |

> [!WARNING]
> **GPU Backend Limitations**: The GPU backend is currently in **Beta** and is limited to vanilla LOWESS and does not support all features of the CPU backend:
>
> - Only Tricube kernel function
> - Only Bisquare robustness method
> - Only Batch adapter
> - No cross-validation
> - No intervals
> - No edge handling (bias at edges, original LOWESS behavior)
> - No zero-weight fallback
> - No diagnostics
> - No streaming or online mode

1. **CPU Backend (`Backend::CPU`)**: The default and recommended choice. It is faster for all standard dense computations, supports all features (cross-validation, intervals, etc.), and has zero setup overhead.

2. **GPU Backend (`Backend::GPU`)**: Use **only** if you have a massive dataset (> 10 million points) **AND** you are using no or very small `delta` optimization (e.g., `delta(0.01)`). In this specific "sparse" scenario, the GPU scales better than the CPU. for dense computation, the CPU is still faster.

> [!NOTE]
> **GPU vs CPU Precision**: Results from the GPU backend are not guaranteed to be identical to the CPU backend due to:
>
> - Different floating-point precision
> - No padding at the edges in the GPU backend
> - Different scale estimation methods (MAD in CPU, MAR in GPU)

## Result Structure

```rust
pub struct LowessResult<T> {
    /// Sorted x values (independent variable)
    pub x: Vec<T>,

    /// Smoothed y values (dependent variable)
    pub y: Vec<T>,

    /// Point-wise standard errors of the fit
    pub standard_errors: Option<Vec<T>>,

    /// Confidence interval bounds (if computed)
    pub confidence_lower: Option<Vec<T>>,
    pub confidence_upper: Option<Vec<T>>,

    /// Prediction interval bounds (if computed)
    pub prediction_lower: Option<Vec<T>>,
    pub prediction_upper: Option<Vec<T>>,

    /// Residuals (y - fit)
    pub residuals: Option<Vec<T>>,

    /// Final robustness weights from outlier downweighting
    pub robustness_weights: Option<Vec<T>>,

    /// Detailed fit diagnostics (RMSE, R^2, Effective DF, etc.)
    pub diagnostics: Option<Diagnostics<T>>,

    /// Number of robustness iterations actually performed
    pub iterations_used: Option<usize>,

    /// Smoothing fraction used (optimal if selected via CV)
    pub fraction_used: T,

    /// RMSE scores for each fraction tested during CV
    pub cv_scores: Option<Vec<T>>,
}
```

> [!TIP]
> **Using with ndarray:** While the result struct uses `Vec<T>` for maximum compatibility, you can effortlessly convert any field to an `Array1` using `Array1::from_vec(result.y)`.

## Streaming Processing

For datasets that don't fit in memory:

```rust
use fastLowess::prelude::*;

let mut processor = Lowess::new()
    .fraction(0.3)
    .iterations(2)
    .adapter(Streaming)
    .parallel(true)   // Enable parallel chunk processing
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
use fastLowess::prelude::*;

let mut processor = Lowess::new()
    .fraction(0.2)
    .iterations(1)
    .adapter(Online)
    .parallel(false)  // Sequential for lowest per-point latency
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

Check the `examples` directory for advanced usage:

```bash
cargo run --example batch_smoothing
cargo run --example online_smoothing
cargo run --example streaming_smoothing
```

## MSRV

Rust **1.85.0** or later (2024 Edition).

## Validation

Validated against:

- **Python (statsmodels)**: Passed on 44 distinct test scenarios.
- **Original Paper**: Reproduces Cleveland (1979) results.

Check [Validation](https://github.com/thisisamirv/fastLowess/tree/bench/validation) for more information. Small variations in results are expected due to differences in scale estimation and padding.

## Related Work

- [lowess (Rust core)](https://github.com/thisisamirv/lowess)
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
