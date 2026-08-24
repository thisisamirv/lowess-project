<!-- markdownlint-disable MD024 MD033 -->
# LOWESS Project

<p align="center">
  <a href="https://thisisamirv.r-universe.dev/rfastlowess"><img src="https://img.shields.io/badge/R--universe-276DC3?logo=r&logoColor=white" alt="R-universe"></a>
  <a href="https://anaconda.org/conda-forge/r-rfastlowess"><img src="https://img.shields.io/badge/rfastlowess_(R)-44A833?logo=anaconda&logoColor=white" alt="rfastlowess (R)"></a>
  <a href="https://github.com/ropensci/software-review/issues/769"><img src="https://badges.ropensci.org/769_status.svg" alt="Status at rOpenSci Software Peer Review"></a>
  <a href="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-r.yml"><img src="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-r.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/thisisamirv/lowess-project/main/dev/logo.png" alt="One LOWESS to Rule Them All" width="400">
  <br>
  <em>One LOWESS to Rule Them All</em>
</p>

The fastest, most robust, and most feature-complete language-agnostic LOWESS (Locally Weighted Scatterplot Smoothing) implementation for **Rust**, **Python**, **R**, **Julia**, **JavaScript**, **C++**, and **WebAssembly**.

The `lowess-project` also offers bindings for Rust, Python, R, Julia, Node.js, WebAssembly, and C++ — see the [full repository](https://github.com/thisisamirv/lowess-project).

---

## Installation

> [!NOTE]
>
> Currently available for R, Python, Rust, Julia, Node.js, WebAssembly, and C++. See the [Installation Guide](https://thisisamirv.github.io/lowess-project/r/articles/installation.html) for detailed installation instructions.

### GPU Backend

In addition to `parallel = true` (multi-core CPU), the batch `Lowess` class in every binding — except WebAssembly — as well as the `fastLowess` Rust crate itself, can run on the GPU via `wgpu` (Vulkan/Metal/DX12). It's opt-in and worth enabling for high-throughput processing of large datasets (roughly 10k+ points); for smaller inputs the CPU backend is typically faster. `StreamingLowess`/`OnlineLowess` remain CPU-only. See the [GPU Backend guide](https://thisisamirv.github.io/lowess-project/r/articles/gpu-backend.html) for installation instructions and usage.

## Documentation

> [!NOTE]
>
> ### 📚 [View the full documentation](https://thisisamirv.github.io/lowess-project/r/)

---

## LOESS vs. LOWESS

| Feature | LOESS | LOWESS (This Crate) |
| --- | --- | --- |
| **Polynomial Degree** | Linear, Quadratic, Cubic, Quartic | Linear (Degree 1) |
| **Dimensions** | Multivariate (n-D support) | Univariate (1-D only) |
| **Flexibility** | High (Distance metrics) | Standard |
| **Complexity** | Higher (Matrix inversion) | Lower (Weighted average/slope) |

> [!TIP]
> **Note:** For a **LOESS** implementation, use [`loess-project`](https://github.com/thisisamirv/loess-project).

---

## Why this package?

### Speed

The `lowess` project beats the competition in terms of speed, whether in single-threaded or multi-threaded parallel execution. It is on average **200-327x faster** than Python's `statsmodels.lowess` and **2-3x faster** than R's `lowess`.

For more details on the performance comparison, see the [Benchmarks](https://thisisamirv.github.io/lowess-project/r/articles/benchmarks.html) page.

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

| Feature | This package | statsmodels | R (stats) |
| --- | :---: | :---: | :---: |
| Kernel | 7 options | only Tricube | only Tricube |
| Robustness Weighting | 3 options | only Huber | only Huber |
| Scale Estimation | 2 options | only MAR | only MAR |
| Boundary Padding | 4 options | no padding | no padding |
| Zero Weight Fallback | 3 options | no | no |
| Auto Convergence | yes | no | no |
| Online Mode | yes | no | no |
| Streaming Mode | yes | no | no |
| Confidence Intervals | yes | no | no |
| Prediction Intervals | yes | no | no |
| Cross-Validation | 2 options | no | no |
| Parallel Execution | yes | no | no |
| GPU Acceleration | yes | no | no |
| `no-std` Support | yes | no | no |

## Validation

All implementations are **numerical twins** of R's `lowess`:

| Aspect | Status | Details |
| --- | --- | --- |
| **Accuracy** | ✅ EXACT MATCH | Max diff < 1e-12 across all scenarios |
| **Consistency** | ✅ PERFECT | Multiple scenarios pass with strict tolerance |
| **Robustness** | ✅ VERIFIED | Robust smoothing matches R exactly |

## API Reference

```r
library(rfastlowess)

model <- Lowess(
    fraction = 0.5,
    iterations = 3L,
    delta = 0.01,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    zero_weight_fallback = "use_local_mean",
    boundary_policy = "extend",
    confidence_intervals = 0.95,
    prediction_intervals = 0.95,
    return_diagnostics = TRUE,
    return_residuals = TRUE,
    return_robustness_weights = TRUE,
    return_se = TRUE,
    cv_fractions = c(0.3, 0.5, 0.7),
    cv_method = "kfold",
    cv_k = 5L,
    cv_seed = 123L,
    auto_converge = 1e-4,
    parallel = TRUE
)
custom_weights <- rep(1, length(x))
result <- fit(model, x, y, custom_weights = custom_weights)

# Result structure:
result$x,
result$y,
result$standard_errors,
result$confidence_lower,
result$confidence_upper,
result$prediction_lower,
result$prediction_upper,
result$residuals,
result$robustness_weights,
result$diagnostics,
result$iterations_used,
result$fraction_used,
result$cv_scores
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](https://github.com/thisisamirv/lowess-project/blob/main/CONTRIBUTING.md) for more information.

## Changelog

See [CHANGELOG.md](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md) for a history of changes.

## License

Licensed under [MIT](https://github.com/thisisamirv/lowess-project/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/thisisamirv/lowess-project/blob/main/LICENSE-APACHE).

## Citation

If you use this software in your research, please cite it using the [CITATION.cff](https://github.com/thisisamirv/lowess-project/blob/main/CITATION.cff) file or the BibTeX entry below:

```bibtex
@software{lowess_project,
  author = {Valizadeh, Amir},
  title = {LOWESS Project: High-Performance Locally Weighted Scatterplot Smoothing},
  year = {2026},
  url = {https://github.com/thisisamirv/lowess-project},
  license = {MIT OR Apache-2.0}
}
```
