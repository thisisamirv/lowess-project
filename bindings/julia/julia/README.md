<!-- markdownlint-disable MD024 MD033 -->
# LOWESS Project

<p align="center">
  <a href="https://juliahub.com/ui/Packages/General/FastLOWESS"><img src="https://img.shields.io/badge/Julia-9558B2?logo=julia&logoColor=white" alt="Julia"></a>
  <a href="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-julia.yml"><img src="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-julia.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/thisisamirv/lowess-project/main/dev/logo.png" alt="One LOWESS to Rule Them All" width="400">
  <br>
  <em>One LOWESS to Rule Them All</em>
</p>

The fastest, most robust, and most feature-complete language-agnostic LOWESS (Locally Weighted Scatterplot Smoothing) implementation for **Rust**, **Python**, **R**, **Julia**, **JavaScript**, **C++**, and **WebAssembly**.

The `lowess-project` also offers bindings for Rust, Python, R, Julia, Node.js, WebAssembly, and C++ — see the [full repository](https://github.com/thisisamirv/lowess-project).

---

## Installation & Documentation

> Currently available for R, Python, Rust, Julia, Node.js, WebAssembly, and C++. See the [Installation Guide](https://thisisamirv.github.io/lowess-project/julia/installation/) for detailed installation instructions.
>
> ### 📚 [View the full documentation](https://thisisamirv.github.io/lowess-project/julia/)

### GPU Backend

GPU acceleration (`wgpu`: Vulkan/Metal/DX12) is also supported for high-throughput batch smoothing. See the [GPU Backend guide](https://thisisamirv.github.io/lowess-project/julia/gpu-backend/) for details.

---

## LOESS vs. LOWESS

| Feature | LOESS | LOWESS (This Crate) |
| --- | --- | --- |
| **Polynomial Degree** | Linear, Quadratic, Cubic, Quartic | Linear (Degree 1) |
| **Dimensions** | Multivariate (n-D support) | Univariate (1-D only) |
| **Flexibility** | High (Distance metrics) | Standard |
| **Complexity** | Higher (Matrix inversion) | Lower (Weighted average/slope) |

Read more about how LOWESS works in the [Concepts](https://thisisamirv.github.io/lowess-project/julia/concepts/).

> **Note:** For a **LOESS** implementation, use [`loess-project`](https://github.com/thisisamirv/loess-project).

---

## Why this package?

### Speed

The `lowess` project beats the competition in terms of speed, whether in single-threaded or multi-threaded parallel execution. It is on average **200-327x faster** than Python's `statsmodels.lowess` and **2-3x faster** than R's `lowess`.

For more details on the performance comparison, see the [Benchmarks](https://thisisamirv.github.io/lowess-project/julia/benchmarks/) page.

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

```julia
using FastLOWESS

model = Lowess(;
    fraction=0.5,
    iterations=3,
    delta=NaN,  # NaN for auto
    weight_function="tricube",
    robustness_method="bisquare",
    scaling_method="mad",
    zero_weight_fallback="use_local_mean",
    boundary_policy="extend",
    confidence_intervals=NaN,
    prediction_intervals=NaN,
    return_diagnostics=true,
    return_residuals=true,
    return_robustness_weights=true,
    return_se=true,
    cv_fractions=Float64[], # e.g. [0.3, 0.5]
    cv_method="kfold",
    cv_k=5,
    cv_seed=123,
    auto_converge=NaN,
    parallel=true
)
custom_weights = ones(length(x))
result = fit(model, x, y; custom_weights=custom_weights)

# Result structure:
result.x,
result.y,
result.standard_errors,
result.confidence_lower,
result.confidence_upper,
result.prediction_lower,
result.prediction_upper,
result.residuals,
result.robustness_weights,
result.diagnostics,
result.iterations_used,
result.fraction_used,
result.cv_scores
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
