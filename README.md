<!-- markdownlint-disable MD024 MD033 -->
# LOWESS Project

<p align="center">
  <a href="https://crates.io/crates/lowess"><img src="https://img.shields.io/badge/lowess-000000?logo=rust&logoColor=white" alt="lowess"></a>
  <a href="https://crates.io/crates/fastLowess"><img src="https://img.shields.io/badge/fastLowess-000000?logo=rust&logoColor=white" alt="fastLowess"></a>
  <a href="https://pypi.org/project/fastlowess/"><img src="https://img.shields.io/badge/PyPI-3775A9?logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://thisisamirv.r-universe.dev/rfastlowess"><img src="https://img.shields.io/badge/R--universe-276DC3?logo=r&logoColor=white" alt="R-universe"></a>
  <a href="https://www.npmjs.com/package/fastlowess"><img src="https://img.shields.io/badge/npm-CB3837?logo=npm&logoColor=white" alt="npm"></a>
  <a href="https://juliahub.com/ui/Packages/General/fastlowess_jll"><img src="https://img.shields.io/badge/Julia-9558B2?logo=julia&logoColor=white" alt="Julia"></a>
  <a href="https://www.npmjs.com/package/fastlowess-wasm"><img src="https://img.shields.io/badge/WASM-654FF0?logo=webassembly&logoColor=white" alt="WASM"></a>
  <a href="https://github.com/thisisamirv/lowess-project/releases/latest"><img src="https://img.shields.io/badge/C++-00599C?logo=cplusplus&logoColor=white" alt="C++"></a>
  <br>
  <a href="https://anaconda.org/conda-forge/fastlowess"><img src="https://img.shields.io/badge/fastlowess_(Python)-44A833?logo=anaconda&logoColor=white" alt="fastlowess (Python)"></a>
  <a href="https://anaconda.org/conda-forge/libfastlowess"><img src="https://img.shields.io/badge/libfastlowess_(C++)-44A833?logo=anaconda&logoColor=white" alt="libfastlowess (C++)"></a>
  <a href="https://anaconda.org/conda-forge/r-rfastlowess"><img src="https://img.shields.io/badge/rfastlowess_(R)-44A833?logo=anaconda&logoColor=white" alt="rfastlowess (R)"></a>
  <br>
  <a href="https://github.com/thisisamirv/lowess-project/actions/workflows/ci.yml"><img src="https://github.com/thisisamirv/lowess-project/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/thisisamirv/lowess-project/main/dev/logo.png" alt="One LOWESS to Rule Them All" width="400">
  <br>
  <em>One LOWESS to Rule Them All</em>
</p>

The fastest, most robust, and most feature-complete language-agnostic LOWESS (Locally Weighted Scatterplot Smoothing) implementation for **Rust**, **Python**, **R**, **Julia**, **JavaScript**, **C++**, and **WebAssembly**.

> [!IMPORTANT]
>
> The `lowess-project` contains a complete ecosystem for LOWESS smoothing:
>
> - **[`lowess`](https://crates.io/crates/lowess)** - Core single-threaded Rust implementation with `no_std` support
> - **[`fastLowess`](https://crates.io/crates/fastLowess)** - Parallel CPU and GPU-accelerated Rust wrapper with ndarray integration  
> - **[`R bindings`](https://thisisamirv.r-universe.dev/rfastlowess)** - extendr-based R binding
> - **[`Python bindings`](https://pypi.org/project/fastlowess/)** - PyO3-based Python binding
> - **[`Julia bindings`](https://juliahub.com/ui/Packages/General/fastlowess_jll)** - Native Julia binding with C FFI
> - **[`JavaScript bindings`](https://www.npmjs.com/package/fastlowess)** - Node.js binding
> - **[`WebAssembly bindings`](https://www.npmjs.com/package/fastlowess-wasm)** - WASM binding
> - **[`C++ bindings`](https://github.com/thisisamirv/lowess-project/releases/latest)** - Native C++ binding with CMake integration

---

## Installation

> [!NOTE]
>
> Currently available for R, Python, Rust, Julia, Node.js, WebAssembly, and C++. See [INSTALLATION.md](https://github.com/thisisamirv/lowess-project/blob/main/INSTALLATION.md) for detailed installation instructions.

## Documentation

> [!NOTE]
>
> ### 📚 [View the full documentation](https://lowess.readthedocs.io/)

---

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

## Why this package?

### Speed

The `lowess` project beats the competition in terms of speed, whether in single-threaded or multi-threaded parallel execution. It is on average **200-327x faster** than Python's `statsmodels.lowess` and **2-3x faster** than R's `lowess`.

For more details on the performance comparison, see the [BENCHMARKS](https://github.com/thisisamirv/lowess-project/blob/main/BENCHMARKS.md) file.

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

## API Reference

**R:**

```r
Lowess(
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
)$fit(x, y)

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

**Python:**

```python
from fastlowess import Lowess

model = Lowess(
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
result = model.fit(x, y)

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

**Rust:**

```rust
Lowess::new()
    .fraction(0.5)
    .iterations(3)
    .delta(0.01)
    .weight_function(Tricube)
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
    .adapter(Batch)
    .parallel(true)             // fastLowess only
    .backend(CPU)               // fastLowess only: CPU or GPU
    .build()?;

let result = model.fit(x, y);

// Result structure:
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

**Julia:**

```julia
Lowess(;
    fraction=0.5,
    iterations=3,
    delta=NaN,  # NaN for auto
    weight_function="tricube",
    robustness_method="bisquare",
    zero_weight_fallback="use_local_mean",
    boundary_policy="extend",
    confidence_intervals=NaN,
    prediction_intervals=NaN,
    return_diagnostics=true,
    return_residuals=true,
    return_robustness_weights=true,
    cv_fractions=Float64[], # e.g. [0.3, 0.5]
    cv_method="kfold",
    cv_k=5,
    auto_converge=NaN,
    parallel=true
)

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

**Node.js:**

```javascript
new Lowess({
    fraction: 0.5,
    iterations: 3,
    delta: 0.01,
    weightFunction: "tricube",
    robustnessMethod: "bisquare",
    zeroWeightFallback: "use_local_mean",
    boundaryPolicy: "extend",
    confidenceIntervals: 0.95,
    predictionIntervals: 0.95,
    returnDiagnostics: true,
    returnResiduals: true,
    returnRobustnessWeights: true,
    cvFractions: [0.3, 0.5, 0.7],
    cvMethod: "kfold",
    cvK: 5,
    autoConverge: 1e-4,
    parallel: true
}).fit(x, y)

// Result structure:
result.x,
result.y,
result.standardErrors,
result.confidenceLower,
result.confidenceUpper,
result.predictionLower,
result.predictionUpper,
result.residuals,
result.robustnessWeights,
result.diagnostics,
result.iterationsUsed,
result.fractionUsed,
result.cvScores
```

**WebAssembly:**

```javascript
smooth(x, y, {
    fraction: 0.5,
    iterations: 3,
    delta: 0.01,
    weightFunction: "tricube",
    robustnessMethod: "bisquare",
    zeroWeightFallback: "use_local_mean",
    boundaryPolicy: "extend",
    confidenceIntervals: 0.95,
    predictionIntervals: 0.95,
    returnDiagnostics: true,
    returnResiduals: true,
    returnRobustnessWeights: true,
    cvFractions: [0.3, 0.5, 0.7],
    cvMethod: "kfold",
    cvK: 5,
    autoConverge: 1e-4,
    parallel: true
})

// Result structure:
result.x,
result.y,
result.standardErrors,
result.confidenceLower,
result.confidenceUpper,
result.predictionLower,
result.predictionUpper,
result.residuals,
result.robustnessWeights,
result.diagnostics,
result.iterationsUsed,
result.fractionUsed,
result.cvScores
```

**C++:**

```cpp
fastlowess::LowessOptions options;
options.fraction = 0.5;
options.iterations = 3;
options.delta = 0.01;
options.weight_function = "tricube";
options.robustness_method = "bisquare";
options.zero_weight_fallback = "use_local_mean";
options.boundary_policy = "extend";
options.confidence_intervals = 0.95;
options.prediction_intervals = 0.95;
options.return_diagnostics = true;
options.return_residuals = true;
options.return_robustness_weights = true;
options.cv_fractions = {0.3, 0.5, 0.7};
options.cv_method = "kfold";
options.cv_k = 5;
options.auto_converge = 1e-4;
options.parallel = true;

fastlowess::Lowess model(options);
auto result = model.fit(x, y);

// Result structure:
result.x_vector(),
result.y_vector(),
result.standard_errors(),
result.confidence_lower(),
result.confidence_upper(),
result.prediction_lower(),
result.prediction_upper(),
result.residuals(),
result.robustness_weights(),
result.diagnostics(),
result.iterations_used(),
result.fraction_used(),
result.cv_scores()
```

---

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](https://github.com/thisisamirv/lowess-project/blob/main/CONTRIBUTING.md) file for more information.

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
