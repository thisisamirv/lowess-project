<!-- markdownlint-disable MD024 MD033 -->
# LOWESS Project

<p align="center">
  <a href="https://crates.io/crates/lowess"><img src="https://img.shields.io/badge/lowess-000000?logo=rust&logoColor=white" alt="lowess"></a>
  <a href="https://crates.io/crates/fastLowess"><img src="https://img.shields.io/badge/fastLowess-000000?logo=rust&logoColor=white" alt="fastLowess"></a>
  <a href="https://pypi.org/project/fastlowess/"><img src="https://img.shields.io/badge/PyPI-3775A9?logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://thisisamirv.r-universe.dev/rfastlowess"><img src="https://img.shields.io/badge/R--universe-276DC3?logo=r&logoColor=white" alt="R-universe"></a>
  <a href="https://www.npmjs.com/package/fastlowess"><img src="https://img.shields.io/badge/npm-CB3837?logo=npm&logoColor=white" alt="npm"></a>
  <a href="https://juliahub.com/ui/Packages/General/FastLOWESS"><img src="https://img.shields.io/badge/Julia-9558B2?logo=julia&logoColor=white" alt="Julia"></a>
  <a href="https://www.npmjs.com/package/fastlowess-wasm"><img src="https://img.shields.io/badge/WASM-654FF0?logo=webassembly&logoColor=white" alt="WASM"></a>
  <a href="https://github.com/thisisamirv/lowess-project/releases/latest"><img src="https://img.shields.io/badge/C++-00599C?logo=cplusplus&logoColor=white" alt="C++"></a>
  <br>
  <a href="https://anaconda.org/conda-forge/fastlowess"><img src="https://img.shields.io/badge/fastlowess_(Python)-44A833?logo=anaconda&logoColor=white" alt="fastlowess (Python)"></a>
  <a href="https://anaconda.org/conda-forge/libfastlowess"><img src="https://img.shields.io/badge/libfastlowess_(C++)-44A833?logo=anaconda&logoColor=white" alt="libfastlowess (C++)"></a>
  <a href="https://anaconda.org/conda-forge/r-rfastlowess"><img src="https://img.shields.io/badge/rfastlowess_(R)-44A833?logo=anaconda&logoColor=white" alt="rfastlowess (R)"></a>
  <br>
  <a href="https://github.com/ropensci/software-review/issues/769"><img src="https://badges.ropensci.org/769_status.svg" alt="Status at rOpenSci Software Peer Review"></a>
  <br>
  <a href="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-rust.yml"><img src="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-rust.yml/badge.svg" alt="CI - Rust"></a>
  <a href="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-python.yml"><img src="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-python.yml/badge.svg" alt="CI - Python"></a>
  <a href="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-r.yml"><img src="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-r.yml/badge.svg" alt="CI - R"></a>
  <a href="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-julia.yml"><img src="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-julia.yml/badge.svg" alt="CI - Julia"></a>
  <a href="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-nodejs.yml"><img src="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-nodejs.yml/badge.svg" alt="CI - Node.js"></a>
  <a href="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-wasm.yml"><img src="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-wasm.yml/badge.svg" alt="CI - WASM"></a>
  <a href="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-cpp.yml"><img src="https://github.com/thisisamirv/lowess-project/actions/workflows/ci-cpp.yml/badge.svg" alt="CI - C++"></a>
  <br>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/thisisamirv/lowess-project/main/dev/logo.png" alt="One LOWESS to Rule Them All" width="400">
  <br>
  <em>One LOWESS to Rule Them All</em>
</p>

The fastest, most robust, and most feature-complete language-agnostic LOWESS (Locally Weighted Scatterplot Smoothing) implementation for **Rust**, **Python**, **R**, **Julia**, **JavaScript**, **C++**, and **WebAssembly**.

> The `lowess-project` contains a complete ecosystem for LOWESS smoothing:
>
> - **[`lowess`](https://crates.io/crates/lowess)** - Core single-threaded Rust implementation with `no_std` support
> - **[`fastLowess`](https://crates.io/crates/fastLowess)** - Parallel CPU and GPU-accelerated Rust wrapper with ndarray integration  
> - **[`R bindings`](https://thisisamirv.r-universe.dev/rfastlowess)** - extendr-based R binding
> - **[`Python bindings`](https://pypi.org/project/fastlowess/)** - PyO3-based Python binding
> - **[`Julia bindings`](https://juliahub.com/ui/Packages/General/FastLOWESS)** - Native Julia binding with C FFI
> - **[`JavaScript bindings`](https://www.npmjs.com/package/fastlowess)** - Node.js binding
> - **[`WebAssembly bindings`](https://www.npmjs.com/package/fastlowess-wasm)** - WASM binding
> - **[`C++ bindings`](https://github.com/thisisamirv/lowess-project/releases/latest)** - Native C++ binding with CMake integration

---

## Installation & Documentation

> Currently available for R, Python, Rust, Julia, Node.js, WebAssembly, and C++. See the documentation for your binding/crate below for installation instructions.
>
> | Binding / Crate | Documentation |
> | --- | --- |
> | `lowess` (Rust) | [docs.rs/lowess](https://docs.rs/lowess) |
> | `fastLowess` (Rust) | [docs.rs/fastLowess](https://docs.rs/fastLowess) |
> | Python | [lowess.readthedocs.io](https://lowess.readthedocs.io/) |
> | R | [thisisamirv.github.io/lowess-project/r](https://thisisamirv.github.io/lowess-project/r/) |
> | Julia | [thisisamirv.github.io/lowess-project/julia](https://thisisamirv.github.io/lowess-project/julia/) |
> | Node.js | [thisisamirv.github.io/lowess-project/nodejs](https://thisisamirv.github.io/lowess-project/nodejs/) |
> | WebAssembly | [thisisamirv.github.io/lowess-project/wasm](https://thisisamirv.github.io/lowess-project/wasm/) |
> | C++ | [thisisamirv.github.io/lowess-project/cpp](https://thisisamirv.github.io/lowess-project/cpp/) |

---

## GPU Backend

GPU acceleration (`wgpu`: Vulkan/Metal/DX12) is also supported for high-throughput batch smoothing. See the GPU Backend page in the documentation for your binding/crate for details.

## LOESS vs. LOWESS

| Feature | LOESS | LOWESS (This Crate) |
| --- | --- | --- |
| **Polynomial Degree** | Linear, Quadratic, Cubic, Quartic | Linear (Degree 1) |
| **Dimensions** | Multivariate (n-D support) | Univariate (1-D only) |
| **Flexibility** | High (Distance metrics) | Standard |
| **Complexity** | Higher (Matrix inversion) | Lower (Weighted average/slope) |

> **Note:** For a **LOESS** implementation, use [`loess-project`](https://github.com/thisisamirv/loess-project).

---

## Why this package?

### Speed

The `lowess` project beats the competition in terms of speed, whether in single-threaded or multi-threaded parallel execution. It is on average **200-327x faster** than Python's `statsmodels.lowess` and **2-3x faster** than R's `lowess`.

For more details on the performance comparison, see the Benchmarks page in the documentation for your binding/crate.

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
