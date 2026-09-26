<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md).

## [Unreleased]

### Added

* Added grouped `outputs` and nested `cv` constructor options, plus grouped prediction outputs, while preserving legacy keyword arguments.
* Added an "Alternative Software" guide comparing `fastlowess` with `statsmodels.lowess()`.

## 4.1.0

### Added

* Added musl release binaries for Python.
* Added `retain_model` and `LowessResult.predict(new_x, ...)` for out-of-sample prediction.
* Added `return_derivative` to `Lowess`, `StreamingLowess`, and `OnlineLowess`, exposing each point's local slope.
* Added standard errors and confidence/prediction intervals to `StreamingLowess` and `OnlineLowess`; online intervals require `update_mode="full"`.

### Fixed

* Changed the `OnlineLowess` default `iterations` from `3` to `0`, matching non-robust incremental updates; robustness iterations require `update_mode="full"`.
* Fixed `cv_seed` silently accepting negative values by rejecting them before casting.

## 4.0.0

### Added

* Added a `return_sorted` option to `Lowess`.
* Added a `missing` option to `Lowess`, `StreamingLowess`, and `OnlineLowess` to control non-finite input handling.
* Added GPU wheels for Linux AArch64 and Windows ARM64, and support for installing from a locally built GPU wheel.

### Changed

* Breaking change: removed `return_diagnostics`, `return_residuals`, and `parallel` from the `OnlineLowess` constructor; these options had no effect. `Lowess` and `StreamingLowess` are unaffected.
* Improved the Python API documentation, including a dedicated GPU backend guide with hardware requirements and performance considerations.

### Fixed

* Fixed `install_gpu()` to locate the published GPU wheel.
* Corrected type stubs to match the runtime defaults for streaming and online constructors.

## 3.2.1

### Added

* Added Windows ARM64 wheels for Python.

### Changed

* Clarified that results are returned in input order, even though the algorithm sorts internally.

## 3.2.0

### Changed

* Expanded and reorganized the Python documentation with setup guidance, parameter explanations, and standardized examples.

### Fixed

* Corrected the runtime defaults for `StreamingLowess.fraction` and for `OnlineLowess.fraction`, `window_capacity`, and `update_mode` to match the batch constructor and Rust core.
* Fixed the API reference page rendering empty by correcting its links to the renamed Python API pages.
* Corrected the Handling Outliers example to use a fraction that downweights the injected outlier.

## 3.1.0

### Changed

* Migrated Python documentation from MkDocs to Sphinx, with executable examples, and updated the README with package-specific guidance.
* Breaking change: arguments after the initial positional parameters are now keyword-only for `Lowess`, `StreamingLowess`, and `OnlineLowess`.

## 3.0.0

### Added

* Added an opt-in GPU backend through the `gpu` Cargo feature and `fastlowess.install_gpu()` to download a prebuilt GPU wheel.
* Added cross-reference links in the API documentation to the corresponding user guides.

### Changed

* Breaking change: renamed `OnlineOutput` properties `smoothed` and `std_error` to `y` and `standard_error`.
* Organized Streaming and Online API documentation and tutorials into dedicated user-guide pages.

## 2.0.0

### Added

* Added `OnlineOutput` for `OnlineLowess.add_point()`, exposing the smoothed value, standard error, residual, robustness weight, and iterations used.
* Added `custom_weights` to `Lowess.fit()` for non-negative per-observation batch weights.

### Changed

* Breaking change: replaced `OnlineLowess.update(x, y)` and array-based `add_points(x, y)` with `add_point(x, y)`, returning the smoothed value as `float | None`.
* Added `return_se` and `cv_seed` options to `SmoothOptions`.

## 1.3.0

### Fixed

* Python fitting methods now accept documented array-like inputs by converting them to NumPy `float64` arrays before calling the native extension.

## 1.2.0

### Changed

* Updated PyO3 from v0.27 to v0.28 and NumPy from v0.27 to v0.28.

## 1.1.2

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 1.1.1

### Fixed

* Fixed GPU configuration and initialization problems, and improved recovery after missing hardware/drivers or earlier GPU execution errors.

## 1.1.0

### Added

* Expanded GPU fitting support to selectable kernels, robustness and scaling methods, boundary policies, automatic convergence, prediction, and cross-validation.

### Fixed

* Fixed the `Extend` boundary policy not being applied and improved numerical precision through coordinate centering.
* Fixed GPU integer overflow, initialization failures, and resource exhaustion.

## 1.0.0

### Added

* Added the `mean` scaling method (Mean Absolute Deviation).
* Added asynchronous batch processing.

### Changed

* Released the Python GIL during fitting so other Python threads can run while LOWESS computations execute.

## 0.99.9

### Changed

* Introduced class-based builders for streaming and online processing.

## 0.99.8

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.99.7

### Changed

* Switched to the CPython Stable ABI (`abi3`) for broader Python-version compatibility.

## 0.99.6

### Fixed

* Fixed formatting and links in the shared project README.

## 0.99.5

### Changed

* Reduced the published package size by excluding development-only files.

## 0.99.2

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.99.1

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.99.0

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.7.0

### Added

* Added `NoBoundary` handling and `MAD`/`MAR` scaling methods to the underlying LOWESS core.

### Changed

* Improved fitting and cross-validation performance through SIMD accumulation, reusable buffers, and optimized window and scale calculations.

## 0.6.0

### Added

* Added a seed option for reproducible k-fold cross-validation in the underlying LOWESS core.

### Fixed

* Fixed Batch and Streaming adapter conversion behavior in the LOWESS core.

## 0.5.3

### Changed

* Improved fitting and cross-validation performance by optimizing sorting, window operations, robust scale estimation, regression, and delta interpolation.

## 0.4.0

### Added

* Added support for the parallel-fitting and cross-validation improvements in fastLowess 0.4.0.

### Changed

* Changed the license from AGPL-3.0-or-later to dual MIT OR Apache-2.0.

## 0.3.0

### Fixed

* Fixed the Python API not exposing the `parallel` argument.

## 0.2.0

### Added

* Added support for the fastLowess 0.2.0 features.

### Changed

* Breaking change: changed the package import name from `fastLowess` to `fastlowess`.

## 0.1.0

* Initial Python binding, with Python 3.14 support.
