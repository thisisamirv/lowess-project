<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md).

## [Unreleased]

### Added

* Added `cv_opts()` to configure grouped cross-validation options for `Lowess(cv = ...)`.
* Added an "Alternative Software" vignette comparing `rfastlowess` with `stats::lowess()`.

### Changed

* Breaking change: replaced individual `return_*` arguments with grouped `outputs`, and replaced `Lowess()`'s four `cv_*` arguments with `cv = cv_opts(...)`.
* Represent unavailable diagnostic metrics as R `NA` rather than generic `NaN` values.

## 4.1.0

### Added

* Added `retain_model` and `predict.Lowess()` for out-of-sample prediction.
* Added `return_derivative` to `Lowess()`, `StreamingLowess()`, and `OnlineLowess()`.
* Added standard errors and confidence/prediction intervals to `StreamingLowess()` and `OnlineLowess()`; online intervals require `update_mode = "full"`.

### Fixed

* Changed the `OnlineLowess()` default `iterations` from `3` to `0`, matching non-robust incremental updates; robustness iterations require `update_mode = "full"`.

## 4.0.0

### Added

* Added `return_sorted` to `Lowess()`.
* Added a `missing` option to `Lowess()`, `StreamingLowess()`, and `OnlineLowess()` for non-finite inputs.
* Added support for installing a locally built GPU artifact with `install_gpu()`.

### Changed

* Breaking change: removed `return_diagnostics`, `return_residuals`, and `parallel` from the `OnlineLowess()` constructor; these options had no effect in online mode.
* Breaking change: removed `confidence_intervals` and `prediction_intervals` from `OnlineLowess()` and `StreamingLowess()` constructors because those options were not computed.
* Breaking change: the streaming overlap default now resolves dynamically to `chunk_size / 10`, clamped to `[1, chunk_size - 10]`, instead of a fixed `500`.
* Improved the R API documentation, including the overlap default.

### Fixed

* Fixed the real-time vignette example failing with two data points by aligning its minimum point count with the library's minimum of two.
* Fixed `install_gpu()` replacing a loaded shared library in place; it now installs through a temporary file and atomic rename.

## 3.2.1

### Changed

* Clarified that results are returned in input order, even though the algorithm sorts internally.

## 3.2.0

### Changed

* Expanded and reorganized the R documentation with parameter guidance, examples, and clearer adapter-selection advice.

### Fixed

* Corrected the `OnlineLowess()` defaults to `min_points = 2L` and `update_mode = "incremental"`.
* Corrected the Handling Outliers example to use a fraction that downweights the injected outlier.
* Improved the Online, streaming, merge, robustness, and use-case examples so they show representative output and run successfully.

## 3.1.0

### Changed

* Moved R documentation to GitHub Pages using pkgdown and updated the README with package-specific guidance.
* Lowered the minimum supported R version to 4.4.0.

### Fixed

* Fixed extra positional arguments being accepted outside the initial positional argument; constructors now reject unnamed arguments in later positions.

## 3.0.0

### Added

* Added an opt-in GPU backend through the `gpu` Cargo feature and `install_gpu()` to download a prebuilt GPU library; using it requires restarting R.
* Added S3 generics `fit()`, `process_chunk()`, `finalize()`, and `add_point()`, replacing the previous list-closure API.
* Added cross-reference links in the API documentation to the corresponding user guides.

### Changed

* Breaking change: renamed the `smoothed` and `std_error` fields returned by `OnlineLowess()$add_point()` to `y` and `standard_error`.
* Breaking change: added `...` to `Lowess()`, `StreamingLowess()`, and `OnlineLowess()` so optional arguments must be named.
* Set the minimum supported R version to 4.6.
* Organized Streaming and Online API documentation and tutorials into dedicated user-guide pages.

## 2.0.0

### Added

* Added `custom_weights` to batch fitting for non-negative per-observation weights.

### Changed

* Breaking change: replaced vector-based `OnlineLowess()$add_points(x, y)` with scalar `add_point(x, y)`, which processes one point at a time and returns `NULL` until enough points are available.

## 1.3.0

### Fixed

* Registered the extendr panic hook so Rust panics are reported as R errors instead of crashing the R session.

## 1.2.0

### Added

* Added `print()` and `plot()` examples for `LowessResult` objects.

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
* Added `print()` and `plot()` methods for `LowessResult` objects.

### Changed

* Breaking change: returned `LowessResult` S3 objects instead of raw vectors.

## 0.99.9

### Added

* Made the package available on conda-forge as `r-rfastlowess`.

### Changed

* Introduced class-based builders for streaming and online processing.

## 0.99.8

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.99.7

### Fixed

* Fixed links in the shared project README.

## 0.99.6

### Fixed

* Fixed formatting and links in the shared project README.

## 0.99.5

### Changed

* Reduced the published package size by excluding development-only files.

## 0.99.2

### Added

* Added a pkgdown documentation site and examples for exported functions.
* Added input validation for exported functions.

### Changed

* Breaking change: renamed exported functions to avoid conflicts with base R.

### Fixed

* Fixed package startup messages.

## 0.99.1

### Changed

* Prepared the R package for Bioconductor submission.

## 0.99.0

### Added

* Added `NoBoundary` boundary handling and `MAD`/`MAR` scaling methods from fastLowess v0.4.0.

### Changed

* Changed the license from AGPL-3.0-or-later to dual MIT OR Apache-2.0.

## 0.7.0

### Added

* Added `NoBoundary` handling and `MAD`/`MAR` scaling methods to the underlying LOWESS core.

### Changed

* Improved fitting and cross-validation performance through SIMD accumulation, reusable buffers, and optimized window and scale calculations.

## 0.6.0

### Added

* Added a seed option for reproducible k-fold cross-validation.

### Fixed

* Fixed Batch and Streaming adapter conversion behavior in the LOWESS core.

## 0.5.3

### Changed

* Improved fitting and cross-validation performance by optimizing sorting, window operations, robust scale estimation, regression, and delta interpolation.

## 0.4.0

### Added

* Added support for fastLowess v0.4.0 fitting and cross-validation improvements.

### Changed

* Changed the license from AGPL-3.0-or-later to dual MIT OR Apache-2.0.

## 0.3.0

### Added

* Added installation from R-universe without requiring Rust.

### Changed

* Updated the cross-validation API for fastLowess v0.3.0.

## 0.2.0

### Added

* Added support for fastLowess v0.2.0 fitting improvements.

## 0.1.0

### Added

* Initial R binding for fastLowess.
