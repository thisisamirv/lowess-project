<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md).

## [Unreleased]

### Added

* Added grouped `outputs` and `cv` keywords for `Lowess`, grouped outputs for Streaming and Online constructors, and retained-model prediction.
* Added an "Alternative Software" guide comparing `FastLOWESS.jl` with the general-purpose LOESS implementation in `Loess.jl`.

### Changed

* Represent unavailable diagnostic metrics as `nothing` instead of `NaN` sentinels.

## 4.1.0

### Added

* Added musl release binaries for Julia.
* Added `retain_model` and `predict(model, new_x; kwargs...)` for out-of-sample prediction.
* Added `return_derivative` to `Lowess`, `StreamingLowess`, and `OnlineLowess`.
* Added standard errors and confidence/prediction intervals to `StreamingLowess` and `OnlineLowess`; online intervals require `update_mode = "full"`.

### Fixed

* Changed the `OnlineLowess` default `iterations` from `3` to `0`, matching non-robust incremental updates; robustness iterations require `update_mode = "full"`.

## 4.0.0

### Added

* Added a `return_sorted` option to `Lowess`.
* Added a `missing` option to `Lowess`, `StreamingLowess`, and `OnlineLowess` to control non-finite input handling.
* Added GPU installer support for a local prebuilt artifact and a prebuilt Linux AArch64 GPU library.

### Changed

* Breaking change: removed `return_diagnostics`, `return_residuals`, and `parallel` from `OnlineLowess`; these options were accepted but had no effect in online mode.
* Breaking change: `StreamingLowess`'s `overlap` default now resolves dynamically to `chunk_size / 10` instead of a fixed `500`.
* Improved the Julia API documentation, including the dynamic overlap default.

### Fixed

* Clarified that installing the GPU library requires restarting Julia before the GPU backend can be used.

## 3.2.1

### Changed

* Clarified that results are returned in input order, even though the algorithm sorts internally.

### Fixed

* Corrected `OnlineLowess` defaults to `min_points = 2` and `update_mode = "incremental"`.
* Fixed Windows local library discovery to resolve the native library at runtime instead of reusing a stale precompiled path.

## 3.2.0

### Changed

* Expanded and reorganized the Julia documentation with setup guidance, parameter explanations, and standardized examples.

### Fixed

* Corrected the Handling Outliers example to use a fraction that downweights the injected outlier.
* Shortened confidence-interval and standard-error examples to display the first five values instead of all 100.

## 3.1.0

### Changed

* Moved Julia documentation to GitHub Pages with executable Documenter.jl examples and updated the README with package-specific guidance.

## 3.0.0

### Added

* Added an opt-in GPU backend through the `gpu` Cargo feature and a `backend` option on `Lowess`; `install_gpu()` downloads a prebuilt library.
* Added cross-reference links in the Julia API documentation to the corresponding user guides.

### Changed

* Breaking change: renamed `OnlineOutput` fields `smoothed` and `std_error` to `y` and `standard_error`.
* Organized Streaming and Online API documentation and tutorials into dedicated user-guide pages.

### Fixed

* Fixed `LowessResult.iterations_used` returning the raw FFI sentinel `-1` instead of `nothing` when robustness iterations are not applicable.

## 2.0.0

### Added

* Added the `custom_weights` keyword to batch `fit` for non-negative per-observation weights.

### Changed

* Breaking change: replaced `add_points(online, x, y)` with `add_point(online, x, y)`, which processes one point and returns its smoothed value or `nothing` until enough points are available.

## 1.3.0

### Fixed

* Fixed Windows local Julia runs to load the native library at runtime, avoiding stale library paths from precompiled packages.

## 1.2.0

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 1.1.2

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 1.1.1

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 1.1.0

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 1.0.0

### Added

* Added the `mean` scaling method (Mean Absolute Deviation).

### Fixed

* Caught Rust panics at the FFI boundary and reported them as Julia errors rather than unwinding across the language boundary.

## 0.99.9

### Added

* Registered the package in JuliaRegistries.

### Changed

* Introduced class-based builders for streaming and online processing.

## 0.99.8

* Initial implementation of `FastLOWESS.jl`.
