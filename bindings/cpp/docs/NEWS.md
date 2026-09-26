\page news News

<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md).

## [Unreleased]

### Changed

* Breaking change: replaced flat `return_*`/`cv_*` fields with grouped `outputs` and nested `cv` options; prediction outputs are grouped as well.
* Requires C++17 for the public wrapper's use of `std::optional`.
* Represent unavailable diagnostics as empty `std::optional<double>` values instead of `NaN` sentinels.

## 4.1.0

### Added

* Added musl release binaries for C++.
* Added `retain_model`, `LowessResult::predict_model()`, and prediction RAII types.
* Added `return_derivative` to `LowessOptions` and `OnlineOptions`.
* Added standard errors and confidence/prediction intervals to `StreamingOptions` and `OnlineOptions`; online intervals require `update_mode = "full"`.

### Fixed

* Changed the `OnlineLowess` default `iterations` from `3` to `0`, matching the non-robust incremental update mode; robustness iterations require `update_mode = "full"`.

## 4.0.0

### Added

* Added `return_sorted` to `LowessOptions`.
* Added `missing` to `LowessOptions` and `OnlineOptions` (inherited by `StreamingOptions`) to control non-finite input handling.
* GPU installers now accept a local path to an existing GPU artifact; prebuilt GPU artifacts are available for Linux and Windows ARM64.

### Changed

* Breaking change: removed `return_diagnostics`, `return_residuals`, and `parallel` from `OnlineOptions`; these options were accepted but had no effect in online mode.
* Breaking change: removed `confidence_intervals` and `prediction_intervals` from `OnlineOptions`; `StreamingOptions` no longer inherits these options.
* Breaking change: removed the unused `custom_weights` field from `OnlineOptions`.
* Breaking change: `StreamingOptions::overlap` now defaults dynamically to `chunk_size / 10` instead of a fixed `500`.
* Improved the C++ API documentation.

## 3.2.1

### Added

* Added Linux, Windows, and macOS ARM64 release binaries for C++.

### Changed

* Clarified that `LowessResult.x` is returned in input order, even though the algorithm sorts internally.

## 3.2.0

### Added

* Made the C++ library available through Spack as `fastlowess-cpp`.

### Changed

* Expanded and reorganized the C++ documentation, including a dedicated GPU backend guide with hardware requirements and performance considerations.

### Fixed

* Corrected the `OnlineOptions` defaults to use `min_points = 2` and `update_mode = "incremental"`.
* Corrected C++ documentation examples and rendering, including the Handling Outliers and Detecting Outliers examples.

## 3.1.0

### Changed

* Moved C++ documentation to GitHub Pages at <https://thisisamirv.github.io/lowess-project/cpp/> and updated the README with package-specific guidance.
* Breaking change: removed the legacy snake_case compatibility layer; the public C++ method API uses camelBack names.
* Added CMake packaging guidance for Windows installation and downstream `find_package(fastlowess CONFIG REQUIRED)` use.

## 3.0.0

### Added

* Added an opt-in GPU backend through the `gpu` Cargo feature and a `backend` option on `Lowess`.
* Added cross-reference links in the C++ API documentation to the corresponding user guides.

### Changed

* Breaking change: renamed `OnlineOutput::smoothed()` and `OnlineOutput::std_error()` to `y()` and `standard_error()`.
* Organized Streaming and Online API documentation and tutorials into dedicated user-guide pages.

## 2.0.0

### Added

* Added `custom_weights` to `LowessOptions` and a `Lowess::fit()` overload accepting per-observation weights for batch fits.

### Changed

* Breaking change: renamed public C++ methods and options from camelCase to snake_case.
* Breaking change: replaced `OnlineLowess::add_points(x, y)` with `add_point(x, y)`, which processes one point and returns its smoothed value or `std::nullopt` until enough points are available.

## 1.3.0

### Added

* Added CMake packaging documentation for Windows installation, `find_package`, and build-tree package discovery.

### Changed

* Updated the public C++ method API to use camelBack names and removed the legacy snake_case compatibility layer.

## 1.2.0

### Fixed

* Fixed a memory leak in `OnlineLowess`.

## 1.1.2

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 1.1.1

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 1.1.0

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 1.0.0

### Added

* Added the `mean` scaling method (Mean Absolute Deviation).

### Changed

* Breaking change: replaced exception-based error handling with `Expected<T>` results for core operations; callers must check and handle returned errors.

## 0.99.9

### Added

* Made the C++ library available on conda-forge as `libfastlowess`.

## 0.99.8

* Initial implementation of the C++ library.
