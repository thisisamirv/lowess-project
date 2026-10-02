\page news News

<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md).

## [Unreleased]

### Added

* Enabled the optional wgpu DirectX 12 backend for GPU-enabled Windows builds. DXC is loaded dynamically with an FXC fallback, avoiding eager imports of `dxcompiler.dll` and `dxil.dll`.
* Added GLES for Android GPU builds with a target-scoped `wgpu` feature; Windows continues to omit the GLES-only loader imports.

### Changed

* Breaking change: replaced flat interval levels with grouped `intervals.confidence`, `intervals.prediction`, and `intervals.bootstrap` on Batch, Streaming, Online, and Predict options. Use the optional outer `seed` for CV and fit-time bootstrap, or on Predict options for prediction-time bootstrap; zero is a valid seed.
* Breaking change: replaced flat `return_*`/`cv_*` fields with grouped `outputs` and nested `cv` options; prediction outputs are grouped as well.
* Requires C++17 for the public wrapper's use of `std::optional`.
* Represent unavailable diagnostics as empty `std::optional<double>` values instead of `NaN` sentinels.
* Changed C ABI collection lengths from `unsigned long` to `size_t` so 64-bit Windows can represent full vector lengths; rebuild C/C++ binaries against the updated header and library.

### Fixed

* Fixed global OLS fits treating predictor values with a large offset as degenerate; translated inputs now retain their fitted slope.
* Fixed fraction-1 global fits ignoring custom weights, including when Batch sorts observations by x.
* Fixed Batch `missing = "drop"` accepting custom weights with a length different from the original input; weights are validated before rows are dropped.
* Fixed standard errors for global weighted fits to account for observation weights and weighted prediction leverage.
* Fixed local Batch and retained-model prediction standard errors ignoring `custom_weights`; local SE moments now include the per-observation case weights.
* Fixed global fits with all-zero custom weights to honor the configured zero-weight fallback policy.
* Fixed fraction-1 global fits ignoring configured robustness iterations; they now reweight observations and report iterations used.
* Fixed Batch cross-validation candidate fits ignoring `custom_weights`; K-fold CV now rejects more folds than observations instead of returning zero scores.
* Fixed Streaming and Online accepting invalid `auto_converge` tolerances; Online also rejects invalid explicit `delta` values while retaining NaN as its default sentinel.
* Fixed GPU Batch fits misaligning fitted values with unsorted inputs; results now preserve input and requested sorted order.
* Fixed GPU Batch silently ignoring `custom_weights`; GPU fit and CV candidate kernels now apply them directly.
* Reduced GPU adapter buffer requirements from 30 storage/32 total buffer bindings to 7 storage/8 total per shader stage by using per-compute-pipeline resource layouts.
* Fixed Online incremental mode accepting positive `delta` and `auto_converge` settings it cannot use; unsupported combinations now error.
* Fixed grouped intervals discarding distinct confidence and prediction levels; each requested coverage is now applied independently.
* Fixed empty or failed results producing invalid vector ranges and leaking owned error/retained-model state; result accessors now return empty vectors safely and RAII releases unclaimed resources.
* Fixed unknown output names being silently ignored; Batch, Streaming, Online, and Predict now reject names not supported by that operation.
* Fixed model initialization failures being hidden behind null-model errors; C++ constructors now throw the original configuration error.
* Fixed `StreamingOptions` silently accepting Batch-only settings and negative iteration counts producing misleading errors; unsupported options and negative counts are now rejected clearly.
* Fixed k-fold counts below two being silently changed to two; active k-fold CV now rejects invalid counts, while LOOCV continues to ignore `cv.k`.
* Fixed GPU installation interpolating paths into shell commands; paths with shell-special characters are handled safely.
* Fixed the GPU installer selecting x86_64/Linux artifacts for unsupported targets; unsupported platform/architecture pairs now fail before download.

* Fixed the default `boundary_policy` (`"extend"`) letting synthetic boundary points bias the shared robustness scale estimate used to reweight every point, compounding across robustness iterations.
* Improved agreement with R/Cleveland on sparse, asymmetric, and high-iteration fits by aligning robustness stopping, local-linear degeneracy handling, neighborhood traversal, delta interpolation, and weighted accumulation.
* Fixed zero-radius neighborhoods dropping tied observations or accumulating normalized weights in a different order under robust fits.
* Fixed Gaussian fits clipping the unbounded kernel to the neighbor window and flooring far-tail weights; all observations now contribute under the standard Gaussian formula.

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

### Changed

* Breaking change: replaced exception-based error handling with `Expected<T>` results for core operations; callers must check and handle returned errors.

## 0.99.9

### Added

* Made the C++ library available on conda-forge as `libfastlowess`.

## 0.99.8

* Initial implementation of the C++ library.
