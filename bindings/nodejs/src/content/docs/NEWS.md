---
title: News
---
<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md).

## [Unreleased]

### Added

* Enabled the optional wgpu DirectX 12 backend for GPU-enabled Windows builds. DXC is loaded dynamically with an FXC fallback, avoiding eager imports of `dxcompiler.dll` and `dxil.dll`.
* Added GLES for Android GPU builds with a target-scoped `wgpu` feature; Windows continues to omit the GLES-only loader imports.
* Added grouped `outputs` arrays and nested `cv` options for batch, streaming, online, and prediction configuration while preserving legacy fields.
* Added a grouped `intervals` option with residual-bootstrap intervals for `Lowess`, `StreamingLowess`, full-update `OnlineLowess`, and `LowessResult.predict()`.

### Changed

* Breaking change: replaced flat interval options and `predict()`'s interval levels with `intervals: { confidence, prediction, bootstrap }`, and replaced flat CV options and `cv.seed` with `cv: { method, k, fractions }` plus one outer `seed` shared by CV and bootstrap. `predict()` has its own `seed`.

### Fixed

* Fixed global OLS fits treating predictor values with a large offset as degenerate; translated inputs now retain their fitted slope.
* Fixed fraction-1 global fits ignoring custom weights, including when Batch sorts observations by x.
* Fixed Batch `missing = "drop"` accepting custom weights with a length different from the original input; weights are validated before rows are dropped.
* Fixed standard errors for global weighted fits to account for observation weights and weighted prediction leverage.
* Fixed global fits with all-zero custom weights to honor the configured zero-weight fallback policy.
* Fixed fraction-1 global fits ignoring configured robustness iterations; they now reweight observations and report iterations used.
* Fixed Batch cross-validation candidate fits ignoring `custom_weights`; K-fold CV now rejects more folds than observations instead of returning zero scores.
* Fixed Streaming and Online accepting invalid `auto_converge` tolerances; Online also rejects invalid explicit `delta` values while retaining NaN as its default sentinel.
* Fixed GPU Batch fits misaligning fitted values with unsorted inputs; results now preserve input and requested sorted order.
* Fixed GPU Batch silently ignoring `custom_weights`; GPU fit and CV candidate kernels now apply them directly.
* Reduced GPU adapter buffer requirements from 30 storage/32 total buffer bindings to 7 storage/8 total per shader stage by using per-compute-pipeline resource layouts.
* Fixed Online incremental mode accepting positive `delta` and `auto_converge` settings it cannot use; unsupported combinations now error.
* Fixed grouped intervals silently collapsing different confidence and prediction levels; mismatches now error during build.

* Fixed `OnlineLowess` and `LowessResult.predict()` ignoring the `outputs` option.
* Fixed the default `boundary_policy` (`"extend"`) letting synthetic boundary points bias the shared robustness scale estimate used to reweight every point, compounding across robustness iterations.
* Improved agreement with R/Cleveland on sparse, asymmetric, and high-iteration fits by aligning robustness stopping, local-linear degeneracy handling, neighborhood traversal, delta interpolation, and weighted accumulation.
* Fixed zero-radius neighborhoods dropping tied observations or accumulating normalized weights in a different order under robust fits.
* Fixed Gaussian fits clipping the unbounded kernel to the neighbor window and flooring far-tail weights; all observations now contribute under the standard Gaussian formula.

## 4.1.0

### Added

* Added `retain_model` and `LowessResult.predict(newX, options)` for prediction.
* Added `return_derivative` to `SmoothOptions`, `StreamingOptions`, and `OnlineOptions`.
* Added standard errors and confidence/prediction intervals to `StreamingOptions` and `OnlineOptions`; online intervals require `update_mode: "full"`.

### Fixed

* Changed the `OnlineLowess` default `iterations` from `3` to `0`, matching non-robust incremental updates; robustness iterations require `update_mode: "full"`.
* Fixed `cv_seed` silently accepting negative values by rejecting them before casting.

## 4.0.0

### Added

* Added `return_sorted` to `SmoothOptions` and `missing` to `SmoothOptions`, `StreamingSmoothOptions`, and `OnlineSmoothOptions`.
* Added `fastlowess.installGpu()` to download a prebuilt GPU addon; the installer also accepts a path to a local GPU artifact.
* Expanded prebuilt GPU binaries to cover all nine platforms supported by the npm package.

### Changed

* Breaking change: split `SmoothOptions` into `SmoothOptions` for batch fits, `StreamingSmoothOptions`, and `OnlineSmoothOptions`; batch-only options now produce a TypeScript error instead of being silently ignored by streaming and online fits.
* Breaking change: removed ineffective `return_diagnostics`, `return_residuals`, and `parallel` options from `OnlineOptions`.
* Breaking change: `StreamingOptions.overlap` now defaults dynamically to `chunk_size / 10`, clamped to `[1, chunk_size - 10]`, rather than a fixed `500`.
* Improved the Node.js API documentation, including the dynamic overlap default.

### Fixed

* Fixed `npm run build` and `npm run build:debug` to produce the platform-specific native addon files expected by the package loader.
* Fixed `installGpu()` to survive native-addon rebuilds, download the filename expected by the loader, and replace existing files on Windows.

## 3.2.1

### Added

* Added prebuilt targets for Linux ARM64 with musl and ARMv7 Linux with hard-float support.

### Changed

* Clarified that results are returned in input order, even though the algorithm sorts internally.

## 3.2.0

### Added

* Added a `VERSION` export so consumers can query the Node.js package version directly.

### Changed

* Reorganized and expanded the Node.js documentation, including a dedicated GPU backend guide with hardware requirements and performance considerations.

### Fixed

* Improved the docs homepage and fixed links in the generated API reference.
* Corrected the Handling Outliers example to use a fraction that downweights the injected outlier.

## 3.1.0

### Changed

* Moved Node.js documentation to GitHub Pages with executable examples and updated the README with package-specific guidance.

### Fixed

* Changed `.build()` configuration failures to return `Status::InvalidArg` instead of `Status::GenericFailure`.

## 3.0.0

### Added

* Added an opt-in GPU backend through the `gpu` Cargo feature and `fastlowess.installGpu()`; using the installed GPU addon requires restarting Node.js.
* Added cross-reference links in the API documentation to the corresponding user guides.

### Changed

* Breaking change: renamed `OnlineOutput` fields `smoothed` and `std_error` to `y` and `standard_error`.
* Organized Streaming and Online API documentation and tutorials into dedicated user-guide pages.

## 2.0.0

### Added

* Added `OnlineOutput` for `OnlineLowess.add_point()`, exposing the smoothed value, standard error, residual, robustness weight, and iterations used.
* Added `return_se` and `cv_seed` options to `SmoothOptions`.
* Added `customWeights` to `fit` and `fit_async` for per-observation batch weights.
* Unknown option keys now throw a `TypeError` listing the valid keys.

### Changed

* Breaking change: renamed public API fields, methods, and options from camelCase to snake_case.
* Breaking change: replaced `OnlineLowess.add_points(x, y)` with `add_point(x, y)`, which processes one point and returns `OnlineOutput | null`.
* Changed `OnlineOptions.window_capacity`'s default from `100` to `1000` and `min_points` from `2` to `3`.
* `OnlineLowess` now forwards all `SmoothOptions` fields to the underlying builder instead of silently ignoring most of them.

## 1.3.0

### Fixed

* Changed configuration errors from `.build()` to return `Status::InvalidArg` rather than a generic runtime failure status.

## 1.2.0

### Fixed

* Fixed reported vulnerabilities in Node.js dependencies.

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

## 0.99.9

### Added

* Made the package available on npm as `fastlowess`.

### Changed

* Introduced class-based builders for streaming and online processing.

## 0.99.8

* Initial implementation of the Node.js binding.
