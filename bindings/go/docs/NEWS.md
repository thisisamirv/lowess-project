---
title: "News"
weight: 100
---

<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md).

## [Unreleased]

### Added

* Enabled the optional wgpu DirectX 12 backend for GPU-enabled Windows builds. DXC is loaded dynamically with an FXC fallback, avoiding eager imports of `dxcompiler.dll` and `dxil.dll`.
* Added GLES for Android GPU builds with a target-scoped `wgpu` feature; Windows continues to omit the GLES-only loader imports.
* Added grouped `Outputs []string`, `CV *CVOptions`, and `Intervals *IntervalsOptions` for fitting and prediction, including residual-bootstrap intervals for all adapters.

### Changed

* Clarified that Batch `ResidualSD` is `1.4826 * MAD`, while Streaming reports the cumulative sample standard deviation of emitted residuals.
* Breaking change: replaced individual `Return*` output fields with `Outputs: []string{...}` for Batch, Streaming, Online, and prediction options.
* Breaking change: replaced flat interval/CV fields and nested CV seed with grouped options and one outer `Seed *uint64` shared by fit-time CV and bootstrap. `PredictOptions` has its own `Seed` for prediction-time bootstrap.
* Represent unavailable diagnostic metrics as `nil` optional values instead of `NaN` sentinels.

### Fixed

* Fixed CPU Gaussian standard errors for fitted and queried values to use the full unbounded kernel support.
* Made even-sample medians and mean-absolute residual scaling overflow-resistant for large finite values, and centered Batch/Streaming R-squared accumulation to preserve one-ULP response variation at large offsets.
* Made Batch RMSE, MAE, and R-squared reductions scale-safe for large finite values, and preserved bisquare downweighting when the tuned residual scale exceeds the numeric range.
* Reject explicitly empty custom weights instead of treating them as omitted.
* Fixed model finalizers potentially releasing native state during in-flight calls; model receivers are now kept alive until cgo calls return.
* Fixed unknown `Outputs` names being silently ignored and `Fit` silently ignoring extra custom-weight slices; invalid inputs now return errors.
* Fixed k-fold CV silently coercing fold counts below two; active k-fold now rejects them while LOOCV continues to ignore `K`.
* Fixed the GPU installer selecting GNU/Linux artifacts on musl systems and accepting incompatible local archives; local archives now require a matching GPU feature, Go ABI major, platform, and architecture marker.
* Fixed Go `int` option and collection lengths narrowing at the C ABI boundary on 64-bit Windows.
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
* Fixed the default `BoundaryPolicy` (`"extend"`) letting synthetic boundary points bias the shared robustness scale estimate used to reweight every point, compounding across robustness iterations.
* Improved agreement with R/Cleveland on sparse, asymmetric, and high-iteration fits by aligning robustness stopping, local-linear degeneracy handling, neighborhood traversal, delta interpolation, and weighted accumulation.
* Fixed zero-radius neighborhoods dropping tied observations or accumulating normalized weights in a different order under robust fits.
* Fixed Gaussian fits clipping the unbounded kernel to the neighbor window and flooring far-tail weights; all observations now contribute under the standard Gaussian formula.

## 4.1.0

### Added

* Added musl release binaries for Go.
* Added `RetainModel` and `Result.PredictModel.Predict(newX, options)` for prediction.
* Added `ReturnDerivative` to `Options`, `StreamingOptions`, and `OnlineOptions`.
* Added standard errors and confidence/prediction intervals to `StreamingOptions` and `OnlineOptions`; online intervals require `UpdateMode = "full"`.

### Changed

* Breaking change: the Go module import path now includes the required `/v4` suffix; update imports using the old unsuffixed path.

### Fixed

* Changed the `OnlineLowess` default `iterations` from `3` to `0`, matching non-robust incremental updates; robustness iterations require `UpdateMode = "full"`.

## 4.0.0

### Added

* Added `ReturnSorted` to `Options`.
* Added `Missing` to `Options`, `StreamingOptions`, and `OnlineOptions` to control non-finite input handling.
* Added `fastlowess.InstallGPU()` to download a prebuilt GPU library; GPU installers also accept a path to an existing local artifact.
* Added prebuilt GPU artifacts for Linux and Windows ARM64.

### Changed

* Breaking change: `OnlineOptions` no longer embeds `Options`; removed `ReturnDiagnostics`, `ReturnResiduals`, `Parallel`, and `Backend`.
* Breaking change: `StreamingOptions` no longer embeds `Options`; removed the inherited interval, standard-error, sorting, cross-validation, and backend fields.
* Breaking change: `StreamingOptions.Overlap` now defaults dynamically to `chunk_size / 10`, clamped to `[1, chunk_size - 10]`, rather than a fixed `500`.
* Improved the Go API documentation, including the dynamic overlap default.

## 3.2.1

### Added

* Added Linux and Windows ARM64 release binaries for Go.

### Changed

* Clarified that results are returned in input order, even though the algorithm sorts internally.

### Fixed

* Corrected the `OnlineOptions` defaults to `MinPoints = 2` and `UpdateMode = "incremental"`.
* Fixed Windows ARM64 builds to target the host architecture rather than x64.
* Fixed Go module license metadata so pkg.go.dev recognizes the MIT and Apache-2.0 licenses.

## 3.2.0

* Initial release of the Go binding.
