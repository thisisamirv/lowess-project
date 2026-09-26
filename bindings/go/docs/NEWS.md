---
title: "News"
weight: 100
---

<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md).

## [Unreleased]

### Added

* Added grouped `Outputs []string` and `CV *CVOptions` options, plus grouped prediction outputs; legacy flat fields remain accepted during migration.

### Changed

* Represent unavailable diagnostic metrics as `nil` optional values instead of `NaN` sentinels.

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
