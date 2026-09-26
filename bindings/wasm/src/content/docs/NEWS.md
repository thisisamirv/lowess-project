---
title: News
---
<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md).

## [Unreleased]

### Added

* Added grouped `outputs` arrays and nested `cv` options for batch, streaming, online, and prediction configuration while preserving legacy fields.

## 4.1.0

### Changed

* Raised the minimum supported Rust version to 1.89 for source builds.

* Added `retain_model` and `LowessResult.predict(newX, options)` for prediction.
* Added `return_derivative` to `SmoothOptions`, `StreamingOptions`, and `OnlineOptions`.

### Fixed

* Fixed reported vulnerabilities in WASM dependencies.

### Fixed

* Changed the `OnlineLowess` default `iterations` from `3` to `0`, matching non-robust incremental updates; robustness iterations require `update_mode: "full"`.

## 4.0.0

### Added

* Added `return_sorted` to `SmoothOptions`.
* Added `missing` to `SmoothOptions`, `StreamingSmoothOptions`, and `OnlineSmoothOptions` to control non-finite input handling.

### Changed

* Breaking change: split `SmoothOptions` into batch `SmoothOptions` and separate `StreamingOptions` and `OnlineOptions`; batch-only fields are no longer accepted by streaming or online APIs.
* Breaking change: `StreamingOptions.overlap` now defaults dynamically to `chunk_size / 10`, clamped to `[1, chunk_size - 10]`, instead of a fixed `500`.
* Improved the WASM API documentation, including the dynamic overlap default.

### Fixed

* Corrected the WASM option documentation for overlap, `window_capacity`, and `update_mode` defaults.

## 3.2.1

### Changed

* Clarified that results are returned in input order, even though the algorithm sorts internally.

## 3.2.0

### Changed

* Expanded and reorganized the WASM documentation with setup guidance, parameter information, and standardized examples.

### Fixed

* Corrected documentation figures and mathematical rendering.
* Corrected the Handling Outliers example to use a fraction that downweights the injected outlier.

## 3.1.0

### Fixed

* Fixed the `Extend` boundary policy not being applied and improved numerical precision through coordinate centering.
* Fixed adapter execution errors being silently ignored instead of propagated.

* Moved WASM documentation to GitHub Pages and updated the README with package-specific guidance.

## 3.0.0

### Added

* Added cross-reference links in the API documentation to the corresponding user guides.

### Changed

* Breaking change: renamed `OnlineOutput` getters `smoothed` and `std_error` to `y` and `standard_error`.
* Organized Streaming and Online API documentation and tutorials into dedicated user-guide pages.

### Fixed

* Fixed `OnlineLowess.add_point()` returning `undefined` instead of `null` until the sliding window contains enough points.

## 2.0.0

### Added

* Added `custom_weights` to `LowessOptions` for non-negative per-observation batch weights.

### Changed

* Breaking change: renamed JavaScript-facing option keys and API methods from camelCase to snake_case; option objects must use snake_case keys.
* Breaking change: renamed `OnlineLowess.update(x, y)` to `add_point(x, y)`.

## 1.3.0

### Changed

* Raised the minimum supported Rust version to 1.89 for source builds.

## 1.2.0

### Fixed

* Fixed reported vulnerabilities in WASM dependencies.

## 1.1.2

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 1.1.1

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 1.1.0

### Fixed

* Fixed the `Extend` boundary policy not being applied and improved numerical precision through coordinate centering.
* Fixed adapter execution errors being silently ignored instead of propagated.

## 1.0.0

### Added

* Added the `mean` scaling method (Mean Absolute Deviation).
* Added `init_panic_hook()` for reporting Rust panics as JavaScript errors.

## 0.99.9

### Added

* Made the package available on npm as `fastlowess-wasm`.

### Changed

* Introduced class-based builders for streaming and online processing.

## 0.99.8

* Initial implementation of the WebAssembly binding.
