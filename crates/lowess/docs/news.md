<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md).

## \[Unreleased\]

### Added

* Added `LowessBuilder::outputs(names)` as a grouped replacement for the individual output toggles. Unknown names are collected and reported together by `.build()`.
* Added grouped cross-validation configuration through `CVBuilder` and `.cv(...)`.
* Added `PredictBuilder::outputs(names)` to group prediction outputs such as standard errors and derivatives.

### Changed

* Marked `WeightFunction` as non-exhaustive so future variants do not break downstream exhaustive matches.

### Fixed

* Improved robust fitting for sparse and outlier-heavy data, including neighborhoods with effectively zero robustness weights.
* Improved numerical stability and agreement with Cleveland's LOWESS reference for near-degenerate, asymmetric, and high-iteration fits.

## 4.1.0

### Added

* Added standard errors and confidence/prediction intervals to streaming and online fits; online interval outputs require `update_mode("full")`.
* Added local derivatives to batch, streaming, and online results.
* Added out-of-sample prediction for batch fits, with configurable standard errors, intervals, derivatives, and extrapolation.

### Fixed

* Changed the `OnlineLowess` default `iterations` from `3` to `0`, matching the default `update_mode = "incremental"` non-robust single-point fit; robustness iterations now require `update_mode = "full"`.
* Fixed standard errors collapsing to zero for observations with zero robustness weight, and improved confidence-interval calibration for local-linear and global OLS fits.
* Fixed `OnlineLowess`'s default `"incremental"` mode silently ignoring robustness iterations: `iterations > 0` is now rejected at `.build()` (`RobustnessIterationsRequireFullUpdateMode`) unless `update_mode("full")` is set, the Online `iterations` default is now `0`, and `iterations_used` is reported even without auto-convergence.
* Fixed seeded k-fold cross-validation selecting incorrect fractions for shuffled validation folds.
* Fixed local and global linear fits losing their slope on small-scale predictor data.
* Fixed k-fold cross-validation scores to pool prediction errors before calculating RMSE, consistent with leave-one-out cross-validation.

## 4.0.0

### Added

* Added `.return_sorted()` to the batch builder, to return results sorted ascending by `x` instead of input order. Default `false`.
* Added a `missing` option (`"error"` default, or `"drop"`) controlling non-finite (NaN/Inf) `x`/`y` handling: Batch/Streaming drop non-finite rows (and matching `custom_weights`); Online skips non-finite points, returning `Ok(None)`. Length mismatches always error.

### Changed

* Breaking change: `Streaming::convert()` no longer resolves `overlap` to a flat `500` when unset; it now resolves dynamically to `chunk_size / 10` (clamped to `[1, chunk_size - 10]`). This affects callers relying on the previous flat default with a customized `chunk_size`.
* Improved API documentation for the lowess crate significantly.

## 3.2.1

### Fixed

* Clarified `LowessResult.x` documentation: results are returned in input order, not sorted order, even though the algorithm sorts internally.

## 3.2.0

### Changed

* Standardized the lowess crate documentation: consolidated README setup content and parameter guidance, renamed the batch-adapter heading, replaced flowcharts with decision tables, standardized API examples and page structure, and converted superscripts to ASCII.
* Removed the GPU acceleration section from the API docs because the lowess crate has no GPU feature.
* Updated `Diagnostics` display output to use `R2` instead of the Unicode superscript form.

### Fixed

* Fixed inline/display LaTeX math rendering as literal text on docs.rs; added a `katex-header.html` that renders it client-side with KaTeX.
* Fixed rustdoc cross-reference links in the lowess crate by replacing unresolved relative links with intra-doc links.
* Fixed the Handling Outliers quickstart example in the lowess crate: increased `fraction` from `0.5` to `0.7` because the six-point example otherwise fit the injected outlier exactly instead of downweighting it.
* Capped the lowess crate Detecting Outliers example output at five lines.

## 3.1.0

### Changed

* Moved crate documentation from ReadTheDocs to <https://docs.rs/lowess>.
* Updated the lowess crate README to be package-specific instead of using the generic shared README.

## 3.0.0

### Added

* Added `See: ...` cross-reference links after option headings in the lowess crate API docs, pointing to the corresponding user guide.

### Fixed

* Fixed `docs/api/rust.md` showing Rust enum variants instead of the string option values accepted by the API.

### Changed

* Breaking change: Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`, matching `LowessResult`.
* Split Streaming/Online content from the lowess crate API reference into dedicated pages, moved tutorials into the user-guide use-cases section, and standardized API examples with expected output.

## 2.0.0

### Added

* Added `iterations_used: Option<usize>` field to `OnlineOutput<T>`, reporting the number of robustness iterations performed when `UpdateMode::Full` is active. Returns `Some(0)` for the degenerate two-point linear fit and `None` when `UpdateMode::Incremental` is used.
* Added `ParseErrors(Vec<LowessError>)` variant to `LowessError`, which collects all string-parse failures that accumulate in the builder and reports them together when `build()` is called.
* Added `"take_first"` and `"take_last"` as accepted string aliases for `MergeStrategy::TakeFirst` and `MergeStrategy::TakeLast`.
* Added `"resmooth"` as an accepted string alias for `UpdateMode::Full` and `"single"` as an alias for `UpdateMode::Incremental`, aligning string-parse behaviour with the `loess-rs` crate.
* Added `custom_weights(Vec<T>)` builder method on `LowessBuilder` (Batch adapter only). Accepts a vector of non-negative per-observation weights that are multiplied into the distance and robustness weights before each local regression, allowing known-bad points to be suppressed (`0.0`) or high-quality measurements to be emphasised.

### Changed

* Added `Lowess<T>`, `StreamingLowess<T>`, and `OnlineLowess<T>` type aliases as the primary user-facing constructors (e.g. `StreamingLowess::new().chunk_size(50).build()`). Mode-specific builder methods (`chunk_size`, `overlap`, `window_capacity`, `min_points`, `update_mode`) are now called directly on the type alias rather than after `.adapter()`.
* Breaking change: Made `BatchLowessBuilder`, `StreamingLowessBuilder`, and `OnlineLowessBuilder` internal-only: all public setter methods have been removed from these types. All smoothing configuration now flows through `LowessBuilder<T, Mode>` (exposed via the type aliases above). Callers that used setters on adapter builders must update their code.
* Breaking change: Changed all enum-typed builder methods to accept strings instead: `weight_function`, `robustness_method`, `scaling_method`, `boundary_policy`, `zero_weight_fallback`, `merge_strategy`, and `update_mode` now take `impl IntoEnum<T>` (accepting both enum variants and strings such as `.weight_function("tricube")`) rather than requiring enum variants to be imported. Callers passing enum variants must use strings instead.
* Breaking change: Replaced the `cross_validate(CVConfig)` builder method (which required importing `KFold` or `LOOCV` types) with a string-based cross-validation API: `.cv_method("kfold")` / `.cv_method("loocv")`, `.cv_k(n)`, `.cv_fractions(vec![...])`, and `.cv_seed(n)`. `KFold` and `LOOCV` are no longer exported from the prelude. Callers using the old `cross_validate` API must migrate to the string-based cross-validation options.
* Breaking change: Changed `build()` to wrap all accumulated string-parse errors in a `LowessError::ParseErrors(Vec<LowessError>)` value instead of surfacing only the first error. Code matching on `LowessError::InvalidOption` from `build()` must be updated.

## 1.3.0

### Changed

* Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

## 1.2.0

### Fixed

* Fixed lowess user documentation.

## 1.1.2

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 1.1.1

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 1.1.0

### Added

* Added `Mean` scaling method (Mean Absolute Deviation)

### Fixed

* Batch, Streaming, and Online adapters now propagate fitting errors to callers instead of treating failed fits as successful.
* Fixed a bug where the `Extend` boundary policy was never applied.
* Improved numerical precision for large-offset data through coordinate centering.

## 1.0.0

### Changed

* Improved robustness for custom numeric types.

## 0.99.9

### Changed

* Bump rust version to 1.88 for better stability
* Improve API docs

## 0.99.8

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.99.7

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.99.6

### Fixed

* Fixed README formatting and links.

## 0.99.5

### Changed

* Reduced package size significantly by removing unnecessary dev files and docs from the final package.

### Fixed

* Fixed `StreamingAdapter` indexing bug that caused merged overlap points to be skipped in output
* Simplified `StreamingAdapter` API: user now provides contiguous, non-overlapping chunks while the adapter handles internal buffering and merging
* Standardized `OnlineLowess` default `min_points` to 2 (enabling smoothing after just one point)
* Sanitized residual output to avoid "negative zero" (`-0.0000`) display for near-zero values

## 0.99.2

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.99.1

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.99.0

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.7.0

### Added

* `NoBoundary` variant to `BoundaryPolicy` enum (original Cleveland behavior)
* `ScalingMethod` enum with `MAR` and `MAD` variants for configurable robust scale estimation
* SIMD-optimized weighted least squares accumulation for `f64` and `f32`

### Changed

* Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
* Improved performance of window-weight computation, median calculation, and iteration loops.
* Added boundary thresholds for numerical stability
* Unified scale estimation logic under `ScalingMethod`
* Optimized K-Fold Cross-Validation performance

## 0.6.0

### Added

* `cv_seed` field to `CVConfig` for reproducible K-Fold cross-validation

### Changed

* Refactored `cross_validate` API to use `CVConfig` struct
* Updated `prelude` to export enum variants directly
* Removed the `CVMethod` and `CrossValidationStrategy` enums and corresponding type exports from the prelude.

### Fixed

* Various broken documentation links
* Bug in `Batch` and `Streaming` adapter conversion logic

## 0.5.3

### Changed

* Improved sorting, window operations, MAD computation, regression, interpolation, and cross-validation performance.

## 0.4.0

### Changed

* Transformed into core LOWESS implementation
* Removed `rayon` and `ndarray` dependencies
* Improved performance from 4-16× to 4-29× faster than statsmodels
* Changed license from MIT to dual AGPL-3.0 and Commercial License
* Removed convenience re-exports

## 0.3.0

### Changed

* Updated the minimum supported Rust version to 1.86.0.
* Modified features: default std mode includes ndarray/std and rayon
* Improved documentation

### Fixed

* no-std build now compiles successfully

## 0.2.0

### Changed

* Renamed "quartic" kernel to "biweight"
* Cross-validation now uses true k-fold validation
* Online LOWESS performs O(span) incremental updates
* Numerous performance optimizations and numerical stability improvements

## 0.1.0

### Added

* Initial LOWESS implementation based on Cleveland (1979)
* Type-safe builder pattern API
* Support for `f32` and `f64` types
* Seven kernel weight functions
* Statistical features (standard errors, confidence/prediction intervals)
* Comprehensive diagnostics
* Cross-validation with multiple strategies
* Delta-based interpolation
* Streaming and online processing variants
* Optional `parallel` and `ndarray` features
* Comprehensive error handling
* Extensive documentation
