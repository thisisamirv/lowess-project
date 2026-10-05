<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md).

## \[Unreleased\]

### Added

* Enabled the optional wgpu DirectX 12 backend for GPU-enabled Windows builds. DXC is loaded dynamically with an FXC fallback, avoiding eager imports of `dxcompiler.dll` and `dxil.dll`.
* Added GLES for Android GPU builds with a target-scoped `wgpu` feature; Windows continues to omit the GLES-only loader imports.
* Added residual-bootstrap standard errors and percentile confidence/prediction intervals via `.intervals(IntervalsBuilder::new().bootstrap(n))` (Batch, per-chunk Streaming, and full-update Online per sliding window). With `parallel(true)` Batch and Streaming refits run concurrently and match a sequential run; Online refits sequentially. Online incremental mode rejects them at `.build()`.

### Changed

* Clarified that Batch `residual_sd` is `1.4826 * MAD`, while Streaming reports the cumulative sample standard deviation of emitted residuals.
* Breaking change: a single `.seed(seed)` on the outer Rust fit builder now controls both CV folds and residual-bootstrap draws; nested CV and interval seed setters were removed. A seed alone does not enable bootstrap.
* Breaking change: removed individual `confidence_intervals`, `prediction_intervals`, `bootstrap_intervals`, and `bootstrap_seed` Rust wrapper methods; use `.intervals(IntervalsBuilder::new()...)` instead.
* Breaking change: replaced individual `return_*` and `cv_*` options with grouped `.outputs([...])` and `.cv(CVBuilder...)` configuration.

### Fixed

* Fixed CPU Gaussian standard errors for fitted and queried values to use the full unbounded kernel support.
* Made even-sample medians and mean-absolute residual scaling overflow-resistant for large finite values, and centered Batch/Streaming R-squared accumulation to preserve one-ULP response variation at large offsets.
* Fixed global OLS fits treating predictor values with a large offset as degenerate; translated inputs now retain their fitted slope.
* Fixed fraction-1 global fits ignoring custom weights, including when Batch sorts observations by x.
* Fixed Batch `missing = "drop"` accepting custom weights with a length different from the original input; weights are validated before rows are dropped.
* Fixed standard errors for global weighted fits to account for observation weights and weighted prediction leverage.
* Fixed local Batch standard errors ignoring `custom_weights`; parallel SEs now use the same case-weighted local moments as sequential fits.
* Fixed global fits with all-zero custom weights to honor the configured zero-weight fallback policy.
* Fixed fraction-1 global fits ignoring configured robustness iterations; they now reweight observations and report iterations used.
* Fixed Batch cross-validation candidate fits ignoring `custom_weights`; K-fold CV now rejects more folds than observations instead of returning zero scores.
* Fixed Streaming and Online accepting invalid `auto_converge` tolerances; Online also rejects invalid explicit `delta` values while retaining NaN as its default sentinel.
* Fixed GPU Batch fits misaligning fitted values with unsorted inputs; results now preserve input and requested sorted order.
* Fixed GPU Batch silently ignoring `custom_weights`; GPU fit and CV candidate kernels now apply them directly.
* Reduced GPU adapter buffer requirements from 30 storage/32 total buffer bindings to 7 storage/8 total per shader stage by using per-compute-pipeline resource layouts.
* Fixed Online incremental mode accepting positive `delta` and `auto_converge` settings it cannot use; unsupported combinations now error.
* Fixed grouped intervals discarding distinct confidence and prediction levels; each requested coverage is now applied independently.
* Improved agreement with R/Cleveland on sparse, asymmetric, and high-iteration fits by aligning effective-zero robustness stopping, local-linear degeneracy handling, span and endpoint traversal, delta interpolation, and adjusted-weight accumulation.
* Fixed zero-radius neighborhoods dropping tied observations and accumulating adjusted weights in an order that diverged under robust iterations; the full tied run now contributes using reference-compatible normalization.
* Fixed synthetic boundary observations contaminating robustness scale estimates; padding points still receive weights but no longer affect the scale used to reweight original observations.
* Fixed Gaussian fits clipping the unbounded kernel to the nearest-neighbor window and flooring far-tail weights; the neighbor count now selects bandwidth while all observations contribute under the standard Gaussian formula.
* Fixed parallel fits refitting tied x-values separately when `delta = 0`, which made results on all-tied input depend on the refit position.
* Fixed GPU Gaussian fits and standard errors to include observations across the full input range.

## 4.1.0

### Added

* Added standard errors and confidence/prediction intervals to parallel Online and Streaming fits; online interval outputs require `update_mode("full")`.

### Fixed

* Changed the `OnlineLowess` default `iterations` from `3` to `0`, matching the non-robust incremental update mode; robustness iterations require `update_mode("full")`.
* Fixed standard errors collapsing to zero for observations with zero robustness weight in parallel and GPU interval calculations, and improved confidence-interval calibration.
* Fixed local GPU fits losing their slope for small-scale predictor data and corrected standard errors for degenerate predictor scales.

## 4.0.0

### Added

* Added `return_sorted` to the batch builder, returning results in ascending `x` order instead of input order.
* Added a `missing` option (`"error"` by default or `"drop"`) for non-finite `x`/`y` values: batch and streaming fits drop rows (and matching `custom_weights`), while online fits skip points and return `Ok(None)`. Length mismatches still error.

## 3.2.1

### Changed

* Clarified that `LowessResult.x` is returned in input order, even though the algorithm sorts internally.

## 3.2.0

### Changed

* Expanded the crate documentation with setup guidance, parameter explanations, standardized examples, and a dedicated GPU backend guide describing hardware requirements and performance considerations.

### Fixed

* Corrected crate examples and documentation rendering, including cross-references and mathematical notation.

## 3.1.0

### Changed

* Moved crate documentation to <https://docs.rs/fastLowess> and updated the README with package-specific guidance.

## 3.0.0

### Added

* Added cross-reference links in the API documentation to the corresponding user guides.

### Changed

* Breaking change: renamed `OnlineOutput` fields `smoothed` and `std_error` to `y` and `standard_error`.
* Added GPU backend selection to fastLowess configuration and adjusted GPU build features to avoid requiring unavailable Windows DLLs.
* Organized Streaming and Online API documentation and tutorials into dedicated user-guide pages.

### Fixed

* Corrected the Rust API reference to show the string option values accepted by the API.

## 2.0.0

### Added

* Added `iterations_used` to `OnlineOutput<T>` to report robustness iterations performed in full update mode.
* Added `LowessError::ParseErrors` to report accumulated builder string-parse errors together.
* Added string aliases for merge strategies and update modes.
* Added `custom_weights(Vec<T>)` to the batch builder for per-observation weighting.

### Changed

* Added `Lowess`, `StreamingLowess`, and `OnlineLowess` as the primary constructors, with mode-specific options configured directly on each builder.
* Breaking change: made adapter builder types internal; configure smoothing options through `LowessBuilder<T, Mode>` and the public constructors.
* Breaking change: changed enum-valued options to accept strings as well as enum variants.
* Breaking change: replaced `cross_validate(CVConfig)` with string-based cross-validation options; `KFold` and `LOOCV` are no longer exported from the prelude.
* Breaking change: `build()` now returns all accumulated string-parse failures in `LowessError::ParseErrors` instead of only the first error.
* Breaking change: made `Lowess`, `StreamingLowess`, and `OnlineLowess` dedicated wrapper types that default to parallel execution; callers using adapter-based construction must migrate to the corresponding constructor.
* Breaking change: narrowed the prelude exports to the primary user-facing types; import other configuration types directly.

## 1.3.0

### Changed

* Raised the minimum supported Rust version to 1.89.

### Fixed

* Fixed GPU cross-validation failures and stabilized GPU buffer downloads.

## 1.2.0

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 1.1.2

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 1.1.1

### Fixed

* Fixed GPU initialization and configuration issues, and improved recovery after GPU execution errors so later GPU operations can continue.

## 1.1.0

### Added

* Added the `Mean` scaling method.
* Added GPU support for selectable weight functions, robustness methods, scaling methods, zero-weight fallbacks, boundary policies, and automatic convergence.
* Added GPU support for prediction, confidence intervals, and cross-validation.

### Fixed

* Fixed GPU initialization failures and integer overflow for datasets larger than `u32::MAX`.

## 1.0.0

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.99.9

### Changed

* Raised the minimum supported Rust version to 1.88.

## 0.99.8

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.99.7

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.99.6

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.99.5

### Changed

* Reduced package size by excluding development-only files from published packages.

## 0.99.2

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.99.1

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.99.0

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.7.0

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.6.0

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.5.3

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.4.0

### Changed

* Improved parallel fitting and cross-validation performance, including reduced allocations for large datasets and tied predictor values.

* Changed the license from AGPL-3.0-or-later to dual MIT OR Apache-2.0.

## 0.3.0

### Added

* Added CPU (default) and GPU Cargo features, with a `backend()` option for selecting the execution backend.

### Changed

* Breaking change: renamed `Extended*LowessBuilder` types to `Parallel*LowessBuilder`.
* Breaking change: removed the Sequential, parallel, and ndarray adaptors.

## 0.2.0

### Changed

* Improved fitting performance by replacing linear anchor scans with binary search and reducing per-iteration calculations.

## 0.1.0

### Added

* Initial release with parallel execution support.
