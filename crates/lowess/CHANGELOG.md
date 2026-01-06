# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0]

### Added

- Added `NoBoundary` variant to `BoundaryPolicy` enum, allowing users to disable boundary padding entirely (original Cleveland behavior).
- Added `ScalingMethod` enum with `MAR` (Median Absolute Residual) and `MAD` (Median Absolute Deviation) variants for configurable robust scale estimation.
- Added SIMD-optimized weighted least squares accumulation for `f64` (using `f64x2`) and `f32` (using `f32x8`) via the `wide` crate.
- Added `WLSSolver` trait for type-specific SIMD dispatch in regression computations.
- Added `CVBuffer` struct in `primitives/buffer.rs` for pre-allocated cross-validation scratch buffers.
- Added `VecExt` trait in `primitives/buffer.rs` for efficient vector reuse via `assign` and `assign_slice`.
- Added persistent scratch buffers to `OnlineBuffer` and `StreamingBuffer` to eliminate allocations in hot loops.

### Changed

- Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0.
- Refactored partition-related types: moved `SortResult` from inline struct to `primitives/sorting.rs`.
- Replaced `RangeInclusive` iterations with `while` loops in `RegressionContext::fit` for improved performance.
- Optimized `compute_window_weights` with `while` loops instead of iterator-based loops.
- Optimized `median_inplace` in `ScalingMethod` with `while` loop for finding the largest value in the lower half.
- Added boundary thresholds (`h1`, `h9`) to weight computation for numerical stability at near-zero and near-boundary distances.
- Unified scale estimation logic under `ScalingMethod`, replacing the previous `mad.rs` module.
- Refactored `LowessExecutor` to accept optional external buffers, enabling callers to manage and reuse scratch space.
- Optimized K-Fold Cross-Validation performance by moving training data preparation and sorting outside the candidate evaluation loop.
- Moved `auto_convergence` and `interval_method` into `LowessExecutor` for better encapsulation during buffered execution.

## [0.6.0]

### Added

- Added `cv_seed` field to `CVConfig` for reproducible K-Fold cross-validation results.
- K-Fold CV now shuffles data indices when a seed is provided in `CVConfig`.
- Added `Backend` enum (`CPU`, `GPU`) in new `primitives/backend.rs` module as a placeholder for GPU acceleration support in downstream crates.
- Added `FitPassFn` type alias for GPU-optimized fit processing in downstream crates.
- Added development-only fields to `LowessConfig`, `LowessExecutor`, and all adapter builders: `custom_fit_pass`, `custom_cv_pass`, `custom_interval_pass`, `backend`, and `parallel`. These are marked `#[doc(hidden)]` and intended for extension crates like `fastLowess`.
- Added `from_config` and `to_config` methods to `LowessExecutor` for simplified configuration management.

### Changed

- Refactored `cross_validate` API to use `CVConfig` struct with constructor methods: `KFold(k, fractions)` and `LOOCV(fractions)`.
- Refactored `Window::recenter` to be bidirectional, allowing the window to slide both left and right.
- Replaced manual `extern crate alloc` declarations with `no_std`-compliant imports where appropriate.
- Internal: Replaced `CVMethod` configuration logic with `CVKind`.
- Updated `prelude` to export enum variants directly (e.g., `Batch`, `Tricube`, `Bisquare`) instead of the enum types themselves, simplifying API usage.
- Reorganized `src/engine/executor.rs` into a unified logical flow, moving from high-level entry points to low-level mathematical primitives.
- Standardized the visual separation of internal development features using prominent comment header blocks (e.g., `// DEV`) across all builders and executors.
- Hidden internal-only and development-focused fields and methods from public documentation using `#[doc(hidden)]` to maintain a clean public API surface.
- Consolidated standard error calculation logic into a unified `compute_std_errors` helper, supporting both internal math and custom external hooks.
- Updated all adapter conversion logic, setter methods, and core test suite to align with the new development field structure.
- Standardized internal documentation and code interaction patterns for better readability and maintainability.
- Updated `Makefile` to automatically evaluate `std`, `no-std`, and `dev` features, and added example execution targets.
- Updated `CONTRIBUTING.md` to reflect the new Makefile structure and development workflow.

### Fixed

- Fixed various broken documentation links in `engine/mod.rs`, `engine/executor.rs`, `engine/validator.rs`, and `engine/output.rs`.
- Fixed `WeightParams` struct and tests to remove unused `use_robustness` field, resolving a clippy warning.
- Fixed a bug in `Batch` and `Streaming` adapter conversion logic where `return_diagnostics` and `compute_residuals` fields were not being correctly propagated from the generic builder.

### Removed

- Removed unused `GLSModel::local_wls` method from `regression.rs`.
- Removed `CVMethod` and `CrossValidationStrategy` enums in favor of unified `CVConfig` struct.
- Removed `Adapter`, `BoundaryPolicy`, `CrossValidationStrategy`, `MergeStrategy`, `RobustnessMethod`, `UpdateMode`, `WeightFunction`, and `ZeroWeightFallback` type exports from `prelude`.
- Removed the `type Result<T>` alias (previously in `api.rs` and exported via prelude) which shadowed `std::result::Result`. This alias caused ambiguity by implicitly binding `LowessError`. We now strictly follow Rust idioms: explicit `Result<LowessResult<T>, LowessError>` return types using the standard library `Result`.
- Removed `.cargo/config.toml` to simplify build configuration.

## [0.5.3]

### Changed

- Consolidated validation logic into `src/engine/validator.rs`.
- Removed redundant configuration defaults in `api.rs`.
- Optimized `src/primitives/sorting.rs` with fast path for sorted data and improved memory layout.
- Inlined hot methods in `src/primitives/window.rs`.
- Optimized `src/math/mad.rs` by promoting inplace computation to the primary `compute_mad` API, making buffer mutation explicit and reducing allocations library-wide.
- Refactored `src/algorithms/robustness.rs` to use a provided scratch buffer, making robustness weight updates entirely allocation-free.
- Optimized `src/algorithms/regression.rs` with fused `local_wls_with_sum` loops that leverage pre-calculated weight sums, eliminating redundant computations and nested loops.
- Improved `src/math/kernel.rs` with vectorized `slice::fill` operations and early termination for out-of-window calculations on sorted data.
- Updated `src/evaluation/diagnostics.rs` and `src/evaluation/intervals.rs` to leverage the new allocation-aware math patterns.
- Optimized `src/algorithms/interpolation.rs` with vectorized `slice::fill` for tied values and precomputed slopes to reduce divisions in interpolation loops.
- Refactored `src/evaluation/cv.rs` to use scratch buffers for training data in cross-validation, eliminating allocations in k-fold CV and LOOCV.
- Added batched linear interpolation with linear scanning in `cv.rs` for sorted test sets, replacing repeated binary searches in k-fold CV.
- Added pre-allocated scratch buffers to `src/adapters/online.rs` to eliminate per-call allocations in `add_point()`.
- Optimized `src/adapters/streaming.rs` to move sorted data instead of cloning when no overlap buffer exists.
- Simplified residual calculation in `src/adapters/batch.rs` with idiomatic zip iterator.
- Optimized delta interpolation in `src/engine/executor.rs` by replacing O(n) linear scan with O(log n) binary search using `partition_point` for anchor point discovery, providing significant speedup on dense data with large delta values.

## [0.5.2]

### Changed

- Dropped LaTeX-style math formulas from the documentation due to rendering issues on docs.rs.
- Improved documentation.

## [0.5.1]

### Added

- Added convenience setters to the adapters, allowing users to access a more flexible API, in which parameters can be set after adapter parameter is set.

## [0.5.0]

### Added

- Added more explanation on how to use the streaming mode in the documentation.
- Added LaTeX-style math formulas to the documentation.
- Added explicit `BoundaryPolicy` variants (`Extend`, `Reflect`, `Zero`) to all adapters. This moves from the default asymmetric windowing at boundaries to explicit data padding, which allows for centered windows at the edges and significantly reduces boundary bias.
- Added `SmoothPassFn` to the public API, allowing users to pass a custom smooth pass function to the executor.
- The Online adapter now supports delta parameter, which is useful for performance if you have a very large window_capacity.
- Added `Incremental` update mode to the Online adapter, which allows for true incremental updates to the model. Now it supports both `Full` (a full re-smooth of the window) and `Incremental` (only the latest point is smoothed) update modes.
- Added `auto_convergence` support to both Online and Streaming adapters, which allows for early stopping of the robustness process.
- Added `return_robustness_weights` parameter to the Online and Streaming adapters, allowing users to retrieve the weights used for outlier downweighting.
- Added `return_diagnostics` support to the Streaming adapter, providing cumulative global diagnostics (RMSE, MAE, R², Residual SD) across chunks.
- Improved API to not allow parameter overrides, thus preventing accidental misconfiguration.
- Added extensive tests for degenerate and edge cases, ensuring the library's robustness and future-proofing the codebase.

### Fixed

- Fixed a logic bug in `LinearRegression::fit` where zero-radius windows (degenerate bandwidth) would return a raw sum instead of a corrected weighted average.
- Fixed `GLSModel::fit_wls` to correctly handle unnormalized weights by explicitly computing the weight sum.
- Fixed a major bug in `api` where the streaming and online adapters were using the defaults instead of the parameters passed to the builder.
- Fixed a bug in Acklam's inverse CDF calculation resulting in incorrect z-scores (and thus incorrect confidence intervals) at high confidence levels (>=0.975).
- Added a safety guard for empty input or invalid window to `kernel` submodule.

### Changed

- Polished documentation.
- Changed some internal caller names to maintain consistency with the rest of the codebase.
- Delegate parameter defaults from builder to adapters.

### Removed

- Removed the unused internal utility `skip_close_points` and its associated tests, as executor handles this logic.
- Removed the confusing convenience constructors `quick` and `robust` in favor of more explicit builders.
- Removed the `max_iterations` parameter in favor of the `iterations` parameter which is now capped at 1000, with 3 as the default. This will simplify the API and make it more consistent.

## [0.4.1]

### Added

- Added a new flag to executor and adapters allowing passing a custom smooth pass function. This can be used to delegate the smoothing process to another crate.
- Added a few benchmarking examples.

### Changed

- Changed the `Cargo.toml` license from dual AGPL-3.0 and Commercial to AGPL-3.0-or-later. It is required to be able to publish the crate to crates.io. The commercial license is still available for purchase.
- Small styling changes to some of the imports.
- Renamed the `testing` module to `internals` to better reflect its purpose. It is used for exposing internal private modules for testing and access by other crates.
- Updated K-Fold cross-validation to compute the mean of fold RMSEs instead of global RMSE.

### Fixed

- Updated `LowessResult` to return the final robustness weights from the iterative process instead of the initial weights.
- Inline delta logic directly into `fit_and_interpolate_remaining` with a simple forward linear scan, achieving maximum performance.
- Fixed small discrapancies in `robustness` submodule docs.
- Fixed `README.md` to include the correct license.

## [0.4.0]

### Overview

- The main goal of this release is to transform the project into a core LOWESS implementation, removing all dependencies and adapting the API to be more generic and flexible. For this matter, we will drop the ndarray and rayon dependencies, refactor the code to simplify the flow, and adapt the API to be more generic and flexible.
- The performance is improved from 4-16× faster performance than Python's statsmodels in the previous version to 4-29× faster performance than Python's statsmodels.
- The API is simplified and made more generic and flexible.
- Benchmarking and validation code is moved to a separate branch.
- Various other improvements are made to improve performance, ease of use, and maintainability.
- The MRSV is reduced from 1.86 to 1.85 (the minimum for Rust edition 2024), making the crate more compatible with other crates.

### Changed

- Changed the license from MIT to dual AGPL-3.0 and Commercial License.
- Improved the ci.yml file to ease the release process.
- The documentation is improved significantly, with better and more centered examples, diagrams, and tables.
- The LOC count is reduced from 3863 to 3263, making the codebase more maintainable.
- More validation logic is added to ensure the library's correctness.

### Added

- Added a Makefile to ease the development process.
- New examples are added to ease the usage of the library.

### Removed

- Removed `rayon` and `ndarray` dependencies completely.
- Removed convenicnece re-exports all over the codebase.
- Removed all validation and comparison code.
- Removed all benchmarking code.

## [0.3.0]

### Changed

- Updated the Rust version to 1.86.0.
- Updated the criterion version to 0.8.
- Modified the features:
  - In default (std) mode: ndarray/std and rayon are included by default.
  - In no_std mode: ndarray (no std) is included by default, but rayon is not included.
  - Now, instead of having default, ndarray, and parallel features, we simply have a default std build and a no-std build (cargo build --no-default-features).
  - This will simplify the build process, ensuring in a std build the user gets the optimal performance (but with the option of disabling rayon during runtime, i.e., 'parallel = false'), while also providing a no_std build for users who need to compile without std.
- Updated the project structure in CONTRIBUTING.md.
- Improved documentation.
- Updated the benchmarking results.

### Fixed

- Now the no-std build compiles and runs successfully.

### Removed

- Removed the sequential, parallel, and ndarray adaptors. Now, they are part of standard adaptor.

## [0.2.0]

### Changed

#### Architecture & Organization

- Restructured project to reduce intra-module dependencies for improved maintainability (see CONTRIBUTING.md for new structure).

#### API & Terminology

- Renamed "quartic" kernel to "biweight" to match standard statistical terminology (implementation unchanged)
- Renamed `compute_robust_weights` to `compute_bisquare_weights` for specificity
- Renamed `compute_robust_scale` to `compute_mad_scale` for clarity
- Added `HealthCheckConfig` for customizable thresholds in `health_check_with()`, keeping `health_check()` backward-compatible

#### Algorithm Improvements

- Cross-validation now uses true k-fold validation (parallelized when appropriate) then refits on full data with selected fraction
- When both CV and auto-convergence enabled, selected fraction is re-fit using auto-convergence instead of fixed iterations
- Online LOWESS now performs O(span) incremental local WLS updates per insertion instead of re-fitting entire window
- Streaming mode now honors `builder.delta` parameter (previously disabled)
- Parallel execution now uses delta-aware path for large datasets (n ≥ PARALLEL_THRESHOLD)

#### Performance Optimizations

- Merged `compute_weight_fast` and `compute_weight` with early-return optimization for bounded kernels
- Reuse single preallocated weights buffer in sequential path to reduce memory allocations
- Precompute current point's kernel weight to avoid recomputation in standard error calculation
- Extract LOWESS loop into `lowess_parallel_impl` to deduplicate parallel entry points
- Parallelize residual computation in parallel implementation for n ≥ PARALLEL_THRESHOLD
- Replace runtime bandwidth checks with debug-only assertions in hot loops
- Truncate Gaussian kernel at |u| > 6.0 for speed (returns tiny positive weight to avoid zero normalization issues)
- Added `PI_OVER_2` constant to avoid recomputing in cosine kernel

#### Numerical Stability & Robustness

- Use stable sort in `sort_by_x` to preserve relative order of tied x-values for determinism
- Scale threshold now uses `max(SCALE_THRESHOLD * mean_abs, MIN_TUNED_SCALE)` to prevent vanishing thresholds
- `compute_mad_scale` returns small positive epsilon instead of zero when MAD == 0
- Fixed Gaussian kernel conversion to treat failed `to_f64` as infinite distance (zero weight) instead of zero distance
- Guard against tiny bandwidths in `OnlineLowess` with `min_bandwidth` threshold (1e-10)
- Clamp and validate `max_iterations` (treat 0 as 1, cap at 1000)
- Guard `compute_intervals` against non-finite interval widths by replacing with small epsilon
- `WeightParams::new` validates/clamps bandwidth and computes h1/h9 for numerical safety

#### Standard Error & Confidence Intervals

- Fixed leverage calculation using precomputed current-point weight for normalization
- Fixed standard error computation: use effective sample size (sum of weights) minus model parameters for degrees of freedom
- Fixed effective_df computation: derive leverages as (SE / residual_sd)² instead of summing SE²
- Enforce confidence-level validation: `approximate_z_score` returns Result and propagates errors
- Fixed `validate_standard_errors` to allow zero values and cap at 10×y_range
- Added input validation to interval functions: verify lengths, handle zero-length, propagate errors

#### Window & Index Management

- Introduced `WindowBounds` struct to replace separate left/right parameters
- `compute_weights` returns rightmost included index, used by `fit_point`
- Zero weight buffer from `left..n` at start to prevent stale weights
- Defensive bounds check in `fit_point` returns None for out-of-range indices
- `update_window` checks and clamps indices, returns safely on invalid input

#### Delta & Interpolation

- Fixed `interpolate_gap` to handle duplicate x endpoints by averaging fits
- Clarified `skip_close_points` index calculation using `next = last + 1 + pos`
- Added optional `fitted_mask` parameter to `skip_close_points` for future optimization
- Prevent division-by-zero in `interpolate_prediction` for duplicate bracketing x values

### Streaming & Chunking

- Fixed overlap handling: correctly detect pre-overlapped input to avoid duplication
- Added overlap/chunk size validation
- Clamp overlap to at most half chunk size
- Handle zero/empty chunk sizes to prevent infinite partitioning
- `ChunkedLowessBuilder::process_chunks` now non-consuming (`&self`) with per-chunk validation
- Support both sequential and parallel execution without consuming builder

#### Cross-Validation

- Centralized training-set construction with `build_subset_from_indices`
- Updated k-fold/LOOCV to use centralized construction, eliminating duplicate slicing
- Prevent panic by defensively handling empty `cv_scores` with safe default index

#### Code Clarity

- Extracted magic literal `2.0` into `LINEAR_PARAMS` constant for intercept + slope
- Renamed `kernel_idx` to `kernel_value` for accuracy
- Simplified `is_effectively_zero` to use scale-relative epsilon threshold
- Added explicit bandwidth validation to `compute_distance_weights`
- Clarified determinism notes in `parallel.rs`
- Documented Effective Degrees of Freedom calculation
- Added warning about computational expense of `compute_effective_df`
- Added debug assertion to `initialize_window`

#### Build & Dependencies

- Unified `fit_impl` with runtime parallel selection, removing duplicate cfg-gated variants
- Relaxed ndarray `fit_ndarray` bounds: removed unnecessary Send/Sync/'static constraints
- Added max limit for `set_num_threads`

### Removed

#### Deprecated Functions

- Removed global standard-error fallback (σ_resid / √n)
- Removed duplicate `compute_robust_weights` from `kernel.rs`
- Removed `find_rightmost_point` and `compute_range` (now handled by `compute_weights`)
- Removed `compute_weight_batch` (no planned use)
- Removed `compute_bisquare_weights` (replaced by single `compute_bisquare_weight` API)
- Removed unused diagnostic functions: `are_weights_uniform`, `effective_sample_size`
- Removed unimplemented functions: `compute_leverage`, `compute_effective_df`, `compute_residual_variance`, `weighted_polynomial_fit`, `locally_constant_fit`
- Removed duplicate `normalize_weights` and `compute_weighted_average` from `regression.rs`
- Removed `recommended_for_lowess` and `most_efficient` from `kernel.rs`

### Fixed

- Kernel efficiency calculation now returns accurate values (previously off by <0.005)
- Gaussian kernel conversion handles extreme values correctly
- Streaming overlap logic prevents duplicate/missing points
- Parallel builds now consistent with non-parallel builds (unified `fit_impl`)
- Zero-weight fallbacks now use correct window bounds from `compute_weights`

## [0.1.0]

### Added

#### Core Functionality

- Initial implementation of LOWESS (Locally Weighted Scatterplot Smoothing) algorithm based on Cleveland (1979)
- Type-safe builder pattern API via `Lowess::new()` for ergonomic configuration
- Support for `f32` and `f64` floating-point types through generic `Float` trait
- Robust smoothing via iteratively reweighted least squares (IRLS) with configurable iterations
- Deterministic output with reproducible results for fixed inputs and configuration

#### Weight Functions & Kernels

- Seven kernel weight functions:
  - Tricube (default, Cleveland's original)
  - Epanechnikov
  - Gaussian
  - Uniform
  - Quartic
  - Cosine
  - Triangle
- Kernel metadata via `WeightFunctionInfo` trait

#### Statistical Features

- Per-point standard error estimation
- Confidence intervals (mean response) at configurable levels (e.g., 95%)
- Prediction intervals (new observations) at configurable levels
- Optional residual computation
- Robustness weights output from IRLS iterations

#### Diagnostics

- Comprehensive fit diagnostics including:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² (coefficient of determination)
  - Akaike Information Criterion (AIC)
  - Corrected AIC (AICc) for small samples
  - Effective degrees of freedom
  - Count of downweighted points

#### Cross-Validation

- Automatic fraction selection via cross-validation
- Multiple CV strategies:
  - Simple train/test split (default)
  - K-fold cross-validation
  - Leave-one-out cross-validation (LOOCV)
- CV score reporting for model selection transparency

#### Performance Optimizations

- Delta-based interpolation for dense data (reduces O(n²) to effectively O(n×k))
- Configurable delta threshold (auto-calculates to 1% of x-range by default)
- Auto-convergence for IRLS iterations with configurable tolerance
- Early stopping when fitted values stabilize

#### Memory-Efficient Processing

- `StreamingLowess` for chunk-based processing of large datasets
- `OnlineLowess` for real-time sliding window smoothing
- `ChunkedLowess` for processing data in predefined chunks
- All streaming variants support the full configuration API

#### Optional Features

- **`parallel`**: Multi-threaded cross-validation and fitting via Rayon
  - Parallel CV across candidate fractions
  - Configurable chunk sizes for parallel processing
  - Linear speedup with available cores
- **`ndarray`**: Seamless integration with ndarray ecosystem
  - `fit_ndarray()` for direct ndarray input
  - Conversion methods: `to_ndarray()`, `as_ndarray()`
  - Support for confidence and prediction intervals in ndarray format
- **`std`** (default): Standard library support
  - Disable for `no_std` compatibility (requires `alloc`)

#### Error Handling

- Explicit `LowessError` enum with variants:
  - `EmptyInput`: Empty input arrays
  - `MismatchedInputs`: Different lengths for x and y
  - `InvalidFraction`: Fraction not in (0, 1]
  - `InvalidDelta`: Negative delta value
  - `InvalidNumericValue`: NaN or infinite values in inputs
  - `TooFewPoints`: Insufficient data for smoothing
  - `InvalidConfidenceLevel`: Confidence level not in (0, 1)
- No panics in release builds (defensive programming with debug assertions)
- Graceful degradation with documented fallback behaviors

#### Numerical Stability

- MAD (Median Absolute Deviation) for robust scale estimation
- Fallback to mean absolute residual when MAD ≈ 0
- Minimum tuned scale clamping to avoid division by zero
- Zero-weight neighborhood handling with configurable policies:
  - `UseLocalMean`: Fallback to local average
  - `ReturnOriginal`: Use original y value
  - `ReturnNone`: Propagate failure
- Uniform weight fallback when all kernel weights are zero
- Machine-epsilon aware comparisons in `is_effectively_zero()`

#### Documentation

- Comprehensive crate-level documentation with:
  - Conceptual overview of LOWESS
  - Parameter descriptions and typical ranges
  - Output structure documentation
  - Error handling guidance
  - Performance and operational best practices
  - Determinism and numeric safety guarantees
- Module-level documentation for all public modules:
  - `builder`: Builder pattern and result types
  - `core`: Main fitting algorithm
  - `kernel`: Weight functions
  - `regression`: Weighted least squares
  - `robustness`: IRLS implementation
  - `confidence`: Interval estimation
  - `streaming`: Memory-efficient variants
  - `utils`: Input validation and helpers
  - `parallel`: Parallel execution (feature-gated)
- Function-level documentation with examples
- Production usage guidelines and monitoring recommendations

#### Convenience Functions

- `lowess()`: One-line smoothing with defaults
- `lowess_with_fraction()`: Quick smoothing with custom fraction
- `lowess_robust()`: Pre-configured robust smoothing (5 iterations)
- `Lowess::robust()`: Builder preset for robust fitting

### Implementation Details

#### Algorithm Fidelity

- Faithful implementation of Cleveland (1979) algorithm
- Compatible with R's `stats::lowess()` function
- Bisquare (Tukey's biweight) robustness weighting
- Tricube kernel as default (Cleveland's original choice)
- MAD-based scale estimation with 1.4826 correction factor

#### Performance Characteristics

- O(n²) complexity for basic fitting
- O(n×k) effective complexity with delta optimization (k ≪ n for dense data)
- Linear speedup with parallel cross-validation (when `parallel` feature enabled)
- Minimal allocations through buffer reuse in hot paths

#### Testing

- Comprehensive unit tests for all modules
- Integration tests for end-to-end workflows
- Edge case coverage (empty inputs, single points, identical values, outliers)
- Numerical stability tests (near-zero scales, extreme values)
- Cross-validation correctness tests

#### Platform Support

- All Rust targets with `f32`/`f64` support
- `no_std` compatible with `alloc` (disable default features)
- Tested on Linux, macOS, and Windows
- MSRV (Minimum Supported Rust Version): 1.70.0

### Notes

This initial release provides a production-ready LOWESS implementation with advanced features beyond traditional ports:

- Statistical intervals and diagnostics for scientific computing
- Streaming variants for large-scale data processing
- Parallel execution for computational efficiency
- Comprehensive error handling and numerical stability
- Extensive documentation with domain-specific guidance (genomics, bioinformatics)

The API is designed for stability and follows Rust best practices. Future releases will maintain backward compatibility according to semantic versioning.
