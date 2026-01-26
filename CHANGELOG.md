<!-- markdownlint-disable MD024 MD046 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.0.0

### Changed

**lowess:**

- Refactored the constants to make the library robust safely against custom numeric types.
- Minor improvements to the documentation.

## 0.99.9

### Changed

**Monorepo:**

- Bump rust version to 1.88 for better stability
- Change function-based builder pattern in the bindings to class-based builder pattern, allowing true streaming and online processing
- Improve API docs

**Julia:**

- Package is now registered on JuliaRegistries

**C++:**

- Library is now available on conda-forge (libfastlowess)

**R:**

- Package is now available on conda-forge (r-rfastlowess)

**Node.js:**

- Package is now available on npm (fastlowess)

**WASM:**

- Package is now available on npm (fastlowess-wasm)

## 0.99.8

### Added

**Julia:**

- Initial implementation

**Node.js:**

- Initial implementation

**WASM:**

- Initial implementation

**C++:**

- Initial implementation

## 0.99.7

### Changed

**Python:**

- Switch to Stable ABI for CPython

### Fixed

**Monorepo:**

- Fix README file links
- Fix Makefile bug with R versioning

## 0.99.6

### Fixed

**Monorepo:**

- Fix README file formats and links

## 0.99.5

### Changed

**Monorepo:**

- Reduced package size significantly by removing unnecessary dev files and docs from the final package.
- Implemented comprehensive Cargo workspace inheritance pattern
- Unified MSRV to 1.85.0
- Centralized all metadata (version, authors, edition, license, etc.) in root `Cargo.toml`
- All crates now use `workspace = true` for shared configuration
- Created unified `README.md` for all crates/packages
- Created unified `CHANGELOG.md` for all crates/packages
- Created unified `LICENSE` for all crates/packages
- Created unified `.gitignore` for all crates/packages
- Added comprehensive badges from all packages

### Fixed

**lowess:**

- Fixed `StreamingAdapter` indexing bug that caused merged overlap points to be skipped in output
- Simplified `StreamingAdapter` API: user now provides contiguous, non-overlapping chunks while the adapter handles internal buffering and merging
- Standardized `OnlineLowess` default `min_points` to 2 (enabling smoothing after just one point)
- Sanitized residual output to avoid "negative zero" (`-0.0000`) display for near-zero values

## 0.99.2

### Changed

**R:**

- Prepared package for rOpenSci Software Peer Review
- Renamed main functions to avoid conflicts with base R:
  - `smooth()` → `fastlowess()`
  - `smooth_online()` → `fastlowess_online()`
  - `smooth_streaming()` → `fastlowess_streaming()`
- Updated documentation to reflect new API and rOpenSci guidelines

### Added

**R:**

- Documentation website using `pkgdown` with automated deployment
- Comprehensive function documentation with examples and cross-references
- URL validation using `urlchecker` in CI workflow
- Rigorous parameter validation for all exported functions
- Expanded test suite achieving >96% coverage
- Codecov CI workflow and badge

### Fixed

**R:**

- Documentation URLs
- Package startup messages
- `pkgcheck` workflow to run on host runner

## 0.99.1

### Changed

**R:**

- Modified package for Bioconductor submission

## 0.99.0

### Added

**R:**

- Support for new features in `fastLowess` v0.4.0, including `NoBoundary` boundary policy and `MAD`/`MAR` scaling methods

### Changed

**R:**

- Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
- Updated documentation

## 0.7.0

### Added

**lowess:**

- `NoBoundary` variant to `BoundaryPolicy` enum (original Cleveland behavior)
- `ScalingMethod` enum with `MAR` and `MAD` variants for configurable robust scale estimation
- SIMD-optimized weighted least squares accumulation for `f64` and `f32`
- `WLSSolver` trait for type-specific SIMD dispatch
- `CVBuffer` struct for pre-allocated cross-validation scratch buffers
- `VecExt` trait for efficient vector reuse
- Persistent scratch buffers to `OnlineBuffer` and `StreamingBuffer`

### Changed

**lowess:**

- Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
- Refactored partition-related types
- Replaced `RangeInclusive` iterations with `while` loops for improved performance
- Optimized `compute_window_weights` and `median_inplace`
- Added boundary thresholds for numerical stability
- Unified scale estimation logic under `ScalingMethod`
- Refactored `LowessExecutor` to accept optional external buffers
- Optimized K-Fold Cross-Validation performance

## 0.6.0

### Added

**lowess:**

- `cv_seed` field to `CVConfig` for reproducible K-Fold cross-validation
- `Backend` enum (`CPU`, `GPU`) as placeholder for GPU acceleration
- Development-only fields: `custom_fit_pass`, `custom_cv_pass`, `custom_interval_pass`, `backend`, `parallel`
- `from_config` and `to_config` methods to `LowessExecutor`

### Changed

**lowess:**

- Refactored `cross_validate` API to use `CVConfig` struct
- Refactored `Window::recenter` to be bidirectional
- Updated `prelude` to export enum variants directly
- Reorganized `src/engine/executor.rs` into unified logical flow
- Hidden internal-only fields from public documentation

### Fixed

**lowess:**

- Various broken documentation links
- `WeightParams` struct to remove unused field
- Bug in `Batch` and `Streaming` adapter conversion logic

### Removed

**lowess:**

- Unused `GLSModel::local_wls` method
- `CVMethod` and `CrossValidationStrategy` enums
- Type exports from `prelude` that caused ambiguity
- `.cargo/config.toml`

## 0.5.3

### Changed

**lowess:**

- Consolidated validation logic into `src/engine/validator.rs`
- Optimized sorting, window operations, MAD computation, and regression
- Refactored robustness to use scratch buffers (allocation-free)
- Optimized interpolation and cross-validation
- Optimized delta interpolation with binary search

## 0.4.0

### Added

**fastLowess:**

- Zero-allocation parallel fitting via `fit_all_points_parallel`
- Parallel CV memory reuse via `cv_pass_parallel`
- Refined delta optimization for tied x-values
- Parallel anchor precomputation for large datasets
- Cache-oblivious tile-based processing

**Python:**

- Support for new features in `fastLowess` v0.4.0

**R:**

- Support for new features in `fastLowess` v0.4.0

### Changed

**lowess:**

- Transformed into core LOWESS implementation
- Removed `rayon` and `ndarray` dependencies
- Improved performance from 4-16× to 4-29× faster than statsmodels
- Changed license from MIT to dual AGPL-3.0 and Commercial License
- Reduced LOC from 3863 to 3263

**fastLowess:**

- Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
- Updated `lowess` dependency to v0.7.0
- Implemented thread-local `GpuExecutor` persistence
- Added intelligent buffer capacity management for GPU
- Refactored GPU compute kernel with shared memory tiling

**Python:**

- Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
- Updated documentation

**R:**

- Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
- Updated documentation

### Removed

**lowess:**

- Validation and comparison code
- Benchmarking code
- Convenience re-exports

## 0.3.0

### Added

**fastLowess:**

- `cpu` (default) and `gpu` Cargo features
- GPU execution engine in `src/engine/gpu.rs`
- `fit_pass_gpu` function for GPU-accelerated processing
- `backend()` setter method to all builders
- Tests for GPU engine and parallel execution consistency

**R:**

- Option to install from R-universe without Rust

### Changed

**lowess:**

- Updated Rust version to 1.86.0
- Modified features: default std mode includes ndarray/std and rayon
- Improved documentation

**fastLowess:**

- Renamed builders: `Extended*LowessBuilder` → `Parallel*LowessBuilder`
- Migrated `parallel` field to core `lowess` crate
- Updated `lowess` dependency to v0.6.0
- Made `ndarray` and `rayon` optional dependencies

**Python:**

- Updated `fastLowess` dependency to v0.3.0
- Refactored internal API usage
- Updated cross-validation parameter handling

**R:**

- Updated to `fastLowess` v0.3.0 and `lowess` v0.6.0
- Updated cross-validation API

### Fixed

**lowess:**

- no-std build now compiles successfully

**Python:**

- Documentation build errors
- Bug where `parallel` argument was not exposed

**R:**

- Automated vendor checksum fixing for CI builds

### Removed

**fastLowess:**

- `.cargo/config.toml`
- Type exports from `prelude` that shadowed std types
- Sequential, parallel, and ndarray adaptors

## 0.2.0

### Added

**Python:**

- Support for new features in `fastLowess` v0.2.0

**R:**

- Support for new features in `fastLowess` v0.2.0

### Changed

**lowess:**

- Restructured project to reduce intra-module dependencies
- Renamed "quartic" kernel to "biweight"
- Cross-validation now uses true k-fold validation
- Online LOWESS performs O(span) incremental updates
- Numerous performance optimizations and numerical stability improvements

**fastLowess:**

- Replaced linear scan with binary search in `compute_anchor_points`
- Eliminated per-iteration division in `interpolate_gap`
- Aligned with `lowess` crate v0.5.3 optimizations

**Python:**

- Updated documentation
- Changed module name from `fastLowess` to `fastlowess`

**R:**

- Improved documentation

## 0.1.0

### Added

**lowess:**

- Initial LOWESS implementation based on Cleveland (1979)
- Type-safe builder pattern API
- Support for `f32` and `f64` types
- Seven kernel weight functions
- Statistical features (standard errors, confidence/prediction intervals)
- Comprehensive diagnostics
- Cross-validation with multiple strategies
- Delta-based interpolation
- Streaming and online processing variants
- Optional `parallel` and `ndarray` features
- Comprehensive error handling
- Extensive documentation

**fastLowess:**

- Initial release with parallel execution support

**Python:**

- Python binding for `fastLowess`
- Support for Python 3.14

**R:**

- R binding for `fastLowess`
