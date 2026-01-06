# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.99.3] - 2026-01-06

### Changed

**Monorepo:**

- Implemented comprehensive Cargo workspace inheritance pattern
- Centralized all metadata (version, authors, edition, license, etc.) in root `Cargo.toml`
- All crates now use `workspace = true` for shared configuration
- Created unified `README.md` for all crates/packages
- Created unified `CHANGELOG.md` for all crates/packages
- Created unified `LICENSE` for all crates/packages
- Created unified `.gitignore` for all crates/packages
- Added comprehensive badges from all packages

## [0.99.2]

### Changed

**rfastlowess (R):**

- Prepared package for rOpenSci Software Peer Review
- Renamed main functions to avoid conflicts with base R:
  - `smooth()` → `fastlowess()`
  - `smooth_online()` → `fastlowess_online()`
  - `smooth_streaming()` → `fastlowess_streaming()`
- Updated documentation to reflect new API and rOpenSci guidelines

### Added

**rfastlowess (R):**

- Documentation website using `pkgdown` with automated deployment
- Comprehensive function documentation with examples and cross-references
- URL validation using `urlchecker` in CI workflow
- Rigorous parameter validation for all exported functions
- Expanded test suite achieving >96% coverage
- Codecov CI workflow and badge

### Fixed

**rfastlowess (R):**

- Documentation URLs
- Package startup messages
- `pkgcheck` workflow to run on host runner

## [0.99.1]

### Changed

**rfastlowess (R):**

- Modified package for Bioconductor submission

## [0.99.0]

### Added

**rfastlowess (R):**

- Support for new features in `fastLowess` v0.4.0, including `NoBoundary` boundary policy and `MAD`/`MAR` scaling methods

### Changed

**rfastlowess (R):**

- Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
- Updated documentation

## [0.7.0]

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

## [0.6.0]

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

## [0.5.3]

### Changed

**lowess:**

- Consolidated validation logic into `src/engine/validator.rs`
- Optimized sorting, window operations, MAD computation, and regression
- Refactored robustness to use scratch buffers (allocation-free)
- Optimized interpolation and cross-validation
- Optimized delta interpolation with binary search

## [0.4.0]

### Added

**fastLowess:**

- Zero-allocation parallel fitting via `fit_all_points_parallel`
- Parallel CV memory reuse via `cv_pass_parallel`
- Refined delta optimization for tied x-values
- Parallel anchor precomputation for large datasets
- Cache-oblivious tile-based processing

**fastlowess (Python):**

- Support for new features in `fastLowess` v0.4.0

**rfastlowess (R):**

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

**fastlowess (Python):**

- Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
- Updated documentation

**rfastlowess (R):**

- Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
- Updated documentation

### Removed

**lowess:**

- Validation and comparison code
- Benchmarking code
- Convenience re-exports

## [0.3.0]

### Added

**fastLowess:**

- `cpu` (default) and `gpu` Cargo features
- GPU execution engine in `src/engine/gpu.rs`
- `fit_pass_gpu` function for GPU-accelerated processing
- `backend()` setter method to all builders
- Tests for GPU engine and parallel execution consistency

**rfastlowess (R):**

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

**fastlowess (Python):**

- Updated `fastLowess` dependency to v0.3.0
- Refactored internal API usage
- Updated cross-validation parameter handling

**rfastlowess (R):**

- Updated to `fastLowess` v0.3.0 and `lowess` v0.6.0
- Updated cross-validation API

### Fixed

**lowess:**

- no-std build now compiles successfully

**fastlowess (Python):**

- Documentation build errors
- Bug where `parallel` argument was not exposed

**rfastlowess (R):**

- Automated vendor checksum fixing for CI builds

### Removed

**fastLowess:**

- `.cargo/config.toml`
- Type exports from `prelude` that shadowed std types
- Sequential, parallel, and ndarray adaptors

## [0.2.0]

### Added

**fastlowess (Python):**

- Support for new features in `fastLowess` v0.2.0

**rfastlowess (R):**

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

**fastlowess (Python):**

- Updated documentation
- Changed module name from `fastLowess` to `fastlowess`

**rfastlowess (R):**

- Improved documentation

## [0.1.0]

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

**fastlowess (Python):**

- Python binding for `fastLowess`
- Support for Python 3.14

**rfastlowess (R):**

- R binding for `fastLowess`
