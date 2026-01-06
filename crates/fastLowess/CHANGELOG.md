# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0]

### Added

- Added zero-allocation parallel fitting via `fit_all_points_parallel` using `for_each_init`.
- Added parallel CV memory reuse via `cv_pass_parallel` using `map_init`.
- Added refined delta optimization to `compute_anchor_points` and interpolate logic to skip redundant fits for tied x-values and copy previous values instead.
- Added parallel anchor precomputation for large datasets (n > 100K) using chunk-and-merge strategy.
- Added cache-oblivious tile-based processing in `fit_all_points_parallel` for large window sizes.

### Changed

- Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0.
- Updated `lowess` dependency to v0.7.0 to include breaking change fixes and performance improvements.
- Consolidate imports for `BoundaryPolicy`, `MergeStrategy`, and `UpdateMode` to match `lowess` crate reorganization.
- Updated `executor` to use `RegressionContext::fit()` instead of the removed `LinearRegression` struct.
- Updated parallel cross-validation logic to align with new `CVKind::run` signature.
- Centralized GPU module imports.
- **Performance (GPU)**: Implemented thread-local `GpuExecutor` persistence to avoid `wgpu` re-initialization overhead.
- **Performance (GPU)**: Added intelligent buffer capacity management to reuse existing GPU buffers across calls.
- **Performance (GPU)**: Record all robustness iterations into a single command buffer submission, reducing CPU-GPU synchronization latency.
- **Performance (GPU)**: Refactored the `fit_anchors` compute kernel to use `@workgroup` shared memory tiled loading, significantly reducing global memory bandwidth pressure.

### Fixed

- Added `WLSSolver` trait bounds to all logic to satisfy new `lowess` requirements.
- Minor documentation updates.
- Updated documentation examples and parameter tables to include `boundary_policy` and `scaling_method`.

## [0.3.0]

### Added

- Added `cpu` (default) and `gpu` Cargo features for modular platform-specific optimizations.
  - `cpu`: Enables `rayon` and `ndarray`-based parallel execution (default).
  - `gpu`: Adds `wgpu`, `bytemuck`, `pollster`, and `futures-intrusive` dependencies for GPU acceleration infrastructure.
- Added GPU execution engine placeholder in `src/engine/gpu.rs` with conditional compilation (`#[cfg(feature = "gpu")]`).
- Added `fit_pass_gpu` function for GPU-accelerated fit processing (feature-gated).
- Added `backend()` setter method to all builder structs for runtime backend selection.
- Added tests for GPU engine (`tests/gpu_tests.rs`) and parallel execution consistency.

### Changed

- **Renamed builders**: `Extended*LowessBuilder` → `Parallel*LowessBuilder` for clarity (e.g., `ExtendedBatchLowessBuilder` → `ParallelBatchLowessBuilder`).
- Migrated `parallel` field from builder-level to core `lowess` crate's `dev` structure, accessed via `.parallel()` setter.
- Updated `lowess` dependency from v0.5 to v0.6.0 with `dev` feature.
- Updated `prelude` to export enum variants directly (e.g., `Batch`, `Tricube`, `Bisquare`) instead of enum types.
- Refactored all adapters to use `builder.base.parallel()` and `builder.base.backend()` setter methods instead of direct field access.
- Updated parallel CV to use `cv_seed` for reproducible cross-validation (aligning with core crate).
- Applied comprehensive conditional compilation (`#[cfg(feature = "cpu")]`, `#[cfg(feature = "gpu")]`) throughout codebase.
- Updated `Makefile` to build, test, and document both `cpu` and `gpu` variants separately.
- Updated `CONTRIBUTING.md` to reflect new feature flag structure and development workflow.
- Made `ndarray` and `rayon` optional dependencies (enabled via `cpu` feature).

### Removed

- Removed `.cargo/config.toml` to simplify build configuration.
- Removed `Adapter`, `BoundaryPolicy`, `CrossValidationStrategy`, `MergeStrategy`, `RobustnessMethod`, `UpdateMode`, `WeightFunction`, and `ZeroWeightFallback` type exports from `prelude`.
- Removed the `type Result<T>` alias which shadowed `std::result::Result`. We now strictly follow Rust idioms: explicit `Result<LowessResult<T>, LowessError>` return types.

## [0.2.2]

### Changed

- Replaced linear O(n) scan in `compute_anchor_points` with `partition_point` binary search, reducing anchor discovery complexity from O(n) to O(log n) per anchor point.
- Eliminated per-iteration division in `interpolate_gap` by precomputing the slope once, reducing computational overhead in the interpolation loop.
- Replaced iterator-based assignment with `slice::fill` for tied x-values, enabling SIMD vectorization.
- Aligned with `lowess` crate v0.5.3 optimizations for consistent performance characteristics.

## [0.2.1]

### Changed

- Drop LaTeX formatting due to docs.rs rendering issues.
- Improve documentation.

## [0.2.0]

- For changes to the core logic and the API, see the [lowess](https://github.com/av746/lowess) crate.

### Added

- Added more explanation on how to use the streaming mode in the documentation.
- Added convenience wrappers to adapters, allowing for a more flexible API.
- Added support for the new features in the `lowess` crate version 0.5.1.

## [0.1.0]

### Added

- Initial release
