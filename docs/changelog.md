<!-- markdownlint-disable MD024 MD046 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 2.0.0

### Changed

**Monorepo:**

- Renamed all public API method and option names from camelCase to snake_case across every binding and all documentation. This is a **breaking change** for all consumers of the C++, Node.js, and WASM APIs.
- Converted all documentation tables to compact single-space format.
- Updated `.clang-tidy` to configure `lower_case` as the required naming convention for functions and member functions, matching the new snake_case public API.
- Moved `BENCHMARKS.md`, `CHANGELOG.md`, and `CONTRIBUTING.md` from the repository root into `docs/` and added them to the documentation site navigation.
- Added a `[patch.crates-io]` section to the root `Cargo.toml` so all workspace bindings resolve `fastLowess` and `lowess` to the local workspace crates during development, replacing the previously-used registry (crates.io) versions.
- Eliminated all local `parse_*` functions that each binding previously duplicated independently. Option parsing and builder application now delegates to `fastLowess::binding_support`, ensuring consistent aliases, validation messages, and behaviour across every language frontend.
- Replaced direct use of `KFold` / `LOOCV` constructor types in the cross-validation path with `binding_support::apply_cross_validation`.

**lowess and fastLowess:**

- Added `Lowess<T>`, `StreamingLowess<T>`, and `OnlineLowess<T>` type aliases as the primary user-facing constructors (e.g. `StreamingLowess::new().chunk_size(50).build()`). Mode-specific builder methods (`chunk_size`, `overlap`, `window_capacity`, `min_points`, `update_mode`) are now called directly on the type alias rather than after `.adapter()`.
- Made `BatchLowessBuilder`, `StreamingLowessBuilder`, and `OnlineLowessBuilder` internal-only: all public setter methods have been removed from these types. All smoothing configuration now flows through `LowessBuilder<T, Mode>` (exposed via the type aliases above). This is a **breaking change** for any code that called setter methods on an adapter builder directly.
- Changed all enum-typed builder methods to accept strings instead: `weight_function`, `robustness_method`, `scaling_method`, `boundary_policy`, `zero_weight_fallback`, `merge_strategy`, and `update_mode` now take `impl IntoEnum<T>` (accepting both enum variants and strings such as `.weight_function("tricube")`) rather than requiring enum variants to be imported. This is a **breaking change** for any code passing enum variants directly.
- Added a `parse` module to both `lowess` and `fastLowess` defining the `IntoEnum<E>` trait and its macro-generated impls for all enum-typed builder parameters. This allows builder methods to accept either a typed enum value (e.g. `.weight_function(WeightFunction::Tricube)`) or a string (e.g. `.weight_function("tricube")`) interchangeably.
- Replaced the `cross_validate(CVConfig)` builder method (which required importing `KFold` or `LOOCV` types) with a string-based cross-validation API: `.cv_method("kfold")` / `.cv_method("loocv")`, `.cv_k(n)`, `.cv_fractions(vec![...])`, and `.cv_seed(n)`. `KFold` and `LOOCV` are no longer exported from the prelude. This is a **breaking change** for any code using the old `cross_validate` API.
- Added a `binding_support` module providing shared helpers for all language binding frontends: string-to-enum parse functions (`parse_weight_function`, `parse_robustness_method`, `parse_scaling_method`, `parse_boundary_policy`, `parse_zero_weight_fallback`, `parse_merge_strategy`, `parse_update_mode`), matching canonical-string display functions, `BuilderOptionSet` / `TypedBuilderOptionSet` structs, and `apply_builder_options` / `apply_typed_builder_options` / `apply_cross_validation` helpers. This consolidates previously duplicated logic that was scattered across every binding into a single source of truth.

**C++:**

- Renamed all public member functions to snake_case: `make_error()`, `has_value()`, `r_squared()`, `effective_df()`, `residual_sd()`, `x_value()`, `y_value()`, `x_vector()`, `y_vector()`, `standard_errors()`, `confidence_lower()`, `confidence_upper()`, `prediction_lower()`, `prediction_upper()`, `robustness_weights()`, `fraction_used()`, `iterations_used()`, `process_chunk()`, `add_points()`.
- Replaced `Expected<LowessResult> OnlineLowess::add_points(const std::vector<double>&, const std::vector<double>&)` with `Expected<std::optional<double>> OnlineLowess::add_point(double x, double y)`. The method now processes a single point and returns only that point's smoothed value, or `std::nullopt` if not enough points have been accumulated yet. The underlying C FFI symbol is renamed from `cpp_online_add_points` to `cpp_online_add_point`. This is a **breaking change**.

**Node.js:**

- Renamed all `Diagnostics`, `SmoothOptions`, `StreamingOptions`, and `OnlineOptions` interface fields to snake_case (`r_squared`, `effective_df`, `residual_sd`, `chunk_size`, `merge_strategy`, `window_capacity`, `min_points`, `update_mode`, and all smoothing option fields).
- Renamed binding methods to snake_case: `fit_async`, `process_chunk`, `add_points`.
- Renamed `LowessResultObj` getters to snake_case: `standard_errors`, `confidence_lower`, `confidence_upper`, `prediction_lower`, `prediction_upper`, `robustness_weights`, `cv_scores`, `fraction_used`, `iterations_used`.
- Updated `index.d.ts` to reflect all renamed fields and methods.
- Replaced `add_points(x: Float64Array, y: Float64Array): LowessResultObj` on `OnlineLowess` with `add_point(x: number, y: number): number | null`. The method now processes a single point and returns only that point's smoothed value, or `null` if not enough points have been accumulated yet. This is a **breaking change**.

**WASM:**

- Renamed all JS-facing option keys to snake_case by removing `#[serde(rename = "camelCase")]` attributes from `SmoothOptions`, `StreamingOptions`, and `OnlineOptions`. JSON passed from JavaScript must now use snake_case keys.
- Updated `Diagnostics` getter names to snake_case: `r_squared`, `effective_df`, `residual_sd`.
- Renamed the `update(x: number, y: number)` method on `OnlineLowessWasm` to `add_point(x: number, y: number)`. This is a **breaking change**.

**Python:**

- Renamed the `update(x, y)` method on `OnlineLowess` to `add_point(x, y)` and removed the separate array-based `add_points(x, y)` method. `add_point` processes a single point and returns the smoothed value as `float | None`. This is a **breaking change**.

**R:**

- Replaced `$add_points(x, y)` (vector inputs returning a list result) on `OnlineLowess` with `$add_point(x, y)` (scalar inputs returning `numeric` or `NULL`). The method now processes one point at a time and returns `NULL` until enough points have been accumulated. This is a **breaking change**.

**Julia:**

- Replaced `add_points(online, x::Vector{Float64}, y::Vector{Float64}) :: LowessResult` with `add_point(online, x::Float64, y::Float64) :: Union{Float64, Nothing}`. The function now processes a single point and returns the smoothed value, or `nothing` if not enough points have been accumulated yet. The underlying C FFI symbol is renamed from `jl_online_lowess_add_points` to `jl_online_lowess_add_point`. This is a **breaking change**.

### Fixed

**C++:**

- Fixed remaining `yVector()` call in `testBasicSmoothSerial` that was missed during the snake_case rename (now `y_vector()`).

## 1.3.0

### Added

**Monorepo:**

- Added prerequisites for different bindings and platforms to `CONTRIBUTING.md`
- Updated `docs/assets/diagrams/lowess_smoothing_concept.svg` to correctly illustrate LOWESS concepts (robustness iterations, bisquare re-weighting, outlier downweighting) instead of the generic LOESS algorithm it previously depicted.
- Modified `docs/requirements.txt` to update the versions of the documentation dependencies.
- Improved CI tests and coverage.
- Modified Makefile to be truely cross-platform.
- Added sanitizer check for all bindings and crates.

**lowess:**

- Upgraded `wide` to version 1.4.

**fastLowess:**

- Upgraded `rayon` to version 1.12.
- Upgraded `wgpu` to version 29.0.

**C++:**

- Added dedicated CMake packaging documentation in `bindings/cpp/CMAKE.md` for Windows installation, `find_package(fastlowess CONFIG REQUIRED)`, and build-tree package discovery.

**R:**

- Upgraded `rextendr` scaffold to 0.5.0: bumped `Config/rextendr/version` in `DESCRIPTION` and updated `entrypoint.c` to register the extendr panic hook (`register_extendr_panic_hook()`), so Rust panics now surface as R errors instead of crashing the session.
- Added `dev/fix_rd_style.R` post-processing script to automatically normalize Rd file indentation (to 4 spaces) and wrap long lines (> 80 characters), ensuring compliance with CRAN/pkgcheck stylistic notes.
- Added `bindings/r/_pkgdown.yml` configuration and updated the `Makefile` to use `pkgdown::build_site()`, satisfying the `pkgcheck` requirement for a dedicated documentation website.
- Added automatic copying of shared tests from the project root into the R package within `dev/prepare_cran.sh`, ensuring `R CMD build` is self-contained.
- Added direct `extendr` wrapper coverage tests plus extra validation-path tests in the R package, lifting `covr::package_coverage()` to 100% and clearing the package-level `pkgcheck`/`goodpractice` coverage complaints.

**WASM:**

- Upgraded `oxlint` to 1.63.

**Node.js:**

- Upgraded `oxlint` to 1.63.
- Upgraded `napi-rs/cli` to 3.6.

### Changed

**lowess:**

- Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

**fastLowess:**

- Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

**R:**

- Refactored the R binding validation helpers to reuse `validate_common_args()` and `coerce_nullable()` in production code, split `validate_params()` into smaller helper validators, and consolidated duplicated constructor parameter documentation with `@inheritParams` before regenerating the Rd files.
- Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.
- Abides by new rOpenSci standards.

**Python:**

- Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

**Node.js:**

- Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

**Julia:**

- Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

**WASM:**

- Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

**C++:**

- Removed the legacy snake_case compatibility layer; the public C++ method API now uses camelBack, while variables and constants follow lower_case.
- Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

### Fixed

**Monorepo:**

- Fixed R ASAN tests failing to compile vignettes by passing `--no-build-vignettes` to `rcmdcheck`.
- Upgraded ASAN test environment to use modern `rocker/r-devel-san:latest` image and `RDscript` to resolve outdated `readelf` warnings.
- Fixed `Makefile` idempotency checks on Linux by providing a default `/tmp` fallback for the `TEMP` directory variable.
- Fixed accidental root `Cargo.toml` workspace isolation leaks by adding checked-in `pre-commit` and `pre-push` git hook guards that restore `Cargo.toml.bak` when present and fail loudly if required workspace members are still commented out.
- Added a repo-local `.cargo/config.toml` that sets `CC=clang-cl` for `x86_64-pc-windows-msvc`, fixing Criterion 0.8 benchmark builds on Windows when `cc-rs` would otherwise pick `clang.exe` and fail to link `alloca`.

**R:**

- Added strict pre-flight check for Pandoc in `make r` with a clear error message, as it is strictly required for building R Markdown vignettes on all platforms.
- Fixed `make r` failure on Windows when the native `tar.exe` (bsdtar) doesn't support GNU-specific reproducible build flags by adding an automatic fallback.
- Fixed `FIND: Parameter format not correct` error on Windows by replacing `find` commands with cross-platform `rm -f` wildcard expansions to prevent clashing with Windows' native `find.exe`.
- Fixed local compilation error by automatically installing required Python TOML dependencies (`tomli`, `tomli_w`) in the Makefile's R build target.
- Fixed `make r` vignette build crash by adding the missing `BiocStyle` R dependency to the Makefile's automated installation step.
- Replaced the deprecated `devtools::build_vignettes()` command with `pkgdown::build_articles()` for local vignette previewing, and added `pkgdown` to the development dependencies.
- Fixed `cargo test` and `cargo build` failing on Windows due to the default MSVC linker trying to link against `R.lib` by enforcing `--target x86_64-pc-windows-gnu` for all R-bound Cargo commands.
- Fixed `R CMD check` `WARNING` on Windows caused by R's own `Boolean.h` using a C23 enum underlying type feature: the previous pragma-based suppression triggered a CRAN `NOTE`. Fixed by pre-defining a standard C `Rboolean` enum and setting the `R_EXT_BOOLEAN_H_` header guard in `entrypoint.c` to prevent R's problematic version from loading.
- Fixed `R CMD check` `ERROR` in the testing phase: `test_check()` always resolves tests from `tests/testthat/` inside the package and ignores custom path arguments. Fixed by updating `dev/prepare_cran.sh` to copy shared tests from `tests/r/testthat/` into `bindings/r/tests/testthat/` before `R CMD build`, and simplifying `testthat.R` to a standard `test_check("rfastlowess")` call.
- Fixed missing R packages (`devtools`, `remotes`) in the Makefile's dependency installation step.
- Fixed `%1 is not a valid Win32 application` and `section below image base` DLL linkage errors on Windows by adding `-Wl,--strip-all` to `Makevars.win` `PKG_LIBS`.
- Fixed rOpenSci `pkgcheck` warning by adding documentation website URL to `DESCRIPTION`.
- Fixed CRAN note regarding non-API call `R_NamespaceRegistry` by upgrading `extendr-api` dependency to `0.9.0`.
- Fixed compilation error by providing the `Result` alias that was removed from `extendr_api::prelude` in `0.9.0`.
- Fixed the remaining SRR/pkgcheck findings in the R package by removing dead internal helper paths, reducing `validate_params()` cyclomatic complexity, eliminating duplicated roxygen parameter blocks, and covering the generated `extendr-wrappers.R` dispatch paths.

**Python:**

- Fixed `make python` failing when `ruff` is not installed globally by bootstrapping `ruff` inside the Python virtual environment before formatting and linting.
- Fixed `make python` on Windows by selecting the correct virtual environment activation script (`.venv/Scripts/activate` instead of the Unix-only `.venv/bin/activate`).
- Fixed the Python public API to actually accept documented array-like inputs by coercing `Lowess.fit()`, `StreamingLowess.process_chunk()`, and `OnlineLowess.add_points()` arguments via `np.asarray(..., dtype=np.float64)` before calling the native extension.
- Fixed Python wrapper analyzer issues by switching native extension lookups to runtime imports, avoiding wrapper class name shadowing in `TYPE_CHECKING`, and adding explicit wrapper docstrings.
- Fixed false-positive Pylint warnings in `bindings/python/python/fastlowess/_core.pyi` by marking stub-only ellipsis bodies and signature arguments as intentional.

**C++:**

- Fixed `cbindgen` idempotency check failure by adding automatic installation of the `cbindgen` CLI tool if missing.
- Fixed explicit pointer checks, braces, named constants, and value-semantic result ownerships with compatibility wrappers.
- Fixed all clang-tidy findings.
- Fixed regenerated `fastlowess.h` clang-tidy regressions by normalizing the auto-generated header during the C++ bindings build, so FFI parameter naming and unused generated includes no longer come back after regeneration.
- Fixed `make cpp` on Windows by making C++ symbol-export verification, CMake test execution, DLL runtime resolution, and Unix-specific test steps platform-aware.
- Fixed MSVC `size_t` to `unsigned long` narrowing warnings in the C++ wrapper at the FFI boundary with explicit conversions.
- Fixed C++ CMake package integration by generating and installing `fastlowessConfig.cmake` and related package export files for downstream `find_package` use.

**fastLowess:**

- Fixed GPU execution under `wgpu` 29 by updating instance and pipeline layout setup, separating shader-written indirect dispatch data from the actual indirect dispatch buffer, stabilizing GPU buffer downloads, and correcting batched cross-validation dispatch offsets so the GPU integration test suite passes again.

**Julia:**

- Fixed Windows local Julia runs by exporting an absolute `FASTLOWESS_LIB` path from the `Makefile` and moving DLL discovery in `FastLOWESS.jl` to runtime (`__init__()` plus runtime `ccall`), preventing stale precompiled library paths from being reused.
- Linted the source code.

**WASM:**

- Fixed deprecated JavaScript license-audit warnings by replacing the transient `npx license-checker` usage in the `Makefile` with a repo-local Node.js license summary script that still fails on GPL-family licenses.
- Linted the source code.

**Node.js:**

- Fixed `make nodejs` on Windows when `/bin/bash` could not launch `npm` from `C:/Program Files/nodejs` by using `npm.cmd`/`npx.cmd` in the `Makefile`.
- Fixed deprecated JavaScript license-audit warnings by replacing the transient `npx license-checker` usage in the `Makefile` with a repo-local Node.js license summary script that still fails on GPL-family licenses.
- Linted the source code.

## 1.2.0

### Added

**R:**

- Added new tests.
- Added a reference to the `CONTRIBUTING.md` file.
- Added new examples for the `print` and `plot` methods.
- Added test coverage evaluation.
- Added missing srr tags.

**Node.js:**

- Added advanced License Compliance check.
- Added advanced dependency check.
- Added advanced outdated dependency check.
- Added advanced lock file check.
- Added advanced TypeScript check.

**WASM:**

- Added advanced License Compliance check.
- Added advanced dependency check.
- Added advanced outdated dependency check.
- Added advanced lock file check.
- Added WASM size check.

### Changed

**fastLowess:**

- Updated `wgpu` to v27.0 from v26.0.

**Python:**

- Updated `pyo3` to v0.28 from v0.27.
- Updated `numpy` to v0.28 from v0.27.

**R:**

- Removed the `coerce_params` dead code.
- Spread srr tags to different files and removed extra tags from `srr-stats-standards.R`.

**Node.js:**

- Switched from `eslint` to `oxlint` to remove vulnerabilities.

**WASM:**

- Switched from `eslint` to `oxlint` to remove vulnerabilities.

### Fixed

**Monorepo:**

- Fixed project logo.

**lowess:**

- Fixed documentation.
- Fixed SRR tags.

**Julia:**

- Linted examples.

**Node.js:**

- Fixed vulnerabilities.

**WASM:**

- Fixed vulnerabilities.
- Fixed license.

**C++:**

- Fixed memory leak in `OnlineLowess`.

## 1.1.2

### Added

**fastLowess:**

- Added srr tags

## 1.1.1

### Added

**lowess:**

- Added srr tags

### Fixed

**fastLowess:**

- Fixed memory layout mismatch in the `GpuConfig` struct
- Refactored the `GpuExecutor` initialization in both the engine (`gpu.rs`) and tests (`gpu_tests.rs`) to handle missing hardware/drivers gracefully.
- Improved the global executor lock handling to automatically recover from "poisoned" states. This prevents a single test crash from disabling the entire GPU backend for the remainder of the session.

## 1.1.0

### Added

**lowess:**

- Added `Mean` scaling method (Mean Absolute Deviation)
- Added hooks for custom fitting backends
- Added hooks for delegating boundary handling to the executor

**fastLowess:**

- Added `Mean` scaling method (Mean Absolute Deviation)
- Added support for different kernels to the GPU backend
- Added support for different robustness methods to the GPU backend
- Added support for different scaling methods to the GPU backend
- Added support for different zero weight fallbacks to the GPU backend
- Added support for different boundary policies to the GPU backend
- Added support for auto convergence to the GPU backend
- Added support for predictiona and confidence interval calculation to the GPU backend
- Added support for cross-validation to the GPU backend

### Fixed

**lowess:**

- `FitPassFn` now returns `Result` to allow error propagation from custom fitting backends (e.g. GPU).
- Adapters (Batch, Streaming, Online) now propagate errors from the executor instead of assuming success.
- Fixed a bug where the `Extend` boundary policy was never applied.
- Implemented Coordinate Centering to preserve precision during accumulation.

**fastLowess:**

- Fixed potential integer overflow in GPU engine when dataset size exceeds `u32::MAX`.
- Fixed panic in GPU initialization by propagating errors to the caller.
- Fixed inefficient memory allocation in `fit_all_points_tiled` by reusing scratch buffers across tiles.
- Fixed resource exhaustion in GPU backend by using a global `Mutex` for the executor instead of thread-local storage.

## 1.0.0

### Added

**Julia:**

- Added `mean` scaling method (Mean Absolute Deviation)

**C++:**

- Added `mean` scaling method (Mean Absolute Deviation)

**R:**

- Added `mean` scaling method (Mean Absolute Deviation)
- Implemented `print` and `plot` methods for `LowessResult` objects
- Added srr tags

**Python:**

- Added `mean` scaling method (Mean Absolute Deviation)

**Node.js:**

- Added `mean` scaling method (Mean Absolute Deviation)
- Added JSDoc documentation to `lib.rs` for napi-rs generation
- Added asynchronous support for batch processing

**WASM:**

- Added `mean` scaling method (Mean Absolute Deviation)
- Added an `init_panic_hook` function in `src/lib.rs` to be called by JS users during startup.
- Added JSDoc documentation to `lib.rs`
- Refactored the verbose `Reflect::get` boilerplate using `serde` and `serde-wasm-bindgen`. This allows us to define a Rust struct `SmoothOptions` and have `wasm-bindgen` automatically unpack the JS object into it.

### Changed

**lowess:**

- Refactored the constants to make the library robust safely against custom numeric types.
- Minor improvements to the documentation.

**C++:**

- Replaced exception-based error handling with a type-safe `Expected<T>` result type for all core methods (`fit`, `process_chunk`, `finalize`, `add_points`).
- Refactored the internal FFI layer to use the idiomatic Rust `From` trait for converting result types.
- Updated all C++ examples and tests to use the new `Expected` pattern, aligning the library with modern C++ practices.

**Julia:**

- Wrapped all FFI functions in std::panic::catch_unwind. This ensures that if the Rust library panics (e.g., due to an internal assertion), it will be caught and reported as an error to Julia.

**WASM:**

- Updated `eslint/js`, `eslint`, `globals`, and `eslint-plugin-html` packages to their latest versions.

**Node.js:**

- Updated `eslint/js`, `eslint`, and `globals` packages to their latest versions.

**Python:**

- Wrapped the heavy computation logic in `py.allow_threads` to allow Python to release the GIL during computation.

**R:**

- Return results as `LowessResult` S3 objects instead of raw vectors

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
