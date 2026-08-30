<!-- markdownlint-disable MD024 MD025 -->
# fastLowess (development version)

## Changed

* Consolidated every crate/binding README: merged the "Installation" and "Documentation" sections, replaced GitHub-only alert syntax with plain blockquotes, removed the redundant "API Reference" and "Changelog" sections (each now has its own docs-site page), shortened the "GPU Backend" blurb, and added a "Read more" link to the Concepts page. The top-level repository README is unchanged, since it's only ever viewed on GitHub.
* Renamed the batch adapter's "When to Use" heading to "When to Use Batch Adapter" across every binding/crate's API docs.
* Vendored the [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) theme (v2.4.2) for a modern, sidebar-only cpp Doxygen site with automatic dark mode.
* Added `dev/update_changelogs.py`, which regenerates a per-binding/crate `NEWS.md`/`news.md` from the root `CHANGELOG.md`. Wired into every docs site's navigation, the Rust crates' rustdoc module tree, and every `Makefile` `dev` target.
* Moved the duplicated "GPU Acceleration" section (installation, usage, supported features, feature comparison) out of `api.md` in the C++, Node.js, and Python bindings and the `fastLowess` crate into each one's dedicated `gpu-backend.md` guide, which also gained the "Hardware Requirements" and "Performance Considerations" subsections previously only in `api.md`; `api.md` now links to `gpu-backend.md` with a short blurb instead. Removed the same section from the `lowess` crate's `api.md` entirely, since that crate has no `gpu` Cargo feature or GPU backend to document.
* Consolidated `parameters.md`/the auto-generated `@autodocs` parameter reference across every binding and crate (C++, Julia, Node.js, Python, WASM, `fastLowess`, `lowess`): merged its unique content (fraction/iterations choice guidance, `delta`'s per-adapter default, and an inline `zero_weight_fallback` behavior table) into each `api.md`'s builder/options tables (Julia: into the `Lowess`/`StreamingLowess`/`OnlineLowess` docstrings), and removed `parameters.md` itself along with its docs-site navigation entries, `doc::parameters` rustdoc module, and cross-references (now pointing at `api.md`) — the parameter tables, kernel/robustness/boundary/scaling/merge-strategy option lists, and interval/custom-weights code examples it duplicated already live on their own dedicated pages.

## Fixed

* Fixed `docs.yml` triggering GitHub's "pages build and deployment" once per docs job; per-language jobs now upload artifacts, and a single final `deploy` job pushes to `gh-pages` once per run.
* Fixed `docs.yml`'s reliance on GitHub's legacy branch-based Pages deployment, which auto-triggers an unpinned, GitHub-managed "pages build and deployment" job on every `gh-pages` push (surfacing deprecation warnings, e.g. for Node.js 20, that aren't fixable from this repo). The former `deploy` job is now `build`, which still pushes the merged `_site` to `gh-pages` as a cache for future incremental runs, but publishing now goes through `actions/upload-pages-artifact` and a new `deploy` job using the official `actions/deploy-pages`, which this repo pins directly. Requires the repository's Pages source to be switched to "GitHub Actions" in settings.
* Fixed inline/display LaTeX math rendering as literal text on docs.rs; added a `katex-header.html` that renders it client-side with KaTeX.

# fastLowess 3.1.0

## Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.
* Added a GitHub workflow for running validation scripts.

## Changed

* Split the monolithic `.github/workflows/ci.yml` into seven per-language workflow files: `ci-rust.yml`, `ci-python.yml`, `ci-julia.yml`, `ci-nodejs.yml`, `ci-wasm.yml`, `ci-cpp.yml`, and `ci-r.yml`. Each file carries the relevant `ci` (multi-OS matrix), `asan`, and `gpu` jobs for its language.
* Each crate/binding sub-Makefile now runs `dev/verify_snippets.py --lang <lang>` for its own language as the final step of `make default`. The root `docs-test` target remains as a convenience to run all languages at once.
* Split `dev/verify_snippets.py` into a lean orchestrator and a `dev/runners/` package. Each language has its own module (`python.py`, `julia.py`, `nodejs.py`, `r.py`, `wasm.py`, `rust.py`, `cpp.py`) containing its `run_<lang>()` function and a `skip_reason()` predicate. Shared types (`Snippet`, `RunResult`) and utilities live in `runners/base.py`; the registry (`RUNNERS`, `SKIP_CHECKS`) is exported from `runners/__init__.py`.
* Moved CHANGELOG and CONTRIBUTING guides to project root.
* Updated README files to be binding/crate specific instead of one generic README for all bindings/crates.
* Moved crate documentation from ReadTheDocs to <https://docs.rs/fastLowess>.
* `make fastLowess` (`default:`) now only runs `cargo build`. The full dev workflow moves to `make fastLowess-dev`.

## Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare tool names resolved via `PATH`, matching the existing fix in `bindings/r/src/cargo-config.toml`.

# fastLowess 3.0.0

## Changed

* Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
* Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/lowess/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
* Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/lowess/` → `crates/lowess/tests/lowess/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages and `bindings/r/demo/`.
* Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`, matching `LowessResult`. This is a **breaking change**.
* Disabled wgpu's default `dx12` and `gles` features (keeping `vulkan`/`metal`) — both pulled in Windows DLLs not present on every system, causing `--features gpu` builds to fail to even load rather than just failing to find a GPU adapter.
* Exposed GPU backend in `binding_support.rs`.

# fastLowess 2.0.0

## Added

* Added `iterations_used: Option<usize>` field to `OnlineOutput<T>`, reporting the number of robustness iterations performed when `UpdateMode::Full` is active. Returns `Some(0)` for the degenerate two-point linear fit and `None` when `UpdateMode::Incremental` is used.
* Added `ParseErrors(Vec<LowessError>)` variant to `LowessError`, which collects all string-parse failures that accumulate in the builder and reports them together when `build()` is called.
* Added `"take_first"` and `"take_last"` as accepted string aliases for `MergeStrategy::TakeFirst` and `MergeStrategy::TakeLast`.
* Added `"resmooth"` as an accepted string alias for `UpdateMode::Full` and `"single"` as an alias for `UpdateMode::Incremental`, aligning string-parse behaviour with the `loess-rs` crate.
* Added `custom_weights(Vec<T>)` builder method on `LowessBuilder` (Batch adapter only). Accepts a vector of non-negative per-observation weights that are multiplied into the distance and robustness weights before each local regression, allowing known-bad points to be suppressed (`0.0`) or high-quality measurements to be emphasised.
* Centralized all `impl FromStr` blocks for the seven option enums (`WeightFunction`, `BoundaryPolicy`, `ScalingMethod`, `RobustnessMethod`, `ZeroWeightFallback`, `MergeStrategy`, `UpdateMode`) directly in `api.rs`, consolidating previously scattered implementations into a single source of truth. Parse and canonical-name helpers are exposed via `lowess::internals::alias` (requires `dev` feature), allowing `fastLowess::binding_support` to delegate all string-to-enum parsing through that path.
* Added module-level `defaults.rs` files within each sub-module (`math/`, `algorithms/`, `adapters/`) to centralize default values close to the types they govern, propagating them from a single source of truth to ensure consistency across bindings and crates.

## Changed

* Renamed all public API method and option names from camelCase to snake_case across every binding and all documentation. This is a **breaking change** for all consumers of the C++, Node.js, and WASM APIs.
* Converted all documentation tables to compact single-space format.
* Updated `.clang-tidy` to configure `lower_case` as the required naming convention for functions and member functions, matching the new snake_case public API.
* Moved `BENCHMARKS.md`, `CHANGELOG.md`, and `CONTRIBUTING.md` from the repository root into `docs/` and added them to the documentation site navigation.
* Added a `[patch.crates-io]` section to the root `Cargo.toml` so all workspace bindings resolve `fastLowess` and `lowess` to the local workspace crates during development, replacing the previously-used registry (crates.io) versions.
* Eliminated all local `parse_*` functions that each binding previously duplicated independently. Option parsing and builder application now delegates to `fastLowess::binding_support`, ensuring consistent aliases, validation messages, and behaviour across every language frontend.
* Replaced direct use of `KFold` / `LOOCV` constructor types in the cross-validation path with `binding_support::apply_cross_validation`.
* Split the Julia release CI workflow into two separate workflows: `release-julia-jll.yml` (triggered on release, opens the Yggdrasil PR) and `release-julia-register.yml` (manual dispatch, triggers JuliaRegistrator once the JLL PR is merged).
* Major documentation improvements.
* Added `Lowess<T>`, `StreamingLowess<T>`, and `OnlineLowess<T>` type aliases as the primary user-facing constructors (e.g. `StreamingLowess::new().chunk_size(50).build()`). Mode-specific builder methods (`chunk_size`, `overlap`, `window_capacity`, `min_points`, `update_mode`) are now called directly on the type alias rather than after `.adapter()`.
* Made `BatchLowessBuilder`, `StreamingLowessBuilder`, and `OnlineLowessBuilder` internal-only: all public setter methods have been removed from these types. All smoothing configuration now flows through `LowessBuilder<T, Mode>` (exposed via the type aliases above). This is a **breaking change** for any code that called setter methods on an adapter builder directly.
* Changed all enum-typed builder methods to accept strings instead: `weight_function`, `robustness_method`, `scaling_method`, `boundary_policy`, `zero_weight_fallback`, `merge_strategy`, and `update_mode` now take `impl IntoEnum<T>` (accepting both enum variants and strings such as `.weight_function("tricube")`) rather than requiring enum variants to be imported. This is a **breaking change** for any code passing enum variants directly.
* Inlined the `IntoEnum<E>` trait and its macro-generated impls for all enum-typed builder parameters directly into `api.rs` (`lowess`) and `binding_support.rs` (`fastLowess`), eliminating a previously separate `parse` module. This allows builder methods to accept either a typed enum value (e.g. `.weight_function(WeightFunction::Tricube)`) or a string (e.g. `.weight_function("tricube")`) interchangeably.
* Replaced the `cross_validate(CVConfig)` builder method (which required importing `KFold` or `LOOCV` types) with a string-based cross-validation API: `.cv_method("kfold")` / `.cv_method("loocv")`, `.cv_k(n)`, `.cv_fractions(vec![...])`, and `.cv_seed(n)`. `KFold` and `LOOCV` are no longer exported from the prelude. This is a **breaking change** for any code using the old `cross_validate` API.
* Added a `binding_support` module providing shared helpers for all language binding frontends: string-to-enum parse functions (`parse_weight_function`, `parse_robustness_method`, `parse_scaling_method`, `parse_boundary_policy`, `parse_zero_weight_fallback`, `parse_merge_strategy`, `parse_update_mode`), matching canonical-string display functions, `BuilderOptionSet` / `TypedBuilderOptionSet` structs, and `apply_builder_options` / `apply_typed_builder_options` / `apply_cross_validation` helpers. This consolidates previously duplicated logic that was scattered across every binding into a single source of truth.
* Renamed the internal `auto_convergence` struct field to `auto_converge` on `BatchLowessBuilder`, `OnlineLowessBuilder`, `StreamingLowessBuilder`, and the executor config types, making the field name consistent with the existing `auto_converge()` setter method. This is a **breaking change** for any code that accessed these fields directly.
* Changed `build()` to wrap all accumulated string-parse errors in a `LowessError::ParseErrors(Vec<LowessError>)` value instead of surfacing only the first error. This is a **breaking change** for code that matched on `LowessError::InvalidOption` as the error returned from `build()`.
* Made the `IntoEnum<E>` trait `pub(crate)` in both `lowess` and `fastLowess`, restricting it to crate-internal use. Callers do not need to name this trait; builder methods continue to accept both enum variants and string literals unchanged.
* Updated `wide` dependency to v1.5, `wgpu` to v30.0, and `pollster` to v1.0.
* `Lowess`, `StreamingLowess`, and `OnlineLowess` are now dedicated wrapper structs around `LowessBuilder<f64>` with string-accepting forwarding methods, rather than type aliases re-exported from the base `lowess` crate. Each wrapper's `build()` delegates to the corresponding parallel adapter and defaults to parallel execution. This is a **breaking change**: replace `.adapter(Batch).build()` with `.build()`, `Lowess::new().adapter(Streaming)` with `StreamingLowess::new()`, and `Lowess::new().adapter(Online)` with `OnlineLowess::new()`.
* The `fastLowess` prelude now exports only `{Lowess, LowessError, LowessResult, OnlineLowess, StreamingLowess}`, removing `LowessBuilder`, `Adapter::{Batch, Online, Streaming}`, and `Backend::{CPU, GPU}`. This is a **breaking change** for code that relied on those names being in scope via `use fastLowess::prelude::*`.

# fastLowess 1.3.0

## Added

* Added prerequisites for different bindings and platforms to `CONTRIBUTING.md`
* Updated `docs/assets/diagrams/lowess_smoothing_concept.svg` to correctly illustrate LOWESS concepts (robustness iterations, bisquare re-weighting, outlier downweighting) instead of the generic LOESS algorithm it previously depicted.
* Modified `docs/requirements.txt` to update the versions of the documentation dependencies.
* Improved CI tests and coverage.
* Modified Makefile to be truely cross-platform.
* Added sanitizer check for all bindings and crates.
* Upgraded `rayon` to version 1.12.
* Upgraded `wgpu` to version 29.0.

## Changed

* Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

## Fixed

* Fixed R ASAN tests failing to compile vignettes by passing `--no-build-vignettes` to `rcmdcheck`.
* Upgraded ASAN test environment to use modern `rocker/r-devel-san:latest` image and `RDscript` to resolve outdated `readelf` warnings.
* Fixed `Makefile` idempotency checks on Linux by providing a default `/tmp` fallback for the `TEMP` directory variable.
* Fixed accidental root `Cargo.toml` workspace isolation leaks by adding checked-in `pre-commit` and `pre-push` git hook guards that restore `Cargo.toml.bak` when present and fail loudly if required workspace members are still commented out.
* Added a repo-local `.cargo/config.toml` that sets `CC=clang-cl` for `x86_64-pc-windows-msvc`, fixing Criterion 0.8 benchmark builds on Windows when `cc-rs` would otherwise pick `clang.exe` and fail to link `alloca`.
* Fixed GPU execution under `wgpu` 29 by updating instance and pipeline layout setup, separating shader-written indirect dispatch data from the actual indirect dispatch buffer, stabilizing GPU buffer downloads, and correcting batched cross-validation dispatch offsets so the GPU integration test suite passes again.

# fastLowess 1.2.0

## Changed

* Updated `wgpu` to v27.0 from v26.0.

## Fixed

* Fixed project logo.

# fastLowess 1.1.2

## Added

* Added srr tags

# fastLowess 1.1.1

## Fixed

* Fixed memory layout mismatch in the `GpuConfig` struct
* Refactored the `GpuExecutor` initialization in both the engine (`gpu.rs`) and tests (`gpu_tests.rs`) to handle missing hardware/drivers gracefully.
* Improved the global executor lock handling to automatically recover from "poisoned" states. This prevents a single test crash from disabling the entire GPU backend for the remainder of the session.

# fastLowess 1.1.0

## Added

* Added `Mean` scaling method (Mean Absolute Deviation)
* Added support for different kernels to the GPU backend
* Added support for different robustness methods to the GPU backend
* Added support for different scaling methods to the GPU backend
* Added support for different zero weight fallbacks to the GPU backend
* Added support for different boundary policies to the GPU backend
* Added support for auto convergence to the GPU backend
* Added support for predictiona and confidence interval calculation to the GPU backend
* Added support for cross-validation to the GPU backend

## Fixed

* Fixed potential integer overflow in GPU engine when dataset size exceeds `u32::MAX`.
* Fixed panic in GPU initialization by propagating errors to the caller.
* Fixed inefficient memory allocation in `fit_all_points_tiled` by reusing scratch buffers across tiles.
* Fixed resource exhaustion in GPU backend by using a global `Mutex` for the executor instead of thread-local storage.

# fastLowess 0.99.9

## Changed

* Bump rust version to 1.88 for better stability
* Change function-based builder pattern in the bindings to class-based builder pattern, allowing true streaming and online processing
* Improve API docs

# fastLowess 0.99.7

## Fixed

* Fix README file links
* Fix Makefile bug with R versioning

# fastLowess 0.99.6

## Fixed

* Fix README file formats and links

# fastLowess 0.99.5

## Changed

* Reduced package size significantly by removing unnecessary dev files and docs from the final package.
* Implemented comprehensive Cargo workspace inheritance pattern
* Unified MSRV to 1.85.0
* Centralized all metadata (version, authors, edition, license, etc.) in root `Cargo.toml`
* All crates now use `workspace = true` for shared configuration
* Created unified `README.md` for all crates/packages
* Created unified `CHANGELOG.md` for all crates/packages
* Created unified `LICENSE` for all crates/packages
* Created unified `.gitignore` for all crates/packages
* Added comprehensive badges from all packages

# fastLowess 0.4.0

## Added

* Zero-allocation parallel fitting via `fit_all_points_parallel`
* Parallel CV memory reuse via `cv_pass_parallel`
* Refined delta optimization for tied x-values
* Parallel anchor precomputation for large datasets
* Cache-oblivious tile-based processing

## Changed

* Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
* Updated `lowess` dependency to v0.7.0
* Implemented thread-local `GpuExecutor` persistence
* Added intelligent buffer capacity management for GPU
* Refactored GPU compute kernel with shared memory tiling

# fastLowess 0.3.0

## Added

* `cpu` (default) and `gpu` Cargo features
* GPU execution engine in `src/engine/gpu.rs`
* `fit_pass_gpu` function for GPU-accelerated processing
* `backend()` setter method to all builders
* Tests for GPU engine and parallel execution consistency

## Changed

* Renamed builders: `Extended*LowessBuilder` → `Parallel*LowessBuilder`
* Migrated `parallel` field to core `lowess` crate
* Updated `lowess` dependency to v0.6.0
* Made `ndarray` and `rayon` optional dependencies

## Removed

* `.cargo/config.toml`
* Type exports from `prelude` that shadowed std types
* Sequential, parallel, and ndarray adaptors

# fastLowess 0.2.0

## Changed

* Replaced linear scan with binary search in `compute_anchor_points`
* Eliminated per-iteration division in `interpolate_gap`
* Aligned with `lowess` crate v0.5.3 optimizations

# fastLowess 0.1.0

## Added

* Initial release with parallel execution support

For the full changelog, see:
<https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md>
