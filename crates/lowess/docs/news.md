<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md).

## lowess (development version)

### Added

* Added self-contained R and original Cleveland LOWESS references under `validation/reference/`, with provenance, dependency, precision, and build notes.
* Added `LowessBuilder::outputs(names)` as a grouped replacement for the individual output toggles. Unknown names are collected and reported together by `.build()`.
* Added grouped cross-validation configuration through `CVBuilder` and `.cv(...)`. `CVBuilder` is in the prelude; the internal `CVOptions` result type remains at the crate root.
* Added `PredictBuilder::outputs(names)` in both Rust crates, supporting `"se"` and `"derivative"` as a grouped replacement for `.return_se()` and `.return_derivative()`.

### Changed

* Updated the vendored `doxygen-awesome-css` theme to v2.5.0 and the Hugo docs build to v0.166.0.
* Removed 62 redundant `#[doc(hidden)]` attributes from private engine/adapter modules; public-API annotations remain unchanged.
* Replaced the cross-validation candidate-fit `unwrap()` with error propagation through sequential CPU, parallel CPU, and GPU CV paths.
* Replaced the `iteration_loop_with_callback` clippy suppression with a typed options bundle for its iteration controls and callbacks.
* Marked `WeightFunction` as non-exhaustive and made GPU handling reject unsupported future kernels explicitly.

### Fixed

* Skip comparison snippets when the optional `statsmodels` dependency is unavailable instead of failing verification.
* Fixed C++ release CI staging the tracked Spack recipe despite the repository's broad `spack/` ignore rule.
* Matched R's effective-zero robustness guard: stop when `6 * median(abs(residuals)) < 1e-7 * mean(abs(residuals))`; centered MAD retains its separate fallback.
* Matched Cleveland/R's local-linear degeneracy rule: suppress the slope when weighted local x-spread is below `0.001 * (max(x) - min(x))`.
* Removed the absolute `1e-12` bisquare scale floor so roundoff-sized residuals are reweighted at their actual scale.
* Matched R's normalized adjusted-weight fitted-value accumulation without parity-, sparsity-, or response-scale-specific branches.
* Separated local-weight adjustment and fitted-response accumulation into R's original loop order, avoiding platform-dependent cancellation in sparse robust fits.
* Separated robustness scale scratch storage from local kernel weights so median selection cannot contaminate the next R-equivalent smoothing pass.
* Matched R's `w * ((x - mean_x) * (x - mean_x))` spread parenthesization, preserving cancellation-scale endpoint fits during robust passes.
* Matched R's even-length `cmad = 3 * (lower + upper)` operation order instead of scaling an averaged median.
* Extended local kernel scans beyond the nominal right window edge until R's `0.999 * h` cutoff, matching `lowest()` on asymmetric neighborhoods.
* Matched R's `1e-7` span-truncation adjustment instead of rounding near-integer neighborhoods with `1e-5`.
* Corrected the high-iteration MAR regression fixture to the current `stats::lowess` output.
* Matched Cleveland/R's delta interpolation order (`alpha * y1 + (1 - alpha) * y0`) so interpolated roundoff residuals preserve the correct robustness-cycle phase.
* Preserved sparse-fit robustness cycles found by `quickcheck`; long-iteration comparisons allow bounded floating-point drift while fixed regressions pin each branch.

## lowess 4.1.0

### Added

* Added musl release binaries for Python, C++, Go, and Julia.
* Added bundled native libraries for the Java binding across 8 platforms, with runtime musl detection and auto-extraction.
* Added Online `return_se`/`confidence_intervals`/`prediction_intervals` support, with `OnlineOutput` now populating `standard_error` and the four interval bounds in `Full` mode. Using them without `update_mode("full")` now fails at `.build()` with `LowessError::StandardErrorRequiresFullUpdateMode`.
* Added Streaming `return_se`/`confidence_intervals`/`prediction_intervals` support, computed per chunk and merged across overlaps. `StreamingBuffer` now carries interval scratch state through `process_chunk()`/`finalize()`.
* Added `return_derivative` to the Batch, Streaming, and Online builders, exposing each point's local slope via `LowessResult::derivative` or `OnlineOutput::derivative`.
* Added out-of-sample prediction to Batch via `.retain_model(true)` and `Predict::call()`, with configurable standard errors, intervals, derivative output, and extrapolation.

### Changed

* Hoisted fully-qualified imports to top-level `use` statements across crates and bindings.
* Flattened the `tests/lowess/` directories into `tests/` directly: each test file is now its own independent integration test binary instead of a submodule of a shared `main.rs`. No test behavior changes.
* Bumped the vendored KaTeX CDN version from `0.18.5` to `0.18.7`, updating SRI hashes to match.

### Fixed

* `dev/bump_version.py` now also updates the Go module's `/vN` major-version-suffix path across `go.mod` files, doc snippets, the doc-snippet runner, and README/docs badges whenever a version bump crosses a major version boundary, so this doesn't regress on the next major release.
* `dev/bump_version.py` now also updates the Maven dependency example version in `bindings/java/docs/modules/ROOT/pages/introduction/installation.adoc`, which was previously left stale after a version bump.
* Changed the `iterations` default for `OnlineLowess` from `3` to `0` across every binding (R, Python, Julia, C++, Go, Java, Node.js, WASM), matching the default `update_mode = "incremental"` non-robust single-point fit; robustness iterations now require `update_mode = "full"`.
* Cleaned up `lowess::prelude` by removing leaked builder and adapter markers.
* Fixed standard errors collapsing to exactly `0` (and confidence intervals collapsing to a `1e-12` fallback width) for observations whose robustness weight reaches zero: `compute_se` now derives the leverage from the local linear design's equivalent kernel rather than the point's own robustness-weighted kernel, so down-weighted outliers keep a positive standard error.
* Fixed systematic over-estimation (roughly 10-35% too wide) of confidence-interval standard errors: `compute_se` used the local-constant leverage `w_i / sum(w)` and divided the weighted residual sum of squares by `sum(w) - 2`, mixing a kernel-weight sum with a parameter count. It now uses the exact local-linear variance multiplier `e1'(X'WX)^-1 (X'W^2 X) (X'WX)^-1 e1` (the squared equivalent-kernel norm `sum_k l_k^2`) and the kernel-corrected residual degrees of freedom `sum(w) - 2 + sum(w^2)/sum(w)`. Reported standard errors now match the Monte-Carlo standard error to within a few percent on linear truth at typical fractions (new `calibration_tests` regression test).
* Fixed `fraction >= 1.0` (the global OLS branch) returning a vector of zeros for standard errors, which collapsed every confidence interval to the `1e-12` fallback width; it now computes OLS standard errors using the classical simple-linear-regression formula `sigma_hat * sqrt(1/n + (x0 - xbar)^2 / Sxx)`, matching `stats::lm`'s `se.fit`.
* Removed unused `pub use` re-exports and updated the few callers that used them.
* Fixed `OnlineLowess`'s default `"incremental"` mode silently ignoring robustness iterations: `iterations > 0` is now rejected at `.build()` (`RobustnessIterationsRequireFullUpdateMode`) unless `update_mode("full")` is set, the Online `iterations` default is now `0`, and `iterations_used` is reported even without auto-convergence.
* Fixed seeded k-fold CV (`cv_seed` set) producing inflated scores and selecting the wrong fraction: `interpolate_prediction_batch` used a monotone scan pointer that can't rewind for the shuffled (unordered) test fold, so it now locates each query point via binary search (matching the LOOCV interpolator).
* Fixed the WLS solver silently zeroing the slope for small-magnitude `x`: `fit_wls` used an absolute degeneracy tolerance (`1e-7`) on the centred weighted x-variance, so any dataset whose x-range was below roughly `1e-4` was fitted as a local mean instead of a local line (interior derivatives came out as `0`). The tolerance is now relative to the design's own x-scale. The same absolute-tolerance defect in the `fraction >= 1.0` global OLS path (`fit_ols` and `ols_std_errors`) is fixed the same way.
* Fixed k-fold cross-validation aggregating the mean of per-fold RMSEs instead of pooling: it now pools every test point's squared error and takes one square root, matching LOOCV. Previously `k`-fold with `k == n` (identical to leave-one-out) returned the mean absolute error instead of the RMSE and disagreed with `cv_method("loocv")`.

## lowess 4.0.0

### Added

* Added an "Ideas for Contribution" section to `CONTRIBUTING.md`, listing concrete Batch/Streaming/Online adapter feature gaps (out-of-sample prediction, exposing local slope/derivative, adaptive fraction selection, STL-style decomposition, bootstrap intervals, concurrent chunk processing, checkpointable streaming state, populating `OnlineOutput.standard_error`, time-based window eviction, configurable warm-up) to invite contributions.
* `dev/bump_version.py` now also updates the example crate version in `CONTRIBUTING.md`'s "Individual crate Cargo.toml" snippet.
* Every GPU installer (Python, Node.js, R, Julia, C++, Java, Go) now accepts a local path to an already-built GPU artifact, installing from it directly instead of downloading from a GitHub Release — useful for testing the installer itself or installing an unreleased build. `ci-*.yml`'s `gpu` jobs now build the `gpu` feature locally and exercise this path end-to-end instead of depending on a matching published release existing.
* `release-gpu.yml` now uploads every GPU artifact (across all languages and all versions) to a single perpetual `gpu-builds` release instead of the release page for the version just published, so version release pages stay uncluttered; each asset's filename embeds its source version instead. Every installer's download URL updated to match.
* Added `.return_sorted()` to the batch builder, to return results sorted ascending by `x` instead of input order. Default `false`.
* Added a `missing` option (`"error"` default, or `"drop"`) controlling non-finite (NaN/Inf) `x`/`y` handling: Batch/Streaming drop non-finite rows (and matching `custom_weights`); Online skips non-finite points, returning `Ok(None)`. Length mismatches always error.
* Added `release-rust.yml` to publish to crates.io on release.

### Changed

* Go doc-snippet verification now batch-builds every snippet in one `go build ./...` under a persistent module instead of one `go run` per snippet, then runs the binaries concurrently; `verify_snippets.py`'s `BATCH_RUNNERS` dispatch (previously Rust-only) now covers `go` too.
* C++ doc-snippet verification now resolves the compiler/library/MSVC setup once, then compiles+links+runs every snippet concurrently instead of one at a time. Fixed an MSVC race from concurrent `cl.exe` invocations colliding on a shared `snippet.obj` by giving each snippet its own `/Fo` output and `cwd`.
* Removed the dead, unreachable `compute_residuals`/`parallel`/`backend` fields from `OnlineLowessBuilder`; `StreamingLowessBuilder` lost its unused `backend` field too.
* Breaking change: `Streaming::convert()` no longer resolves `overlap` to a flat `500` when unset; it now resolves dynamically to `chunk_size / 10` (clamped to `[1, chunk_size - 10]`). This affects callers relying on the previous flat default with a customized `chunk_size`.
* Improved API documentation for the lowess crate significantly.

### Fixed

* Fixed `CONTRIBUTING.md` stating a stale Go prerequisite (`1.21+`, actually `1.23+` per `go.mod`/CI), an inaccurate `air` auto-install target (claimed `make r`, actually `make r-dev`), and a stale example crate version (`2.0.0`) in the Workspace Structure section.

## lowess 3.2.1

### Added

* Added `dev/bump_version.py --version X.Y.Z` to bump every crate/binding's version files, `CITATION.cff`, and the Spack recipe in one pass (supports `--dry-run`); now also bumps `Project.toml`'s `fastlowess_jll` compat floor (safe pre-publish since `make julia-dev`/CI relax it to an OR-list at test-time).
* Added an optional `commit` input to every release workflow's `workflow_dispatch` trigger, to pin the built commit for manual runs.
* Added an `aarch64-pc-windows-gnullvm` linker entry to the root `.cargo/config.toml`, matching the existing `x86_64-pc-windows-gnu` one; makes local arm64 Windows builds work without a manual env var.

### Changed

* Added four new pins to `dev/check_pinned_versions.py`: R's `rextendr`/`roxygen2` versions and the vendored KaTeX CDN version in both Rust crates.
* Changed `check-versions.yml` to open/update a GitHub issue instead of failing CI when a pin goes stale or unreachable.
* Bumped the vendored KaTeX CDN version from `0.18.4` to `0.18.5`, updating SRI hashes to match.

### Fixed

* Fixed a handful of `R²`/`O(n²)` Unicode superscripts the earlier ASCII-fication pass missed (added/edited after it ran) — R tests and several `lowess` crate doc-comments — replaced with `R2`/`O(n^2)`.
* Fixed `release-conda.yml`'s `sed` patterns for the feedstock's new rattler-build `recipe/recipe.yaml` format, removing now-dead R-package-name-fix/Python-dependency-injection/`build_r.sh` steps.
* Fixed `dev/check_links.py` false-flagging valid R vignette links to a sibling's rendered `.html` (R CMD build renders `.Rmd`→`.html`) as broken.
* Fixed every binding's/crate's docs and doc-comments describing `LowessResult.x` (and equivalents) as "Sorted x values"; it's actually returned in the same order as the input `x` (the algorithm sorts internally, then un-sorts every output field back to the original order). Also strengthened Python's `test_unsorted_input` to assert this instead of only checking output length.
* Fixed `.github/dependabot.yml`'s `cargo` entry for `/bindings/r/src`, which could never succeed: its `fastLowess = { path = "vendor/fastLowess" }` path dependency is only committed as `vendor.tar.xz`, never as loose files Dependabot can read. Removed the entry and added `extendr-api`'s version to `dev/check_pinned_versions.py` instead, which also uncovered and fixed a version-comparison bug there: comparing raw tuples treated a shorthand pin like `"0.9"` as older than `"0.9.0"` due to tuple-length tiebreaking; now padded to equal length first.

## lowess 3.2.0

### Added

* Added `dev/add-readme-to-docs.py`, which auto-detects the Hugo (Go, Java) vs Starlight (Node.js, WASM) docs-site flavor and embeds `README.md` accordingly; wired into the corresponding Makefiles and `package.json` scripts.
* Added `.github/dependabot.yml`, covering every dependency ecosystem in the repo (`github-actions`, `cargo`, `npm`, `pip`, `maven`, `gomod`). Each directory is grouped so all its updates, including majors, land in a single weekly PR.
* Added `dev/check_pinned_versions.py` and a weekly `.github/workflows/check-versions.yml`, which check hardcoded tool/library version pins that Dependabot can't see (Corrosion's CMake `FetchContent` tag, the vendored doxygen-awesome-css theme, the Checkstyle jar, golangci-lint, and Hugo) against their latest GitHub release and fail CI if any are outdated. Read-only: it never opens PRs or edits files itself.

### Changed

* Modified `verify_snippets.py` to verify snippets and also add the output of the snippets to the markdown file.
* Added a `large` benchmark category (n = 50000) to `benchmarks/rfastlowess.R` and `benchmarks/stats_lowess.R`, since every existing category ran in well under 100ms. Covers 4 scenarios (`large_delta_0`, `large_delta_0.1`, `large_high_iter`, `large_high_fraction`) stressing `delta`, iteration count, and fraction. `benchmarks/compare.py`'s plot grid grew from 5x2 to 7x2 to fit them.
* Added `.gitattributes`, normalizing all text files to LF line endings (`* text=auto eol=lf`) and marking binary formats (images, archives, compiled libraries, `.rds`/`.RData`, etc.) so Git never treats them as text.
* Updated `wide` to v1.7.
* Standardized the lowess crate documentation: consolidated README setup content and parameter guidance, renamed the batch-adapter heading, replaced flowcharts with decision tables, standardized API examples and page structure, and converted superscripts to ASCII.
* Removed the GPU acceleration section from the API docs because the lowess crate has no GPU feature.
* Updated `Diagnostics` display output to use `R2` instead of the Unicode superscript form.

### Fixed

* Fixed `docs.yml` triggering GitHub's "pages build and deployment" once per docs job; per-language jobs now upload artifacts, and a single final `deploy` job pushes to `gh-pages` once per run.
* Fixed `docs.yml`'s reliance on GitHub's legacy branch-based Pages deployment, which auto-triggered an unpinned "pages build and deployment" job on every `gh-pages` push. The former `deploy` job is now `build` (still pushes `_site` to `gh-pages` as a cache); publishing now goes through `actions/upload-pages-artifact` and a new `deploy` job using `actions/deploy-pages`. Requires the repo's Pages source set to "GitHub Actions".
* Fixed every benchmark category in `benchmarks/rfastlowess.R` failing with `attempt to apply non-function`: it called the R6-style `model$fit(x, y)`, but `fit` is an S3 generic (`fit(model, x, y)`), not a field on the `Lowess` object. Also fixed `benchmarks/stats_lowess.R` resolving its `output/` directory relative to the current working directory instead of the script's own location (unlike `rfastlowess.R`, which already did this correctly), so results could land outside `benchmarks/output/` depending on how the script was invoked.
* Fixed inline/display LaTeX math rendering as literal text on docs.rs; added a `katex-header.html` that renders it client-side with KaTeX.
* Fixed every cross-reference link across the `lowess`/`fastLowess` crate docs leading nowhere: these pages are embedded into rustdoc via `#![doc = include_str!(...)]`, so plain relative links render verbatim instead of resolving. Converted them to proper intra-doc links (e.g. `crate::doc::concepts`), validated with `cargo doc --all-features -D warnings`.
* Fixed the Handling Outliers quickstart example in the lowess crate: increased `fraction` from `0.5` to `0.7` because the six-point example otherwise fit the injected outlier exactly instead of downweighting it.
* Capped the lowess crate Detecting Outliers example output at five lines.

## lowess 3.1.0

### Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.
* Added a GitHub workflow for running validation scripts.

### Changed

* Split the monolithic `.github/workflows/ci.yml` into seven per-language workflow files: `ci-rust.yml`, `ci-python.yml`, `ci-julia.yml`, `ci-nodejs.yml`, `ci-wasm.yml`, `ci-cpp.yml`, and `ci-r.yml`. Each file carries the relevant `ci` (multi-OS matrix), `asan`, and `gpu` jobs for its language.
* Each crate/binding sub-Makefile now runs `dev/verify_snippets.py --lang <lang>` for its own language as the final step of `make default`. The root `docs-test` target remains as a convenience to run all languages at once.
* Split `dev/verify_snippets.py` into a lean orchestrator and a `dev/runners/` package. Each language has its own module (`python.py`, `julia.py`, `nodejs.py`, `r.py`, `wasm.py`, `rust.py`, `cpp.py`) containing its `run_<lang>()` function and a `skip_reason()` predicate. Shared types (`Snippet`, `RunResult`) and utilities live in `runners/base.py`; the registry (`RUNNERS`, `SKIP_CHECKS`) is exported from `runners/__init__.py`.
* Moved `CHANGELOG.md` and `CONTRIBUTING.md` to the repository root.
* Moved crate documentation from ReadTheDocs to <https://docs.rs/lowess>.
* `make lowess` (`default:`) now only runs `cargo build`. The full dev workflow moves to `make lowess-dev`.
* Updated the lowess crate README to be package-specific instead of using the generic shared README.

### Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare tool names resolved via `PATH`, matching the existing fix in `bindings/r/src/cargo-config.toml`.

## lowess 3.0.0

### Added

* Added `See: ...` cross-reference links after option headings in the lowess crate API docs, pointing to the corresponding user guide.

### Fixed

* Fixed `docs/api/rust.md` showing Rust enum variants instead of the string option values accepted by the API.

### Changed

* Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
* Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/lowess/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
* Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/lowess/` → `crates/lowess/tests/lowess/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages and `bindings/r/demo/`.
* Breaking change: Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`, matching `LowessResult`.
* Updated `wide` to v1.6.
* Split Streaming/Online content from the lowess crate API reference into dedicated pages, moved tutorials into the user-guide use-cases section, and standardized API examples with expected output.

## lowess 2.0.0

### Added

* Added `iterations_used: Option<usize>` field to `OnlineOutput<T>`, reporting the number of robustness iterations performed when `UpdateMode::Full` is active. Returns `Some(0)` for the degenerate two-point linear fit and `None` when `UpdateMode::Incremental` is used.
* Added `ParseErrors(Vec<LowessError>)` variant to `LowessError`, which collects all string-parse failures that accumulate in the builder and reports them together when `build()` is called.
* Added `"take_first"` and `"take_last"` as accepted string aliases for `MergeStrategy::TakeFirst` and `MergeStrategy::TakeLast`.
* Added `"resmooth"` as an accepted string alias for `UpdateMode::Full` and `"single"` as an alias for `UpdateMode::Incremental`, aligning string-parse behaviour with the `loess-rs` crate.
* Added `custom_weights(Vec<T>)` builder method on `LowessBuilder` (Batch adapter only). Accepts a vector of non-negative per-observation weights that are multiplied into the distance and robustness weights before each local regression, allowing known-bad points to be suppressed (`0.0`) or high-quality measurements to be emphasised.
* Centralized all `impl FromStr` blocks for the seven option enums (`WeightFunction`, `BoundaryPolicy`, `ScalingMethod`, `RobustnessMethod`, `ZeroWeightFallback`, `MergeStrategy`, `UpdateMode`) directly in `api.rs`, consolidating previously scattered implementations into a single source of truth. Parse and canonical-name helpers are exposed via `lowess::internals::alias` (requires `dev` feature), allowing `fastLowess::binding_support` to delegate all string-to-enum parsing through that path.
* Added module-level `defaults.rs` files within each sub-module (`math/`, `algorithms/`, `adapters/`) to centralize default values close to the types they govern, propagating them from a single source of truth to ensure consistency across bindings and crates.

### Changed

* Breaking change: Renamed all public API method and option names from camelCase to snake_case across every binding and all documentation. The public APIs for C++, Node.js, and WASM have changed.
* Converted all documentation tables to compact single-space format.
* Updated `.clang-tidy` to configure `lower_case` as the required naming convention for functions and member functions, matching the new snake_case public API.
* Moved `BENCHMARKS.md`, `CHANGELOG.md`, and `CONTRIBUTING.md` from the repository root into `docs/` and added them to the documentation site navigation.
* Added a `[patch.crates-io]` section to the root `Cargo.toml` so all workspace bindings resolve `fastLowess` and `lowess` to the local workspace crates during development, replacing the previously-used registry (crates.io) versions.
* Eliminated all local `parse_*` functions that each binding previously duplicated independently. Option parsing and builder application now delegates to `fastLowess::binding_support`, ensuring consistent aliases, validation messages, and behaviour across every language frontend.
* Replaced direct use of `KFold` / `LOOCV` constructor types in the cross-validation path with `binding_support::apply_cross_validation`.
* Split the Julia release CI workflow into two separate workflows: `release-julia-jll.yml` (triggered on release, opens the Yggdrasil PR) and `release-julia-register.yml` (manual dispatch, triggers JuliaRegistrator once the JLL PR is merged).
* Major documentation improvements.
* Added `Lowess<T>`, `StreamingLowess<T>`, and `OnlineLowess<T>` type aliases as the primary user-facing constructors (e.g. `StreamingLowess::new().chunk_size(50).build()`). Mode-specific builder methods (`chunk_size`, `overlap`, `window_capacity`, `min_points`, `update_mode`) are now called directly on the type alias rather than after `.adapter()`.
* Breaking change: Made `BatchLowessBuilder`, `StreamingLowessBuilder`, and `OnlineLowessBuilder` internal-only: all public setter methods have been removed from these types. All smoothing configuration now flows through `LowessBuilder<T, Mode>` (exposed via the type aliases above). Callers that used setters on adapter builders must update their code.
* Breaking change: Changed all enum-typed builder methods to accept strings instead: `weight_function`, `robustness_method`, `scaling_method`, `boundary_policy`, `zero_weight_fallback`, `merge_strategy`, and `update_mode` now take `impl IntoEnum<T>` (accepting both enum variants and strings such as `.weight_function("tricube")`) rather than requiring enum variants to be imported. Callers passing enum variants must use strings instead.
* Inlined the `IntoEnum<E>` trait and its macro-generated impls for all enum-typed builder parameters directly into `api.rs` (`lowess`) and `binding_support.rs` (`fastLowess`), eliminating a previously separate `parse` module. This allows builder methods to accept either a typed enum value (e.g. `.weight_function(WeightFunction::Tricube)`) or a string (e.g. `.weight_function("tricube")`) interchangeably.
* Breaking change: Replaced the `cross_validate(CVConfig)` builder method (which required importing `KFold` or `LOOCV` types) with a string-based cross-validation API: `.cv_method("kfold")` / `.cv_method("loocv")`, `.cv_k(n)`, `.cv_fractions(vec![...])`, and `.cv_seed(n)`. `KFold` and `LOOCV` are no longer exported from the prelude. Callers using the old `cross_validate` API must migrate to the string-based cross-validation options.
* Added a `binding_support` module providing shared helpers for all language binding frontends: string-to-enum parse functions (`parse_weight_function`, `parse_robustness_method`, `parse_scaling_method`, `parse_boundary_policy`, `parse_zero_weight_fallback`, `parse_merge_strategy`, `parse_update_mode`), matching canonical-string display functions, `BuilderOptionSet` / `TypedBuilderOptionSet` structs, and `apply_builder_options` / `apply_typed_builder_options` / `apply_cross_validation` helpers. This consolidates previously duplicated logic that was scattered across every binding into a single source of truth.
* Breaking change: Renamed the internal `auto_convergence` struct field to `auto_converge` on `BatchLowessBuilder`, `OnlineLowessBuilder`, `StreamingLowessBuilder`, and the executor config types, making the field name consistent with the existing `auto_converge()` setter method. Callers accessing these fields directly must update their code.
* Breaking change: Changed `build()` to wrap all accumulated string-parse errors in a `LowessError::ParseErrors(Vec<LowessError>)` value instead of surfacing only the first error. Code matching on `LowessError::InvalidOption` from `build()` must be updated.
* Made the `IntoEnum<E>` trait `pub(crate)` in both `lowess` and `fastLowess`, restricting it to crate-internal use. Callers do not need to name this trait; builder methods continue to accept both enum variants and string literals unchanged.
* Updated `wide` dependency to v1.5, `wgpu` to v30.0, and `pollster` to v1.0.

## lowess 1.3.0

### Added

* Added prerequisites for different bindings and platforms to `CONTRIBUTING.md`
* Updated `docs/assets/diagrams/lowess_smoothing_concept.svg` to correctly illustrate LOWESS concepts (robustness iterations, bisquare re-weighting, outlier downweighting) instead of the generic LOESS algorithm it previously depicted.
* Modified `docs/requirements.txt` to update the versions of the documentation dependencies.
* Improved CI tests and coverage.
* Modified Makefile to be truely cross-platform.
* Added sanitizer check for all bindings and crates.
* Upgraded `wide` to version 1.4.

### Changed

* Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

### Fixed

* Fixed R ASAN tests failing to compile vignettes by passing `--no-build-vignettes` to `rcmdcheck`.
* Upgraded ASAN test environment to use modern `rocker/r-devel-san:latest` image and `RDscript` to resolve outdated `readelf` warnings.
* Fixed `Makefile` idempotency checks on Linux by providing a default `/tmp` fallback for the `TEMP` directory variable.
* Fixed accidental root `Cargo.toml` workspace isolation leaks by adding checked-in `pre-commit` and `pre-push` git hook guards that restore `Cargo.toml.bak` when present and fail loudly if required workspace members are still commented out.
* Added a repo-local `.cargo/config.toml` that sets `CC=clang-cl` for `x86_64-pc-windows-msvc`, fixing Criterion 0.8 benchmark builds on Windows when `cc-rs` would otherwise pick `clang.exe` and fail to link `alloca`.

## lowess 1.2.0

### Fixed

* Fixed project logo.
* Fixed documentation.
* Fixed SRR tags.

## lowess 1.1.1

### Added

* Added srr tags

## lowess 1.1.0

### Added

* Added `Mean` scaling method (Mean Absolute Deviation)
* Added hooks for custom fitting backends
* Added hooks for delegating boundary handling to the executor

### Fixed

* `FitPassFn` now returns `Result` to allow error propagation from custom fitting backends (e.g. GPU).
* Adapters (Batch, Streaming, Online) now propagate errors from the executor instead of assuming success.
* Fixed a bug where the `Extend` boundary policy was never applied.
* Implemented Coordinate Centering to preserve precision during accumulation.

## lowess 1.0.0

### Changed

* Refactored the constants to make the library robust safely against custom numeric types.
* Minor improvements to the documentation.

## lowess 0.99.9

### Changed

* Bump rust version to 1.88 for better stability
* Change function-based builder pattern in the bindings to class-based builder pattern, allowing true streaming and online processing
* Improve API docs

## lowess 0.99.7

### Fixed

* Fix README file links
* Fix Makefile bug with R versioning

## lowess 0.99.6

### Fixed

* Fix README file formats and links

## lowess 0.99.5

### Changed

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

### Fixed

* Fixed `StreamingAdapter` indexing bug that caused merged overlap points to be skipped in output
* Simplified `StreamingAdapter` API: user now provides contiguous, non-overlapping chunks while the adapter handles internal buffering and merging
* Standardized `OnlineLowess` default `min_points` to 2 (enabling smoothing after just one point)
* Sanitized residual output to avoid "negative zero" (`-0.0000`) display for near-zero values

## lowess 0.7.0

### Added

* `NoBoundary` variant to `BoundaryPolicy` enum (original Cleveland behavior)
* `ScalingMethod` enum with `MAR` and `MAD` variants for configurable robust scale estimation
* SIMD-optimized weighted least squares accumulation for `f64` and `f32`
* `WLSSolver` trait for type-specific SIMD dispatch
* `CVBuffer` struct for pre-allocated cross-validation scratch buffers
* `VecExt` trait for efficient vector reuse
* Persistent scratch buffers to `OnlineBuffer` and `StreamingBuffer`

### Changed

* Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
* Refactored partition-related types
* Replaced `RangeInclusive` iterations with `while` loops for improved performance
* Optimized `compute_window_weights` and `median_inplace`
* Added boundary thresholds for numerical stability
* Unified scale estimation logic under `ScalingMethod`
* Refactored `LowessExecutor` to accept optional external buffers
* Optimized K-Fold Cross-Validation performance

## lowess 0.6.0

### Added

* `cv_seed` field to `CVConfig` for reproducible K-Fold cross-validation
* `Backend` enum (`CPU`, `GPU`) as placeholder for GPU acceleration
* Development-only fields: `custom_fit_pass`, `custom_cv_pass`, `custom_interval_pass`, `backend`, `parallel`
* `from_config` and `to_config` methods to `LowessExecutor`

### Changed

* Refactored `cross_validate` API to use `CVConfig` struct
* Refactored `Window::recenter` to be bidirectional
* Updated `prelude` to export enum variants directly
* Reorganized `src/engine/executor.rs` into unified logical flow
* Hidden internal-only fields from public documentation
* Removed unused `GLSModel::local_wls` method
* Removed `CVMethod` and `CrossValidationStrategy` enums
* Removed type exports from `prelude` that caused ambiguity
* Removed `.cargo/config.toml`

### Fixed

* Various broken documentation links
* `WeightParams` struct to remove unused field
* Bug in `Batch` and `Streaming` adapter conversion logic

## lowess 0.5.3

### Changed

* Consolidated validation logic into `src/engine/validator.rs`
* Optimized sorting, window operations, MAD computation, and regression
* Refactored robustness to use scratch buffers (allocation-free)
* Optimized interpolation and cross-validation
* Optimized delta interpolation with binary search

## lowess 0.4.0

### Changed

* Transformed into core LOWESS implementation
* Removed `rayon` and `ndarray` dependencies
* Improved performance from 4-16× to 4-29× faster than statsmodels
* Changed license from MIT to dual AGPL-3.0 and Commercial License
* Reduced LOC from 3863 to 3263
* Removed validation and comparison code
* Removed benchmarking code
* Removed convenience re-exports

## lowess 0.3.0

### Changed

* Updated Rust version to 1.86.0
* Modified features: default std mode includes ndarray/std and rayon
* Improved documentation

### Fixed

* no-std build now compiles successfully

## lowess 0.2.0

### Changed

* Restructured project to reduce intra-module dependencies
* Renamed "quartic" kernel to "biweight"
* Cross-validation now uses true k-fold validation
* Online LOWESS performs O(span) incremental updates
* Numerous performance optimizations and numerical stability improvements

## lowess 0.1.0

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

For the full changelog, see:
<https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md>
