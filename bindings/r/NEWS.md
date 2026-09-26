<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md).

## rfastlowess (development version)

### Added

* Added self-contained R and original Cleveland LOWESS references under `validation/reference/`, with provenance, dependency, precision, and build notes.
* Added `cv_opts()` to build grouped cross-validation options for `Lowess(cv = ...)`.
* Added unit coverage for `cv_opts()` and output-flag parsing, bringing R package line coverage to 100%.
* Added `quickcheck` properties covering input-order output, `outputs = "sorted"`, and sparse one-spike initial fits against `stats::lowess`.
* Bounded broad randomized robustness checks to 12 passes, where branch differences already emerge, while fixed regressions pin long-run roundoff cycles at their original iteration counts.
* Extracted shared `stats::lowess` reference helpers into `helper-validation.R`.
* Added `quickcheck` to `Suggests` and run the LOWESS property tests in the standard suite. It is not a runtime dependency. `make r-dev` still installs `quickcheck` locally.
* Added an "Alternative Software" vignette comparing `rfastlowess` with `stats::lowess()`.

### Changed

* Updated the vendored `doxygen-awesome-css` theme to v2.5.0 and the Hugo docs build to v0.166.0.
* Replaced the local result alias with `extendr_api::error::Result`, which remains exported outside the prelude in `extendr-api 0.9.0`.
* Breaking change: replaced individual `return_*` arguments with grouped `outputs`, and replaced `Lowess()`'s four `cv_*` arguments with `cv = cv_opts(...)`.
* Represent unavailable diagnostic metrics as R `NA` rather than generic `NaN` values.
* Added regression comparisons with `stats::lowess` for input-order and sorted output.
* Added committed `statsmodels.lowess` reference fixtures for cross-language validation.
* Made `make r-dev` retry transient Windows `pak` move failures after clearing partial local cache/lock state.
* Added `make r-tests` for the R test phase; `make r-dev` invokes it and runs the LOWESS property suite 30 times in parallel.

### Fixed

* Skip comparison snippets when the optional `statsmodels` dependency is unavailable instead of failing verification.
* Fixed C++ release CI staging the tracked Spack recipe despite the repository's broad `spack/` ignore rule.

## rfastlowess 4.1.0

### Added

* Added musl release binaries for Python, C++, Go, and Julia.
* Added bundled native libraries for the Java binding across 8 platforms, with runtime musl detection and auto-extraction.
* Added `retain_model` and `predict.Lowess()` for out-of-sample prediction.
* Added `return_derivative` to `Lowess()`, `StreamingLowess()`, and `OnlineLowess()`.
* Added `return_se`/`confidence_intervals`/`prediction_intervals` to `StreamingLowess()` and `OnlineLowess()`; Online requires `update_mode = "full"`.
* Added stored golden reference fixtures under `tests/testthat/fixtures/` (with a `make_reference.R` regeneration script and a `PROVENANCE.txt` provenance stamp) pinning the output of engine paths with no external reference — the default `boundary_policy = "extend"`, intervals, robustness weights, streaming, and online — verified by `test-golden.R` within a `1e-10` tolerance (srrstats G5.4c).
* Added explicit G5.9a test: `.Machine$double.eps` scale noise on input `y` produces no meaningful change in smoothed output (verified via `expect_equal(..., tolerance = 1e-10)`).
* Added explicit tests for RE7.0/RE7.0a (noiseless exact predictor relationships, including identical-x and degenerate cases) and RE7.1/RE7.1a (noiseless exact y = f(x) relationships with timing comparison) in `tests/testthat/test-validation.R`. Removed incorrect `@srrstatsNA` tags and added proper `@srrstats` claims in `R/srr-stats-standards.R` and test headers. These tests confirm graceful handling of perfectly noiseless input and that exact data fits at least as fast as noisy equivalents.

### Changed

* Hoisted fully-qualified imports to top-level `use` statements across crates and bindings.

### Fixed

* `dev/bump_version.py` now also updates the Go module's `/vN` major-version-suffix path across `go.mod` files, doc snippets, the doc-snippet runner, and README/docs badges whenever a version bump crosses a major version boundary, so this doesn't regress on the next major release.
* `dev/bump_version.py` now also updates the Maven dependency example version in `bindings/java/docs/modules/ROOT/pages/introduction/installation.adoc`, which was previously left stale after a version bump.
* Changed the `iterations` default for `OnlineLowess` from `3` to `0` across every binding (R, Python, Julia, C++, Go, Java, Node.js, WASM), matching the default `update_mode = "incremental"` non-robust single-point fit; robustness iterations now require `update_mode = "full"`.
* `dev/bump_version.py` now updates the Go `/vN` path and the Java Maven example version.
* Fixed inconsistent Node.js naming in READMEs, doc-site home pages, and `CITATION.cff`.

## rfastlowess 4.0.0

### Added

* Added an "Ideas for Contribution" section to `CONTRIBUTING.md`, listing concrete Batch/Streaming/Online adapter feature gaps (out-of-sample prediction, exposing local slope/derivative, adaptive fraction selection, STL-style decomposition, bootstrap intervals, concurrent chunk processing, checkpointable streaming state, populating `OnlineOutput.standard_error`, time-based window eviction, configurable warm-up) to invite contributions.
* `dev/bump_version.py` now also updates the example crate version in `CONTRIBUTING.md`'s "Individual crate Cargo.toml" snippet.
* Every GPU installer (Python, Node.js, R, Julia, C++, Java, Go) now accepts a local path to an already-built GPU artifact, installing from it directly instead of downloading from a GitHub Release — useful for testing the installer itself or installing an unreleased build. `ci-*.yml`'s `gpu` jobs now build the `gpu` feature locally and exercise this path end-to-end instead of depending on a matching published release existing.
* `release-gpu.yml` now uploads every GPU artifact (across all languages and all versions) to a single perpetual `gpu-builds` release instead of the release page for the version just published, so version release pages stay uncluttered; each asset's filename embeds its source version instead. Every installer's download URL updated to match.
* Added a `return_sorted` option to `Lowess()`.
* Added a `missing` option to `Lowess()`, `StreamingLowess()`, and `OnlineLowess()`.

### Changed

* Go doc-snippet verification now batch-builds every snippet in one `go build ./...` under a persistent module instead of one `go run` per snippet, then runs the binaries concurrently; `verify_snippets.py`'s `BATCH_RUNNERS` dispatch (previously Rust-only) now covers `go` too.
* C++ doc-snippet verification now resolves the compiler/library/MSVC setup once, then compiles+links+runs every snippet concurrently instead of one at a time. Fixed an MSVC race from concurrent `cl.exe` invocations colliding on a shared `snippet.obj` by giving each snippet its own `/Fo` output and `cwd`.
* Breaking change: Removed `return_diagnostics`, `return_residuals`, and `parallel` from `OnlineLowess()`'s constructor; these options were accepted but had no effect in online mode.
* Breaking change: Removed `confidence_intervals` and `prediction_intervals` from `OnlineLowess()`'s and `StreamingLowess()`'s constructors — never actually computed. `Lowess()` is unaffected.
* Improved API documentation for R significantly.
* Updated R documentation to show the dynamic overlap default `chunk_size / 10`, clamped to `[1, chunk_size - 10]`, instead of a flat `500`.

### Fixed

* Fixed `CONTRIBUTING.md` stating a stale Go prerequisite (`1.21+`, actually `1.23+` per `go.mod`/CI), an inaccurate `air` auto-install target (claimed `make r`, actually `make r-dev`), and a stale example crate version (`2.0.0`) in the Workspace Structure section.
* Fixed `use-case-real-time.Rmd`'s dashboard example crashing at 2 data points: the internal `validate_common_args()` hardcoded a stricter `min_points = 3L` than the Rust core's actual minimum of 2. Lowered its default to `2L` to match every other binding.
* Fixed `install_gpu()` segfaulting: it overwrote the currently-loaded shared library in place (`file.copy(overwrite = TRUE)`), which can corrupt a still memory-mapped file and crash later when an unfaulted page is read back from the now-modified file on disk. It now installs via a same-directory temp file plus an atomic `file.rename()`.

## rfastlowess 3.2.1

### Added

* Added `dev/bump_version.py --version X.Y.Z` to bump every crate/binding's version files, `CITATION.cff`, and the Spack recipe in one pass (supports `--dry-run`); now also bumps `Project.toml`'s `fastlowess_jll` compat floor (safe pre-publish since `make julia-dev`/CI relax it to an OR-list at test-time).
* Added an optional `commit` input to every release workflow's `workflow_dispatch` trigger, to pin the built commit for manual runs.
* Added an `aarch64-pc-windows-gnullvm` linker entry to the root `.cargo/config.toml`, matching the existing `x86_64-pc-windows-gnu` one; makes local arm64 Windows builds work without a manual env var.

### Changed

* Added four new pins to `dev/check_pinned_versions.py`: R's `rextendr`/`roxygen2` versions and the vendored KaTeX CDN version in both Rust crates.
* Changed `check-versions.yml` to open/update a GitHub issue instead of failing CI when a pin goes stale or unreachable.

### Fixed

* Fixed a handful of `R²`/`O(n²)` Unicode superscripts the earlier ASCII-fication pass missed (added/edited after it ran) — R tests and several `lowess` crate doc-comments — replaced with `R2`/`O(n^2)`.
* Fixed `release-conda.yml`'s `sed` patterns for the feedstock's new rattler-build `recipe/recipe.yaml` format, removing now-dead R-package-name-fix/Python-dependency-injection/`build_r.sh` steps.
* Fixed `dev/check_links.py` false-flagging valid R vignette links to a sibling's rendered `.html` (R CMD build renders `.Rmd`→`.html`) as broken.
* Fixed every binding's/crate's docs and doc-comments describing `LowessResult.x` (and equivalents) as "Sorted x values"; it's actually returned in the same order as the input `x` (the algorithm sorts internally, then un-sorts every output field back to the original order). Also strengthened Python's `test_unsorted_input` to assert this instead of only checking output length.
* Fixed `.github/dependabot.yml`'s `cargo` entry for `/bindings/r/src`, which could never succeed: its `fastLowess = { path = "vendor/fastLowess" }` path dependency is only committed as `vendor.tar.xz`, never as loose files Dependabot can read. Removed the entry and added `extendr-api`'s version to `dev/check_pinned_versions.py` instead, which also uncovered and fixed a version-comparison bug there: comparing raw tuples treated a shorthand pin like `"0.9"` as older than `"0.9.0"` due to tuple-length tiebreaking; now padded to equal length first.

## rfastlowess 3.2.0

### Added

* Added `dev/add-readme-to-docs.py`, which auto-detects the Hugo (Go, Java) vs Starlight (Node.js, WASM) docs-site flavor and embeds `README.md` accordingly; wired into the corresponding Makefiles and `package.json` scripts.
* Added `.github/dependabot.yml`, covering every dependency ecosystem in the repo (`github-actions`, `cargo`, `npm`, `pip`, `maven`, `gomod`). Each directory is grouped so all its updates, including majors, land in a single weekly PR.
* Added `dev/check_pinned_versions.py` and a weekly `.github/workflows/check-versions.yml`, which check hardcoded tool/library version pins that Dependabot can't see (Corrosion's CMake `FetchContent` tag, the vendored doxygen-awesome-css theme, the Checkstyle jar, golangci-lint, and Hugo) against their latest GitHub release and fail CI if any are outdated. Read-only: it never opens PRs or edits files itself.

### Changed

* Modified `verify_snippets.py` to verify snippets and also add the output of the snippets to the markdown file.
* Added a `large` benchmark category (n = 50000) to `benchmarks/rfastlowess.R` and `benchmarks/stats_lowess.R`, since every existing category ran in well under 100ms. Covers 4 scenarios (`large_delta_0`, `large_delta_0.1`, `large_high_iter`, `large_high_fraction`) stressing `delta`, iteration count, and fraction. `benchmarks/compare.py`'s plot grid grew from 5x2 to 7x2 to fit them.
* Added `.gitattributes`, normalizing all text files to LF line endings (`* text=auto eol=lf`) and marking binary formats (images, archives, compiled libraries, `.rds`/`.RData`, etc.) so Git never treats them as text.
* Removed the `rfastlowess-package` pkgdown topic, which duplicated the adapter class list, and unexported the internal `Nullable()` helper.
* Fixed `_pkgdown.yml` describing the core interface as "R6 classes" when the package actually uses S3 classes.
* Merged `vignettes/parameters.Rmd`'s parameter reference (ranges, defaults, and fraction-choice guidance) into the `@param`/`@details` roxygen docs of `Lowess()`, `StreamingLowess()`, and `OnlineLowess()`, and removed the now-redundant vignette.
* Merged `vignettes/batch.Rmd`, `streaming.Rmd`, and `online.Rmd`'s unique content (When to Use guidance, merge strategy comparison) into the `@description`/`@details` roxygen docs of `Lowess()`, `StreamingLowess()`, and `OnlineLowess()`, and removed the now-redundant vignettes and their orphaned `gap_handling.svg`/`online_comparison.svg` diagrams.
* Standardized R documentation: consolidated README setup content and parameter guidance, renamed the batch-adapter heading, replaced flowcharts with decision tables, standardized API examples and page structure, and converted superscripts to ASCII.

### Fixed

* Fixed `docs.yml` triggering GitHub's "pages build and deployment" once per docs job; per-language jobs now upload artifacts, and a single final `deploy` job pushes to `gh-pages` once per run.
* Fixed `docs.yml`'s reliance on GitHub's legacy branch-based Pages deployment, which auto-triggered an unpinned "pages build and deployment" job on every `gh-pages` push. The former `deploy` job is now `build` (still pushes `_site` to `gh-pages` as a cache); publishing now goes through `actions/upload-pages-artifact` and a new `deploy` job using `actions/deploy-pages`. Requires the repo's Pages source set to "GitHub Actions".
* Fixed every benchmark category in `benchmarks/rfastlowess.R` failing with `attempt to apply non-function`: it called the R6-style `model$fit(x, y)`, but `fit` is an S3 generic (`fit(model, x, y)`), not a field on the `Lowess` object. Also fixed `benchmarks/stats_lowess.R` resolving its `output/` directory relative to the current working directory instead of the script's own location (unlike `rfastlowess.R`, which already did this correctly), so results could land outside `benchmarks/output/` depending on how the script was invoked.
* Fixed `OnlineLowess()`'s `min_points` (was `3L`, should be `2L`) and `update_mode` (was `"full"`, should be `"incremental"`) defaults diverging from the Rust core; updated the roxygen docs and `man/OnlineLowess.Rd` to match.
* Fixed the Handling Outliers quickstart example in R: increased `fraction` from `0.5` to `0.7` because the six-point example otherwise fit the injected outlier exactly instead of downweighting it.
* Fixed the `OnlineLowess()` roxygen example printing one line per point by collecting results and printing only `head(smoothed, 5)`.
* Fixed the `add_point()` roxygen example always printing `NULL` by using `min_points = 2L` and showing the second call.
* Fixed the robustness vignette outlier example printing incidental noise by tightening its threshold from `weight < 0.5` to `weight < 0.05`.
* Fixed the merge vignette example to print the computed overlap size and its percentage of `chunk_size`.
* Fixed the genomics vignette by fitting the current example data before plotting.
* Fixed the real-time vignette Update Modes example by feeding data to the model and plotting its results.

## rfastlowess 3.1.0

### Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.
* Added a GitHub workflow for running validation scripts.
* Added `lenght` gaurds for extra arguments.

### Changed

* Split the monolithic `.github/workflows/ci.yml` into seven per-language workflow files: `ci-rust.yml`, `ci-python.yml`, `ci-julia.yml`, `ci-nodejs.yml`, `ci-wasm.yml`, `ci-cpp.yml`, and `ci-r.yml`. Each file carries the relevant `ci` (multi-OS matrix), `asan`, and `gpu` jobs for its language.
* Each crate/binding sub-Makefile now runs `dev/verify_snippets.py --lang <lang>` for its own language as the final step of `make default`. The root `docs-test` target remains as a convenience to run all languages at once.
* Split `dev/verify_snippets.py` into a lean orchestrator and a `dev/runners/` package. Each language has its own module (`python.py`, `julia.py`, `nodejs.py`, `r.py`, `wasm.py`, `rust.py`, `cpp.py`) containing its `run_<lang>()` function and a `skip_reason()` predicate. Shared types (`Snippet`, `RunResult`) and utilities live in `runners/base.py`; the registry (`RUNNERS`, `SKIP_CHECKS`) is exported from `runners/__init__.py`.
* Moved `CHANGELOG.md` and `CONTRIBUTING.md` to the repository root.
* Simplified `bindings/r/Makefile`: replaced `Cargo.toml.orig` save/restore vendoring with `src/vendor-update.sh`; made `[workspace]` permanent in `src/Cargo.toml`; removed Bioconductor dependencies, redundant `cargo fmt --check`, `NAMESPACE` indentation post-processing, and `pkgdown::build_site` from the dev workflow.
* Moved R documentation from ReadTheDocs to GitHub Pages, served by pkgdown at <https://thisisamirv.github.io/lowess-project/r/>. The ReadTheDocs site no longer includes R-specific content.
* Changed R version dependency to 4.4.0 due to issues with installing Bioconducter packages on R < 4.4.0.
* Replaced the multi-step `install.packages` / `BiocManager::install` package installation logic in `bindings/r/Makefile` with a single [`pak`](https://pak.r-lib.org/)-based block. `pak` handles RSPM binary vs source selection automatically (including Linux), skips already-installed packages, and installs CRAN, Bioconductor (`bioc::` prefix), and R-universe packages in one call.
* `make r` (`default:`) now runs `R CMD INSTALL $(R_DIR)` directly; R's `configure` script handles Rust compilation from the committed `vendor.tar.xz`. The full dev workflow moves to `make r-dev`.
* Updated the R README to be package-specific instead of using the generic shared README.

### Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare tool names resolved via `PATH`, matching the existing fix in `bindings/r/src/cargo-config.toml`.
* Fixed Windows arm64 (R-Universe) build: `ar x` without a member name correctly resolves long-name archive entries (>16 chars stored as `/<offset>`); named extraction silently fails for such entries. Used `objcopy --remove-section=.idata$4` on each extracted `.dll` stub to strip the invalid relocations that lld 19 rejects, then `ar r` to re-insert.
* Fixed `ld.lld` crashing or dropping symbols (`WakeByAddressSingle`, `WaitOnAddress`) on Windows arm64: `--whole-archive` pulls every crate's raw-dylib stub for a given DLL into the link, but different crates' stubs cover different, non-overlapping symbols of that DLL — `--allow-multiple-definition` works on x86_64 but crashes lld's arm64pe backend. Fixed by dropping `--whole-archive` on `gnullvm`; normal archive resolution applies and nothing is lost since `entrypoint.c` already references the extendr init symbol directly.
* Fixed CRAN Windows build (`error: linker not found`): `cargo-config.toml` hardcoded linker/ar as `c:/rtools45/...` absolute paths, which break when Rtools is installed on a different drive. Fixed by using bare tool names resolved via `PATH`.
* Fixed CRAN Windows build (`cannot find -lgcc_eh`): the Rtools gcc lib directory is not writable on CRAN's server, and config-file `rustflags` does not reach build-script linker invocations. `Makevars.win` creates an empty stub via `touch` in `$(TARGET_DIR)/libgcc_mock/` and passes `LIBRARY_PATH` inline on `cargo build`. The path is resolved to an absolute path via `$(pwd)` at shell execution time — a relative path silently fails because Cargo invokes GCC to link build scripts from its own temp directory, not from `src/`.
* Fixed `Lowess(fraction = 0.3, 4)` incorrectly succeeding: `reject_extra_positional_args()` counted unnamed arguments but did not check their position, so a single unnamed arg in any non-first slot passed validation. The check now rejects any unnamed argument that is not in position 1.

## rfastlowess 3.0.0

### Added

* Exposed a `backend` option (`"cpu"` default, `"gpu"`) on `Lowess`, gated behind an opt-in `gpu` Cargo feature not enabled in CRAN/Bioconductor releases. Run `install_gpu()` to download a prebuilt GPU library (requires restarting R), or build locally with `make -f bindings/r/Makefile WITH_GPU=1`.
* Introduced S3 generics `fit()`, `process_chunk()`, `finalize()`, and `add_point()`, replacing the previous list-closure API.
* `bindings/r/Makefile` now auto-installs [Air](https://posit-dev.github.io/air/) if missing, before running `air format`.
* Added a `reject_extra_positional_args()` helper to reject extra unnamed arguments.
* Added `See: ...` cross-reference links after option headings in the R API docs, pointing to the corresponding user guide.

### Fixed

* Fixed incorrect URLs in R binding docs.

### Changed

* Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
* Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/lowess/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
* Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/lowess/` → `crates/lowess/tests/lowess/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages and `bindings/r/demo/`.
* Breaking change: Renamed the `smoothed` and `std_error` fields returned by `OnlineLowess`'s `add_point()` to `y` and `standard_error`.
* Replaced `dev/style_pkg.R` with [Air](https://posit-dev.github.io/air/) for formatting.
* Removed `dev/fix_rd_style.R`, `dev/prepare_cargo.py`, `dev/patch_vendor_crates.py`, `dev/clean_checksums.py`, and `dev/prepare_cran.sh` — their logic is now inlined directly in `bindings/r/Makefile`, so the R build no longer requires any Python scripts.
* Added `...` to `Lowess()`, `StreamingLowess()`, and `OnlineLowess()` to force named arguments for optional parameters.
* Added `Depends: R (>= 4.6)` to `DESCRIPTION` and a matching CI matrix entry.
* Expanded roxygen2 `@param` docs and added a `See Also` section linking to <https://lowess.readthedocs.io/>.
* Expanded `rfastlowess-intro.Rmd` vignettes.
* Split Streaming/Online content from the R API reference into dedicated pages, moved tutorials into the user-guide use-cases section, and standardized API examples with expected output.
* Extended `dev/verify_snippets.py` to execute R code chunks in vignettes.

## rfastlowess 2.0.0

### Added

* Added `custom_weights` parameter to `fit(Lowess, )`. Accepts a numeric vector of non-negative per-observation weights. Batch only.

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
* Breaking change: Replaced `$add_points(x, y)` (vector inputs returning a list result) on `OnlineLowess` with `$add_point(x, y)` (scalar inputs returning `numeric` or `NULL`). The method now processes one point at a time and returns `NULL` until enough points have been accumulated.

## rfastlowess 1.3.0

### Added

* Added prerequisites for different bindings and platforms to `CONTRIBUTING.md`
* Updated `docs/assets/diagrams/lowess_smoothing_concept.svg` to correctly illustrate LOWESS concepts (robustness iterations, bisquare re-weighting, outlier downweighting) instead of the generic LOESS algorithm it previously depicted.
* Modified `docs/requirements.txt` to update the versions of the documentation dependencies.
* Improved CI tests and coverage.
* Modified Makefile to be truely cross-platform.
* Added sanitizer check for all bindings and crates.
* Upgraded `rextendr` scaffold to 0.5.0: bumped `Config/rextendr/version` in `DESCRIPTION` and updated `entrypoint.c` to register the extendr panic hook (`register_extendr_panic_hook()`), so Rust panics now surface as R errors instead of crashing the session.
* Added `dev/fix_rd_style.R` post-processing script to automatically normalize Rd file indentation (to 4 spaces) and wrap long lines (> 80 characters), ensuring compliance with CRAN/pkgcheck stylistic notes.
* Added `bindings/r/_pkgdown.yml` configuration and updated the `Makefile` to use `pkgdown::build_site()`, satisfying the `pkgcheck` requirement for a dedicated documentation website.
* Added automatic copying of shared tests from the project root into the R package within `dev/prepare_cran.sh`, ensuring `R CMD build` is self-contained.
* Added direct `extendr` wrapper coverage tests plus extra validation-path tests in the R package, lifting `covr::package_coverage()` to 100% and clearing the package-level `pkgcheck`/`goodpractice` coverage complaints.

### Changed

* Refactored the R binding validation helpers to reuse `validate_common_args()` and `coerce_nullable()` in production code, split `validate_params()` into smaller helper validators, and consolidated duplicated constructor parameter documentation with `@inheritParams` before regenerating the Rd files.
* Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.
* Abides by new rOpenSci standards.

### Fixed

* Fixed R ASAN tests failing to compile vignettes by passing `--no-build-vignettes` to `rcmdcheck`.
* Upgraded ASAN test environment to use modern `rocker/r-devel-san:latest` image and `RDscript` to resolve outdated `readelf` warnings.
* Fixed `Makefile` idempotency checks on Linux by providing a default `/tmp` fallback for the `TEMP` directory variable.
* Fixed accidental root `Cargo.toml` workspace isolation leaks by adding checked-in `pre-commit` and `pre-push` git hook guards that restore `Cargo.toml.bak` when present and fail loudly if required workspace members are still commented out.
* Added a repo-local `.cargo/config.toml` that sets `CC=clang-cl` for `x86_64-pc-windows-msvc`, fixing Criterion 0.8 benchmark builds on Windows when `cc-rs` would otherwise pick `clang.exe` and fail to link `alloca`.
* Added strict pre-flight check for Pandoc in `make r` with a clear error message, as it is strictly required for building R Markdown vignettes on all platforms.
* Fixed `make r` failure on Windows when the native `tar.exe` (bsdtar) doesn't support GNU-specific reproducible build flags by adding an automatic fallback.
* Fixed `FIND: Parameter format not correct` error on Windows by replacing `find` commands with cross-platform `rm -f` wildcard expansions to prevent clashing with Windows' native `find.exe`.
* Fixed local compilation error by automatically installing required Python TOML dependencies (`tomli`, `tomli_w`) in the Makefile's R build target.
* Fixed `make r` vignette build crash by adding the missing `BiocStyle` R dependency to the Makefile's automated installation step.
* Replaced the deprecated `devtools::build_vignettes()` command with `pkgdown::build_articles()` for local vignette previewing, and added `pkgdown` to the development dependencies.
* Fixed `cargo test` and `cargo build` failing on Windows due to the default MSVC linker trying to link against `R.lib` by enforcing `--target x86_64-pc-windows-gnu` for all R-bound Cargo commands.
* Fixed `R CMD check` `WARNING` on Windows caused by R's own `Boolean.h` using a C23 enum underlying type feature: the previous pragma-based suppression triggered a CRAN `NOTE`. Fixed by pre-defining a standard C `Rboolean` enum and setting the `R_EXT_BOOLEAN_H_` header guard in `entrypoint.c` to prevent R's problematic version from loading.
* Fixed `R CMD check` `ERROR` in the testing phase: `test_check()` always resolves tests from `tests/testthat/` inside the package and ignores custom path arguments. Fixed by updating `dev/prepare_cran.sh` to copy shared tests from `tests/r/testthat/` into `bindings/r/tests/testthat/` before `R CMD build`, and simplifying `testthat.R` to a standard `test_check("rfastlowess")` call.
* Fixed missing R packages (`devtools`, `remotes`) in the Makefile's dependency installation step.
* Fixed `%1 is not a valid Win32 application` and `section below image base` DLL linkage errors on Windows by adding `-Wl,--strip-all` to `Makevars.win` `PKG_LIBS`.
* Fixed rOpenSci `pkgcheck` warning by adding documentation website URL to `DESCRIPTION`.
* Fixed CRAN note regarding non-API call `R_NamespaceRegistry` by upgrading `extendr-api` dependency to `0.9.0`.
* Fixed compilation error by providing the `Result` alias that was removed from `extendr_api::prelude` in `0.9.0`.
* Fixed the remaining SRR/pkgcheck findings in the R package by removing dead internal helper paths, reducing `validate_params()` cyclomatic complexity, eliminating duplicated roxygen parameter blocks, and covering the generated `extendr-wrappers.R` dispatch paths.

## rfastlowess 1.2.0

### Added

* Added new tests.
* Added a reference to the `CONTRIBUTING.md` file.
* Added new examples for the `print` and `plot` methods.
* Added test coverage evaluation.
* Added missing srr tags.

### Changed

* Removed the `coerce_params` dead code.
* Spread srr tags to different files and removed extra tags from `srr-stats-standards.R`.

### Fixed

* Fixed project logo.

## rfastlowess 1.0.0

### Added

* Added `mean` scaling method (Mean Absolute Deviation)
* Implemented `print` and `plot` methods for `LowessResult` objects
* Added srr tags

### Changed

* Return results as `LowessResult` S3 objects instead of raw vectors

## rfastlowess 0.99.9

### Changed

* Bump rust version to 1.88 for better stability
* Change function-based builder pattern in the bindings to class-based builder pattern, allowing true streaming and online processing
* Improve API docs
* Package is now available on conda-forge (r-rfastlowess)

## rfastlowess 0.99.7

### Fixed

* Fix README file links
* Fix Makefile bug with R versioning

## rfastlowess 0.99.6

### Fixed

* Fix README file formats and links

## rfastlowess 0.99.5

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

## rfastlowess 0.99.2

### Changed

* Prepared package for rOpenSci Software Peer Review
* Renamed main functions to avoid conflicts with base R:
* Updated documentation to reflect new API and rOpenSci guidelines

### Added

* Documentation website using `pkgdown` with automated deployment
* Comprehensive function documentation with examples and cross-references
* URL validation using `urlchecker` in CI workflow
* Rigorous parameter validation for all exported functions
* Expanded test suite achieving >96% coverage
* Codecov CI workflow and badge

### Fixed

* Documentation URLs
* Package startup messages
* `pkgcheck` workflow to run on host runner

## rfastlowess 0.99.1

### Changed

* Modified package for Bioconductor submission

## rfastlowess 0.99.0

### Added

* Support for new features in `fastLowess` v0.4.0, including `NoBoundary` boundary policy and `MAD`/`MAR` scaling methods

### Changed

* Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
* Updated documentation

## rfastlowess 0.4.0

### Added

* Support for new features in `fastLowess` v0.4.0

### Changed

* Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
* Updated documentation

## rfastlowess 0.3.0

### Added

* Option to install from R-universe without Rust

### Changed

* Updated to `fastLowess` v0.3.0 and `lowess` v0.6.0
* Updated cross-validation API

### Fixed

* Automated vendor checksum fixing for CI builds

## rfastlowess 0.2.0

### Added

* Support for new features in `fastLowess` v0.2.0

### Changed

* Improved documentation

## rfastlowess 0.1.0

### Added

* R binding for `fastLowess`

For the full changelog, see:
<https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md>
