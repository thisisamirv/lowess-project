---
title: "News"
weight: 100
---

<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md).

## fastlowess (Go) (development version)

### Added

* Added self-contained R and original Cleveland LOWESS references under `validation/reference/`, with provenance, dependency, precision, and build notes.
* Added grouped `Outputs []string`, `CV *CVOptions`, and prediction `Outputs []string` options; legacy flat fields remain accepted during migration.

### Changed

* Updated the vendored `doxygen-awesome-css` theme to v2.5.0 and the Hugo docs build to v0.166.0.
* Represent unavailable diagnostic metrics as `nil` optional values instead of `NaN` sentinels.

### Fixed

* Skip comparison snippets when the optional `statsmodels` dependency is unavailable instead of failing verification.
* Fixed C++ release CI staging the tracked Spack recipe despite the repository's broad `spack/` ignore rule.

## fastlowess (Go) 4.1.0

### Added

* Added musl release binaries for Python, C++, Go, and Julia.
* Added bundled native libraries for the Java binding across 8 platforms, with runtime musl detection and auto-extraction.
* Added `RetainModel` and `Result.PredictModel.Predict(newX, options)` for prediction.
* Added `ReturnDerivative` to `Options`, `StreamingOptions`, and `OnlineOptions`.
* Added `ReturnSE`/`ConfidenceIntervals`/`PredictionIntervals` to `StreamingOptions` and `OnlineOptions`; Online requires `UpdateMode = "full"`.

### Changed

* Hoisted fully-qualified imports to top-level `use` statements across crates and bindings.

### Fixed

* `dev/bump_version.py` now also updates the Go module's `/vN` major-version-suffix path across `go.mod` files, doc snippets, the doc-snippet runner, and README/docs badges whenever a version bump crosses a major version boundary, so this doesn't regress on the next major release.
* `dev/bump_version.py` now also updates the Maven dependency example version in `bindings/java/docs/modules/ROOT/pages/introduction/installation.adoc`, which was previously left stale after a version bump.
* Changed the `iterations` default for `OnlineLowess` from `3` to `0` across every binding (R, Python, Julia, C++, Go, Java, Node.js, WASM), matching the default `update_mode = "incremental"` non-robust single-point fit; robustness iterations now require `update_mode = "full"`.
* Breaking change: The Go module import path now includes the required `/v4` suffix; old unsuffixed imports must be updated.

## fastlowess (Go) 4.0.0

### Added

* Added an "Ideas for Contribution" section to `CONTRIBUTING.md`, listing concrete Batch/Streaming/Online adapter feature gaps (out-of-sample prediction, exposing local slope/derivative, adaptive fraction selection, STL-style decomposition, bootstrap intervals, concurrent chunk processing, checkpointable streaming state, populating `OnlineOutput.standard_error`, time-based window eviction, configurable warm-up) to invite contributions.
* `dev/bump_version.py` now also updates the example crate version in `CONTRIBUTING.md`'s "Individual crate Cargo.toml" snippet.
* Every GPU installer (Python, Node.js, R, Julia, C++, Java, Go) now accepts a local path to an already-built GPU artifact, installing from it directly instead of downloading from a GitHub Release — useful for testing the installer itself or installing an unreleased build. `ci-*.yml`'s `gpu` jobs now build the `gpu` feature locally and exercise this path end-to-end instead of depending on a matching published release existing.
* `release-gpu.yml` now uploads every GPU artifact (across all languages and all versions) to a single perpetual `gpu-builds` release instead of the release page for the version just published, so version release pages stay uncluttered; each asset's filename embeds its source version instead. Every installer's download URL updated to match.
* Added a `ReturnSorted` option to `Options`.
* Added a `Missing` option to `Options`, `StreamingOptions`, and `OnlineOptions`.
* `release-gpu.yml` now also builds GPU libraries for `linux-arm64` and `windows-arm64` (the latter via the same llvm-mingw/gnullvm cross toolchain as `release-go.yml`), matching its platform coverage.
* Added `fastlowess.InstallGPU()`, a one-time GPU downloader: fetches a prebuilt GPU-enabled static library from the matching GitHub Release to `~/.fastlowess/gpu/libfastlowess_go.a`. Since `cgo` links statically at build time, this only downloads the library — rebuild afterwards with `CGO_LDFLAGS` pointing at it (printed on success).

### Changed

* Go doc-snippet verification now batch-builds every snippet in one `go build ./...` under a persistent module instead of one `go run` per snippet, then runs the binaries concurrently; `verify_snippets.py`'s `BATCH_RUNNERS` dispatch (previously Rust-only) now covers `go` too.
* C++ doc-snippet verification now resolves the compiler/library/MSVC setup once, then compiles+links+runs every snippet concurrently instead of one at a time. Fixed an MSVC race from concurrent `cl.exe` invocations colliding on a shared `snippet.obj` by giving each snippet its own `/Fo` output and `cwd`.
* Repinned the macOS x64 job in `release-go.yml` to `macos-15-intel`.
* Breaking change: `OnlineOptions` no longer embeds `Options`; removed `ReturnDiagnostics`, `ReturnResiduals`, `Parallel`, and the never-read `Backend`.
* Breaking change: `StreamingOptions` no longer embeds `Options` either; both lost `ConfidenceIntervals`/`PredictionIntervals`, and `StreamingOptions` also lost `ReturnSE`/`ReturnSorted`/`CVFractions`/`CVMethod`/`CVK`/`CVSeed`/`Backend`.
* Improved API documentation for Go significantly.
* Updated Go documentation to show the dynamic overlap default `chunk_size / 10`, clamped to `[1, chunk_size - 10]`, instead of a flat `500`.

### Fixed

* Fixed `CONTRIBUTING.md` stating a stale Go prerequisite (`1.21+`, actually `1.23+` per `go.mod`/CI), an inaccurate `air` auto-install target (claimed `make r`, actually `make r-dev`), and a stale example crate version (`2.0.0`) in the Workspace Structure section.

## fastlowess (Go) 3.2.1

### Added

* Added `dev/bump_version.py --version X.Y.Z` to bump every crate/binding's version files, `CITATION.cff`, and the Spack recipe in one pass (supports `--dry-run`); now also bumps `Project.toml`'s `fastlowess_jll` compat floor (safe pre-publish since `make julia-dev`/CI relax it to an OR-list at test-time).
* Added an optional `commit` input to every release workflow's `workflow_dispatch` trigger, to pin the built commit for manual runs.
* Added an `aarch64-pc-windows-gnullvm` linker entry to the root `.cargo/config.toml`, matching the existing `x86_64-pc-windows-gnu` one; makes local arm64 Windows builds work without a manual env var.
* Added a `dev/check_pinned_versions.py` pin for the docs-site MathJax CDN version.
* Added ARM64 release binaries to `release-go.yml` (native Linux, cross-compiled Windows via `aarch64-pc-windows-gnullvm` + llvm-mingw), with the same macOS x64/`macos-13` mislabeling fix as C++.
* Added an arm64 job to `ci-go.yml` using the same llvm-mingw toolchain, so arm64 support is verified on every push.

### Changed

* Added four new pins to `dev/check_pinned_versions.py`: R's `rextendr`/`roxygen2` versions and the vendored KaTeX CDN version in both Rust crates.
* Changed `check-versions.yml` to open/update a GitHub issue instead of failing CI when a pin goes stale or unreachable.
* Bumped docs-site MathJax CDN version from `3.2.2` to `4.1.3`, updating `dev/check_pinned_versions.py`'s pattern for MathJax 4's CDN layout.

### Fixed

* Fixed a handful of `R²`/`O(n²)` Unicode superscripts the earlier ASCII-fication pass missed (added/edited after it ran) — R tests and several `lowess` crate doc-comments — replaced with `R2`/`O(n^2)`.
* Fixed `release-conda.yml`'s `sed` patterns for the feedstock's new rattler-build `recipe/recipe.yaml` format, removing now-dead R-package-name-fix/Python-dependency-injection/`build_r.sh` steps.
* Fixed `dev/check_links.py` false-flagging valid R vignette links to a sibling's rendered `.html` (R CMD build renders `.Rmd`→`.html`) as broken.
* Fixed every binding's/crate's docs and doc-comments describing `LowessResult.x` (and equivalents) as "Sorted x values"; it's actually returned in the same order as the input `x` (the algorithm sorts internally, then un-sorts every output field back to the original order). Also strengthened Python's `test_unsorted_input` to assert this instead of only checking output length.
* Fixed `.github/dependabot.yml`'s `cargo` entry for `/bindings/r/src`, which could never succeed: its `fastLowess = { path = "vendor/fastLowess" }` path dependency is only committed as `vendor.tar.xz`, never as loose files Dependabot can read. Removed the entry and added `extendr-api`'s version to `dev/check_pinned_versions.py` instead, which also uncovered and fixed a version-comparison bug there: comparing raw tuples treated a shorthand pin like `"0.9"` as older than `"0.9.0"` due to tuple-length tiebreaking; now padded to equal length first.
* Fixed pkg.go.dev showing "License: None detected" and failing tagged/stable checks: `LICENSE-MIT`/`LICENSE-APACHE` lived one directory above the actual Go module. Copied both into `bindings/go/fastlowess`; `release-go.yml` now also pushes a nested-module `vX.Y.Z` tag.
* Fixed `OnlineOptions`'s `MinPoints`/`UpdateMode` defaults (`3`/`"full"`) diverging from every other binding and the Rust core; now `2`/`"incremental"`.
* Fixed `bindings/go/Makefile` and `ffi.go`'s cgo `LDFLAGS` both unconditionally targeting the x64 GNU build on any Windows host, which would have cross-compiled/linked for the wrong architecture on arm64.

## fastlowess (Go) 3.2.0

### Added

* Added `dev/add-readme-to-docs.py`, which auto-detects the Hugo (Go, Java) vs Starlight (Node.js, WASM) docs-site flavor and embeds `README.md` accordingly; wired into the corresponding Makefiles and `package.json` scripts.
* Added `.github/dependabot.yml`, covering every dependency ecosystem in the repo (`github-actions`, `cargo`, `npm`, `pip`, `maven`, `gomod`). Each directory is grouped so all its updates, including majors, land in a single weekly PR.
* Added `dev/check_pinned_versions.py` and a weekly `.github/workflows/check-versions.yml`, which check hardcoded tool/library version pins that Dependabot can't see (Corrosion's CMake `FetchContent` tag, the vendored doxygen-awesome-css theme, the Checkstyle jar, golangci-lint, and Hugo) against their latest GitHub release and fail CI if any are outdated. Read-only: it never opens PRs or edits files itself.
* Added a new Go binding (`bindings/go`), consuming the `fastLowess` Rust core via a dedicated `cgo`-compatible C ABI (`fastlowess-go` crate, mirroring the C++ binding's FFI approach).

### Changed

* Modified `verify_snippets.py` to verify snippets and also add the output of the snippets to the markdown file.
* Added a `large` benchmark category (n = 50000) to `benchmarks/rfastlowess.R` and `benchmarks/stats_lowess.R`, since every existing category ran in well under 100ms. Covers 4 scenarios (`large_delta_0`, `large_delta_0.1`, `large_high_iter`, `large_high_fraction`) stressing `delta`, iteration count, and fraction. `benchmarks/compare.py`'s plot grid grew from 5x2 to 7x2 to fit them.
* Added `.gitattributes`, normalizing all text files to LF line endings (`* text=auto eol=lf`) and marking binary formats (images, archives, compiled libraries, `.rds`/`.RData`, etc.) so Git never treats them as text.
* Standardized Go documentation: consolidated README setup content and parameter guidance, renamed the batch-adapter heading, replaced flowcharts with decision tables, standardized API examples and page structure, and converted superscripts to ASCII.

### Fixed

* Fixed `docs.yml` triggering GitHub's "pages build and deployment" once per docs job; per-language jobs now upload artifacts, and a single final `deploy` job pushes to `gh-pages` once per run.
* Fixed `docs.yml`'s reliance on GitHub's legacy branch-based Pages deployment, which auto-triggered an unpinned "pages build and deployment" job on every `gh-pages` push. The former `deploy` job is now `build` (still pushes `_site` to `gh-pages` as a cache); publishing now goes through `actions/upload-pages-artifact` and a new `deploy` job using `actions/deploy-pages`. Requires the repo's Pages source set to "GitHub Actions".
* Fixed every benchmark category in `benchmarks/rfastlowess.R` failing with `attempt to apply non-function`: it called the R6-style `model$fit(x, y)`, but `fit` is an S3 generic (`fit(model, x, y)`), not a field on the `Lowess` object. Also fixed `benchmarks/stats_lowess.R` resolving its `output/` directory relative to the current working directory instead of the script's own location (unlike `rfastlowess.R`, which already did this correctly), so results could land outside `benchmarks/output/` depending on how the script was invoked.
* Fixed the Handling Outliers quickstart example in Go: increased `fraction` from `0.5` to `0.7` because the six-point example otherwise fit the injected outlier exactly instead of downweighting it.

## fastlowess (Go) 3.1.0

### Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.
* Added a GitHub workflow for running validation scripts.

### Changed

* Split the monolithic `.github/workflows/ci.yml` into seven per-language workflow files: `ci-rust.yml`, `ci-python.yml`, `ci-julia.yml`, `ci-nodejs.yml`, `ci-wasm.yml`, `ci-cpp.yml`, and `ci-r.yml`. Each file carries the relevant `ci` (multi-OS matrix), `asan`, and `gpu` jobs for its language.
* Each crate/binding sub-Makefile now runs `dev/verify_snippets.py --lang <lang>` for its own language as the final step of `make default`. The root `docs-test` target remains as a convenience to run all languages at once.
* Split `dev/verify_snippets.py` into a lean orchestrator and a `dev/runners/` package. Each language has its own module (`python.py`, `julia.py`, `nodejs.py`, `r.py`, `wasm.py`, `rust.py`, `cpp.py`) containing its `run_<lang>()` function and a `skip_reason()` predicate. Shared types (`Snippet`, `RunResult`) and utilities live in `runners/base.py`; the registry (`RUNNERS`, `SKIP_CHECKS`) is exported from `runners/__init__.py`.
* Moved `CHANGELOG.md` and `CONTRIBUTING.md` to the repository root.

### Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare tool names resolved via `PATH`, matching the existing fix in `bindings/r/src/cargo-config.toml`.

## fastlowess (Go) 3.0.0

### Changed

* Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
* Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/lowess/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
* Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/lowess/` → `crates/lowess/tests/lowess/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages and `bindings/r/demo/`.

## fastlowess (Go) 2.0.0

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

## fastlowess (Go) 1.3.0

### Added

* Added prerequisites for different bindings and platforms to `CONTRIBUTING.md`
* Updated `docs/assets/diagrams/lowess_smoothing_concept.svg` to correctly illustrate LOWESS concepts (robustness iterations, bisquare re-weighting, outlier downweighting) instead of the generic LOESS algorithm it previously depicted.
* Modified `docs/requirements.txt` to update the versions of the documentation dependencies.
* Improved CI tests and coverage.
* Modified Makefile to be truely cross-platform.
* Added sanitizer check for all bindings and crates.

### Fixed

* Fixed R ASAN tests failing to compile vignettes by passing `--no-build-vignettes` to `rcmdcheck`.
* Upgraded ASAN test environment to use modern `rocker/r-devel-san:latest` image and `RDscript` to resolve outdated `readelf` warnings.
* Fixed `Makefile` idempotency checks on Linux by providing a default `/tmp` fallback for the `TEMP` directory variable.
* Fixed accidental root `Cargo.toml` workspace isolation leaks by adding checked-in `pre-commit` and `pre-push` git hook guards that restore `Cargo.toml.bak` when present and fail loudly if required workspace members are still commented out.
* Added a repo-local `.cargo/config.toml` that sets `CC=clang-cl` for `x86_64-pc-windows-msvc`, fixing Criterion 0.8 benchmark builds on Windows when `cc-rs` would otherwise pick `clang.exe` and fail to link `alloca`.

## fastlowess (Go) 1.2.0

### Fixed

* Fixed project logo.

## fastlowess (Go) 0.99.9

### Changed

* Bump rust version to 1.88 for better stability
* Change function-based builder pattern in the bindings to class-based builder pattern, allowing true streaming and online processing
* Improve API docs

## fastlowess (Go) 0.99.7

### Fixed

* Fix README file links
* Fix Makefile bug with R versioning

## fastlowess (Go) 0.99.6

### Fixed

* Fix README file formats and links

## fastlowess (Go) 0.99.5

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

For the full changelog, see:
<https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md>
