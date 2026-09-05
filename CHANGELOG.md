<!-- markdownlint-disable MD024 MD046 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 4.0.0

### Added

**Monorepo:**

- Added an "Ideas for Contribution" section to `CONTRIBUTING.md`, listing concrete Batch/Streaming/Online adapter feature gaps (out-of-sample prediction, exposing local slope/derivative, adaptive fraction selection, STL-style decomposition, bootstrap intervals, concurrent chunk processing, checkpointable streaming state, populating `OnlineOutput.standard_error`, time-based window eviction, configurable warm-up) to invite contributions.
- `dev/bump_version.py` now also updates the example crate version in `CONTRIBUTING.md`'s "Individual crate Cargo.toml" snippet.

**lowess:**

- Added `.return_sorted()` to the batch builder, to return results sorted ascending by `x` instead of input order. Default `false`.
- Added a `missing` option (`"error"` default, or `"drop"`) controlling non-finite (NaN/Inf) `x`/`y` handling: Batch/Streaming drop non-finite rows (and matching `custom_weights`); Online skips non-finite points, returning `Ok(None)`. Length mismatches always error.
- Added `release-rust.yml` to publish to crates.io on release.

**fastLowess:**

- Added `return_sorted` to `BuilderOptionSet`/`TypedBuilderOptionSet` and the `Lowess` (Batch) builder.
- Added `missing` to `BuilderOptionSet`/`TypedBuilderOptionSet` and the `Lowess`/`StreamingLowess`/`OnlineLowess` builders.
- Now published via `release-rust.yml`, 3 minutes after `lowess` to let the crates.io index catch up.

**Python:**

- Added a `return_sorted` option to `Lowess`.
- Added a `missing` option to `Lowess`, `StreamingLowess`, and `OnlineLowess`.

**R:**

- Added a `return_sorted` option to `Lowess()`.
- Added a `missing` option to `Lowess()`, `StreamingLowess()`, and `OnlineLowess()`.

**Julia:**

- Added a `return_sorted` option to `Lowess`.
- Added a `missing` option to `Lowess`, `StreamingLowess`, and `OnlineLowess`.

**Node.js:**

- Added a `return_sorted` option to `Lowess`'s `SmoothOptions`.
- Added a `missing` option to `SmoothOptions`, `StreamingSmoothOptions`, and `OnlineSmoothOptions`.

**WASM:**

- Added a `return_sorted` option to `Lowess`'s `SmoothOptions`.
- Added a `missing` option to `SmoothOptions`, `StreamingSmoothOptions`, and `OnlineSmoothOptions`.

**C++:**

- Added a `return_sorted` option to `LowessOptions`.
- Added a `missing` option to `LowessOptions` and `OnlineOptions` (inherited by `StreamingOptions`).

**Go:**

- Added a `ReturnSorted` option to `Options`.
- Added a `Missing` option to `Options`, `StreamingOptions`, and `OnlineOptions`.

**Java:**

- Added a `returnSorted` option to `Options`.
- Added a `missing` option to `Options.Builder` (shared by `Options`, `StreamingOptions`, and `OnlineOptions`).

### Changed

**Monorepo:**

- Go doc-snippet verification now batch-builds every snippet in one `go build ./...` under a persistent module instead of one `go run` per snippet, then runs the binaries concurrently; `verify_snippets.py`'s `BATCH_RUNNERS` dispatch (previously Rust-only) now covers `go` too.
- C++ doc-snippet verification now resolves the compiler/library/MSVC setup once, then compiles+links+runs every snippet concurrently instead of one at a time. Fixed an MSVC race from concurrent `cl.exe` invocations colliding on a shared `snippet.obj` by giving each snippet its own `/Fo` output and `cwd`.

**docs:**

- Improved API docs for all bindings and crates significantly.
- Fixed several bindings' docs (Node.js, WASM, Go, Java, R) showing a flat `500` default for `overlap` instead of the actual dynamic `chunk_size / 10` (clamped to `[1, chunk_size - 10]`).
- Renamed Python's `docs/guide/adapters.md` and `docs/use-case/{genomics,real-time,time-series}.md` to match every other binding/crate's filenames (`adapter-choice.md`, `use-case-*.md`).

**lowess:**

- Removed the dead, unreachable `compute_residuals`/`parallel`/`backend` fields from `OnlineLowessBuilder`; `StreamingLowessBuilder` lost its unused `backend` field too.
- `Streaming::convert()` no longer resolves `overlap` to a flat `500` when unset; it now resolves dynamically to `chunk_size / 10` (clamped to `[1, chunk_size - 10]`). Breaking change for callers relying on the previous flat default with a customized `chunk_size`.

**fastLowess:**

- Removed the same dead `compute_residuals`/`parallel`/`backend` fields as `lowess`.
- Removed `.confidence_intervals()`, `.prediction_intervals()`, and `.return_se()` from the `StreamingLowess`/`OnlineLowess` wrapper structs — leaked in via the shared builder macro and silently ignored. Breaking change; `Lowess` is unaffected.
- Fixed a stale comment on `binding_support::default_overlap()` referencing the now-removed flat `DEFAULT_STREAMING_OVERLAP` constant in `lowess`.

**Python:**

- Removed `return_diagnostics`, `return_residuals`, and `parallel` from `OnlineLowess`'s constructor — accepted but had no effect. Breaking change; `Lowess`/`StreamingLowess` are unaffected.

**R:**

- Removed `return_diagnostics`, `return_residuals`, and `parallel` from `OnlineLowess()`'s constructor, same reason as Python. Breaking change.
- Removed `confidence_intervals` and `prediction_intervals` from `OnlineLowess()`'s and `StreamingLowess()`'s constructors — never actually computed. Breaking change; `Lowess()` is unaffected.

**Julia:**

- Removed `return_diagnostics`, `return_residuals`, and `parallel` from `OnlineLowess`, same reason as Python. Breaking change.
- `StreamingLowess`'s `overlap` default changed from a fixed `500` to `-1` (sentinel for "use the library default"), resolving dynamically to `chunk_size / 10` like every other binding. Breaking change for customized `chunk_size` callers.

**Node.js:**

- Split `SmoothOptions` into `SmoothOptions` (Batch), `StreamingSmoothOptions`, and `OnlineSmoothOptions`. Passing Batch-only fields to `StreamingLowess`/`OnlineLowess` is now a TypeScript compile-time error instead of a silent no-op. Breaking change; `Lowess` is unaffected.

**WASM:**

- Same `SmoothOptions` split as Node.js, for the same reason. Breaking change.

**C++:**

- Repinned the macOS x64 job in `release-cpp.yml` to `macos-15-intel`.
- Removed `return_diagnostics`/`return_residuals`/`parallel` from `OnlineOptions`, same reason as Python. Breaking change.
- Removed `confidence_intervals`/`prediction_intervals` from `OnlineOptions`; `StreamingOptions` no longer forwards its inherited copies. Breaking change.
- Removed the dead, never-read `custom_weights` field from `OnlineOptions`. Breaking change.
- `StreamingOptions::overlap`'s default changed from a fixed `500` to `-1` (sentinel for "use the library default"), resolving dynamically to `chunk_size / 10`. Breaking change.

**Go:**

- Repinned the macOS x64 job in `release-go.yml` to `macos-15-intel`.
- `OnlineOptions` no longer embeds `Options`; removed `ReturnDiagnostics`, `ReturnResiduals`, `Parallel`, and the never-read `Backend`. Breaking change.
- `StreamingOptions` no longer embeds `Options` either; both lost `ConfidenceIntervals`/`PredictionIntervals`, and `StreamingOptions` also lost `ReturnSE`/`ReturnSorted`/`CVFractions`/`CVMethod`/`CVK`/`CVSeed`/`Backend`. Breaking change.

**Java:**

- Removed `returnDiagnostics`, `returnResiduals`, and `parallel` from `OnlineOptions.Builder`, same reason as Python. Breaking change.
- Removed `confidenceIntervals` and `predictionIntervals` from `OnlineOptions.Builder` and `StreamingOptions.Builder`. Breaking change.

### Fixed

**Monorepo:**

- Fixed `CONTRIBUTING.md` stating a stale Go prerequisite (`1.21+`, actually `1.23+` per `go.mod`/CI), an inaccurate `air` auto-install target (claimed `make r`, actually `make r-dev`), and a stale example crate version (`2.0.0`) in the Workspace Structure section.

**Python:**

- Fixed `_core.pyi` stating stale defaults that diverged from the actual PyO3 runtime (and every other binding): `StreamingLowess.fraction` (`0.3`→`0.67`), `OnlineLowess.fraction` (`0.2`→`0.67`), `OnlineLowess.window_capacity` (`100`→`1000`), and `OnlineLowess.update_mode` (`"full"`→`"incremental"`). Runtime behavior was already correct.

**Node.js:**

- Fixed `OnlineOptions` doc comments stating `min_points` defaults to `3` (actually `2`) and `update_mode` defaults to `"full"` (actually `"incremental"`).

**WASM:**

- Fixed the same class of stale doc-comment defaults as Node.js: `StreamingOptions.overlap` stated a flat `500` instead of the dynamic `chunk_size / 10`, and `OnlineOptions.window_capacity`/`update_mode` stated `100`/`"full"` instead of `1000`/`"incremental"`.

**Java:**

- Fixed `StreamingOptions.Builder.overlap()`'s Javadoc stating a stale flat `500` default; it's actually dynamic (`chunk_size / 10`, clamped to `[1, chunk_size - 10]`).

**R:**

- Fixed `use-case-real-time.Rmd`'s dashboard example crashing at 2 data points: the internal `validate_common_args()` hardcoded a stricter `min_points = 3L` than the Rust core's actual minimum of 2. Lowered its default to `2L` to match every other binding.

## 3.2.1

### Added

**Monorepo:**

- Added `dev/bump_version.py --version X.Y.Z` to bump every crate/binding's version files, `CITATION.cff`, and the Spack recipe in one pass (supports `--dry-run`); now also bumps `Project.toml`'s `fastlowess_jll` compat floor (safe pre-publish since `make julia-dev`/CI relax it to an OR-list at test-time).
- Added an optional `commit` input to every release workflow's `workflow_dispatch` trigger, to pin the built commit for manual runs.
- Added an `aarch64-pc-windows-gnullvm` linker entry to the root `.cargo/config.toml`, matching the existing `x86_64-pc-windows-gnu` one; makes local arm64 Windows builds work without a manual env var.

**Node.js:**

- Added `aarch64-unknown-linux-musl` and `armv7-unknown-linux-gnueabihf` prebuilt targets with matching optional npm subpackages.

**C++:**

- Added ARM64 release binaries to `release-cpp.yml` (Linux, Windows, macOS); the macOS x64 job is now pinned to `macos-13` instead of `macos-latest`, which has been Apple Silicon since 2024 and was silently shipping an arm64 binary mislabeled as x64.

**Go:**

- Added a `dev/check_pinned_versions.py` pin for the docs-site MathJax CDN version.
- Added ARM64 release binaries to `release-go.yml` (native Linux, cross-compiled Windows via `aarch64-pc-windows-gnullvm` + llvm-mingw), with the same macOS x64/`macos-13` mislabeling fix as C++.
- Added an arm64 job to `ci-go.yml` using the same llvm-mingw toolchain, so arm64 support is verified on every push.

**Python:**

- Added a Windows ARM64 wheel-build job to `release-pypi.yml` (Linux/macOS ARM64 wheels already existed), using Python 3.11 as the build interpreter since python.org only ships win-arm64 installers from 3.11 onward; the wheel remains `abi3-py38`-compatible regardless.

### Changed

**Monorepo:**

- Added four new pins to `dev/check_pinned_versions.py`: R's `rextendr`/`roxygen2` versions and the vendored KaTeX CDN version in both Rust crates.
- Changed `check-versions.yml` to open/update a GitHub issue instead of failing CI when a pin goes stale or unreachable.

**lowess/fastLowess:**

- Bumped the vendored KaTeX CDN version from `0.18.4` to `0.18.5`, updating SRI hashes to match.

**Node.js:**

- Updated `@astrojs/starlight` to v0.42, `@napi-rs/cli` to v3.9, and `astro` to v7.3.

**WASM:**

- Updated `@astrojs/starlight` to v0.42 and `astro` to v7.3.

**Java:**

- Pinned the docs-site's jsDelivr MathJax CDN reference to `mathjax@4.1.3` (was the rolling `@3`).
- Bumped `maven-compiler-plugin` to 3.16.0 and `maven-surefire-plugin` to 3.6.0.

**Go:**

- Bumped docs-site MathJax CDN version from `3.2.2` to `4.1.3`, updating `dev/check_pinned_versions.py`'s pattern for MathJax 4's CDN layout.

### Fixed

**Monorepo:**

- Fixed a handful of `R²`/`O(n²)` Unicode superscripts the earlier ASCII-fication pass missed (added/edited after it ran) — R tests and several `lowess` crate doc-comments — replaced with `R2`/`O(n^2)`.
- Fixed `release-conda.yml`'s `sed` patterns for the feedstock's new rattler-build `recipe/recipe.yaml` format, removing now-dead R-package-name-fix/Python-dependency-injection/`build_r.sh` steps.
- Fixed `dev/check_links.py` false-flagging valid R vignette links to a sibling's rendered `.html` (R CMD build renders `.Rmd`→`.html`) as broken.
- Fixed every binding's/crate's docs and doc-comments describing `LowessResult.x` (and equivalents) as "Sorted x values"; it's actually returned in the same order as the input `x` (the algorithm sorts internally, then un-sorts every output field back to the original order). Also strengthened Python's `test_unsorted_input` to assert this instead of only checking output length.
- Fixed `.github/dependabot.yml`'s `cargo` entry for `/bindings/r/src`, which could never succeed: its `fastLowess = { path = "vendor/fastLowess" }` path dependency is only committed as `vendor.tar.xz`, never as loose files Dependabot can read. Removed the entry and added `extendr-api`'s version to `dev/check_pinned_versions.py` instead, which also uncovered and fixed a version-comparison bug there: comparing raw tuples treated a shorthand pin like `"0.9"` as older than `"0.9.0"` due to tuple-length tiebreaking; now padded to equal length first.

**Go:**

- Fixed pkg.go.dev showing "License: None detected" and failing tagged/stable checks: `LICENSE-MIT`/`LICENSE-APACHE` lived one directory above the actual Go module. Copied both into `bindings/go/fastlowess`; `release-go.yml` now also pushes a nested-module `vX.Y.Z` tag.
- Fixed `OnlineOptions`'s `MinPoints`/`UpdateMode` defaults (`3`/`"full"`) diverging from every other binding and the Rust core; now `2`/`"incremental"`.
- Fixed `bindings/go/Makefile` and `ffi.go`'s cgo `LDFLAGS` both unconditionally targeting the x64 GNU build on any Windows host, which would have cross-compiled/linked for the wrong architecture on arm64.

**Julia:**

- Project version not bumped, resulting in unsuccessful release of version 3.2.0.

**Java:**

- Fixed `actions/setup-java@v6`'s deprecation warning by migrating `release-java.yml` to the `-env-var` credential inputs.
- Fixed a Surefire "configured twice" warning by moving `fastlowess.native.dir` into `<systemPropertyVariables>`.
- Fixed ~100 `mvn javadoc:jar` warnings (missing tags/undocumented methods); `maven-javadoc-plugin` now sets `failOnWarning` as a dedicated `make java-dev` check step.
- Fixed `OnlineOptions.Builder()` never calling `parallel(false)`, silently defaulting online mode's `parallel` to `true` unlike every other binding.

**Python:**

- Fixed `release-pypi.yml`/`release-gpu.yml`'s macOS/Windows jobs printing a pip version-check notice on every run; added `PIP_DISABLE_PIP_VERSION_CHECK: "1"`.

**C++:**

- Fixed `release-cpp.yml`'s `spack-release` job failing to `git push` from a detached HEAD; now checks out and pushes to the default branch explicitly.
- Fixed `bindings/cpp/spack/package.py`'s example `url` going stale (only `version()`/`sha256` were auto-updated); `dev/bump_version.py` now refreshes it too.

**Node.js/WASM:**

- Fixed `astro build` failing since Astro 7 no longer bundles `@astrojs/markdown-remark` by default, which the KaTeX plugins need; added it as an explicit devDependency in both.

## 3.2.0

### Added

**Monorepo:**

- Added `dev/add-readme-to-docs.py`, which auto-detects the Hugo (Go, Java) vs Starlight (Node.js, WASM) docs-site flavor and embeds `README.md` accordingly; wired into the corresponding Makefiles and `package.json` scripts.
- Added `.github/dependabot.yml`, covering every dependency ecosystem in the repo (`github-actions`, `cargo`, `npm`, `pip`, `maven`, `gomod`). Each directory is grouped so all its updates, including majors, land in a single weekly PR.
- Added `dev/check_pinned_versions.py` and a weekly `.github/workflows/check-versions.yml`, which check hardcoded tool/library version pins that Dependabot can't see (Corrosion's CMake `FetchContent` tag, the vendored doxygen-awesome-css theme, the Checkstyle jar, golangci-lint, and Hugo) against their latest GitHub release and fail CI if any are outdated. Read-only: it never opens PRs or edits files itself.

**C++:**

- Library is now available on Spack (`fastlowess-cpp`).

**Go:**

- Added a new Go binding (`bindings/go`), consuming the `fastLowess` Rust core via a dedicated `cgo`-compatible C ABI (`fastlowess-go` crate, mirroring the C++ binding's FFI approach).

**Java:**

- Added a new Java binding (`bindings/java`), consuming the `fastLowess` Rust core via a JNI shared library (`fastlowess-java` crate, using the `jni` crate).

**Node.js:**

- Added a `VERSION` export (`bindings/nodejs/index.js`), sourced from `package.json`, so consumers can query the Node.js binding's own version without reaching into `require('fastlowess/package.json')` directly.

### Changed

**Monorepo:**

- Modified `verify_snippets.py` to verify snippets and also add the output of the snippets to the markdown file.
- Added a `large` benchmark category (n = 50000) to `benchmarks/rfastlowess.R` and `benchmarks/stats_lowess.R`, since every existing category ran in well under 100ms. Covers 4 scenarios (`large_delta_0`, `large_delta_0.1`, `large_high_iter`, `large_high_fraction`) stressing `delta`, iteration count, and fraction. `benchmarks/compare.py`'s plot grid grew from 5x2 to 7x2 to fit them.
- Added `.gitattributes`, normalizing all text files to LF line endings (`* text=auto eol=lf`) and marking binary formats (images, archives, compiled libraries, `.rds`/`.RData`, etc.) so Git never treats them as text.

**docs:**

- Consolidated every crate/binding README: merged the "Installation" and "Documentation" sections, replaced GitHub-only alert syntax with plain blockquotes, removed the redundant "API Reference" and "Changelog" sections (each now has its own docs-site page), shortened the "GPU Backend" blurb, and added a "Read more" link to the Concepts page. The top-level repository README is unchanged, since it's only ever viewed on GitHub.
- Renamed the batch adapter's "When to Use" heading to "When to Use Batch Adapter" across every binding/crate's API docs.
- Vendored the [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) theme (v2.4.2) for a modern, sidebar-only cpp Doxygen site with automatic dark mode.
- Added `dev/update_changelogs.py`, which regenerates a per-binding/crate `NEWS.md`/`news.md` from the root `CHANGELOG.md`. Wired into every docs site's navigation, the Rust crates' rustdoc module tree, and every `Makefile` `dev` target.
- Replaced `kernels.md`'s "Choosing a Kernel" mermaid flowchart (every binding/crate) with an equivalent decision table, since Doxygen and rustdoc don't render mermaid and the deeply-nested diamond chain was hard to read even where it did render.
- Replaced `adapter-choice.md`/`adapters.md`'s "Overview" flowchart (mermaid in most bindings/crates, ASCII art in the C++ docs) with an equivalent decision table, unifying on a single rendering-agnostic format across every binding/crate.
- Moved the duplicated "GPU Acceleration" section out of `api.md` (C++, Node.js, Python, `fastLowess`) into each one's dedicated `gpu-backend.md` guide, which also gained "Hardware Requirements"/"Performance Considerations"; `api.md` now just links to it. Removed the section entirely from the `lowess` crate, which has no `gpu` feature.
- Consolidated `parameters.md`/the auto-generated parameter reference across every binding and crate: merged its unique content (fraction/iterations guidance, `delta` defaults, `zero_weight_fallback` behavior) into each `api.md`'s option tables, then removed `parameters.md` and its nav entries/rustdoc module — the option lists and examples it duplicated already live on their own pages.
- Standardized docs across all bindings and crates.
- Replaced the Unicode superscript `²` character with plain ASCII throughout every doc page and doc-comment (`R²` → `R2`; `(y_true - y_pred)²`/`O(window²)` → `^2`), including the `wasm` binding's JSDoc comments, the Julia docstrings in `FastLOWESS.jl`, and the `fastLowess`/`lowess` crates' rustdoc examples. Also changed the `lowess` crate's `Diagnostics` `Display` impl to print `R2` instead of `R²` so the `fastLowess` doc example showing its captured output stays accurate.
- Harmonized the docs-site directory structure across every binding/crate to mirror Go's layout (`introduction/`, `guide/`, `weighting/`, `advanced/`, `use-case/`, `api/`, each grouped under a hub page): Node.js/WASM (Starlight `sidebar`), Python (Sphinx toctree, also removed the dead `mkdocs.yml`), Julia (`Documenter.jl` `pages=[...]`, switched to recursive `walkdir`), and `fastLowess`/`lowess` (nested `#[cfg(doc)]` modules). C++'s Doxygen pages were physically moved to match (`RECURSIVE: YES`), with two hub pages renamed for parity. Java already matched this layout and now carries it into its new Antora site. R's `vignettes/` stays flat (CRAN/pkgdown requirement).

**C++:**

- Restructured the Doxygen site's navigation (previously ~20 flat pages) into six nested hub pages (`Getting Started`, `User Guide`, `Weight & Robustness`, `Advanced`, `Use Cases`, `API`) via `\subpage`, matching every other binding's docs site. Updated `README.md`'s hardcoded Doxygen URLs to match.
- Added a Spack recipe (`bindings/cpp/spack/package.py`, a `CargoPackage` with custom `build()`/`install()` phases). `release-cpp.yml` now updates its `version()`/`sha256` on every release and opens a PR to `spack/spack-packages`, so `fastlowess-cpp` stays installable via `spack install fastlowess-cpp`.
- Bumped the vendored Corrosion CMake module from `v0.5.1` to `v0.6.1`.
- Added CI coverage for more compilers: Clang on Linux and clang-cl on Windows (`make cpp-dev CPP_CMAKE_TOOLSET="-T ClangCL"`) now gate CI; MinGW-w64 and Intel oneAPI (`icpx`) run as non-blocking jobs. See `bindings/cpp/CMAKE.md`'s new "Compiler Support" table.

**R:**

- Removed the `rfastlowess-package` pkgdown topic, which duplicated the adapter class list, and unexported the internal `Nullable()` helper.
- Fixed `_pkgdown.yml` describing the core interface as "R6 classes" when the package actually uses S3 classes.
- Merged `vignettes/parameters.Rmd`'s parameter reference (ranges, defaults, and fraction-choice guidance) into the `@param`/`@details` roxygen docs of `Lowess()`, `StreamingLowess()`, and `OnlineLowess()`, and removed the now-redundant vignette.
- Merged `vignettes/batch.Rmd`, `streaming.Rmd`, and `online.Rmd`'s unique content (When to Use guidance, merge strategy comparison) into the `@description`/`@details` roxygen docs of `Lowess()`, `StreamingLowess()`, and `OnlineLowess()`, and removed the now-redundant vignettes and their orphaned `gap_handling.svg`/`online_comparison.svg` diagrams.

**Node.js:**

- Updated `napi` to v3.12.
- Updated `napi-derive` to v3.6.
- Updated `napi-build` to v2.4.
- Updated `typedoc-plugin-markdown` to v4.13.
- `make nodejs-dev` now runs `npm update` after `npm install`, so dependencies are kept current.

**WASM:**

- Updated `typedoc-plugin-markdown` to v4.13.
- `make wasm-dev` now runs `npm update` after `npm install`, so dependencies are kept current.

**lowess:**

- Updated `wide` to v1.7.

### Fixed

**Monorepo:**

- Fixed `docs.yml` triggering GitHub's "pages build and deployment" once per docs job; per-language jobs now upload artifacts, and a single final `deploy` job pushes to `gh-pages` once per run.
- Fixed `docs.yml`'s reliance on GitHub's legacy branch-based Pages deployment, which auto-triggered an unpinned "pages build and deployment" job on every `gh-pages` push. The former `deploy` job is now `build` (still pushes `_site` to `gh-pages` as a cache); publishing now goes through `actions/upload-pages-artifact` and a new `deploy` job using `actions/deploy-pages`. Requires the repo's Pages source set to "GitHub Actions".
- Fixed every benchmark category in `benchmarks/rfastlowess.R` failing with `attempt to apply non-function`: it called the R6-style `model$fit(x, y)`, but `fit` is an S3 generic (`fit(model, x, y)`), not a field on the `Lowess` object. Also fixed `benchmarks/stats_lowess.R` resolving its `output/` directory relative to the current working directory instead of the script's own location (unlike `rfastlowess.R`, which already did this correctly), so results could land outside `benchmarks/output/` depending on how the script was invoked.

**docs:**

- Fixed the "Handling Outliers" quickstart example (every binding, `lowess`/`fastLowess`) printing nothing: with only 6 points and `fraction = 0.5`, tricube weighting left just 2 effectively-weighted points, which a degree-1 fit reproduces exactly (zero residual). Bumped to `fraction = 0.7`, which correctly downweights the injected outlier.
- Fixed the R `OnlineLowess()` roxygen example printing one line per point (48 lines for a 50-point loop); it now collects the smoothed values and prints only `head(smoothed, 5)`.
- Fixed the R `add_point()` roxygen example always printing `NULL`, since a single call never reaches the default `min_points = 3`; it now uses `min_points = 2L` and shows the second (non-`NULL`) call's result.
- Fixed the R `robustness.Rmd` "Detecting Outliers" example printing 22 lines at the `weight < 0.5` threshold, most of them incidental noise rather than the 3 deliberately injected outliers; tightened to `weight < 0.05`, which isolates the points effectively excluded by the fit.
- Fixed the R `merge.Rmd` "Choosing Chunk Size and Overlap" example constructing a `StreamingLowess` model but never printing anything; it now prints the computed overlap size and its percentage of `chunk_size`.
- Fixed the R `use-case-genomics.Rmd` ChIP-seq example never calling `fit()`, so `result` referenced a stale variable from an earlier chunk and the smoothed line either failed to plot or didn't align with the current example's x-range; added the missing `result <- fit(model, positions, signal_noisy)` call.
- Fixed the R `use-case-real-time.Rmd` "Update Modes" example constructing a `"full"`-mode `OnlineLowess` model but never feeding it data or plotting a result; it now runs the same accumulate-and-plot pattern as the preceding example.
- Fixed the "Detecting Outliers" example's `robustness.md` page (C++, Node.js, WASM, and the `lowess`/`fastLowess` crates) printing an unbounded number of "is likely an outlier" lines; capped output at 5 lines, matching the already-capped Julia and Python versions and the R vignette fix above.
- Fixed the Julia `intervals.md` "Confidence Intervals" and "Standard Errors" examples each looping over all 100 points instead of a short sample; switched to `result.y[1:5]`/`result.confidence_lower[1:5]`/`result.standard_errors[1:5]`-style slicing, matching the already-concise Python version.

**fastLowess:**

- Fixed `cargo doc` failing with `unresolved link to`crate::doc::gpu_backend`` whenever documenting with a feature set that excludes `gpu` (e.g. `--features cpu`): `api.md`'s GPU Backend link isn't itself feature-gated, but the `gpu_backend` doc module was gated behind `#[cfg(all(doc, feature = "gpu"))]`. Changed to `#[cfg(doc)]` (matching every other doc submodule), since it's a plain Markdown page with no dependency on the `gpu` feature's actual code.
- Fixed a misleading comment on `binding_support::default_overlap()` claiming wasm/nodejs use a flat `500`-point overlap while others use `chunk_size / 10`; every binding actually computes `chunk_size / 10` via the same `build_streaming()` helper. No behavior changed, only the comment.

**C++:**

- Fixed `OnlineOptions`' `min_points` (was `3`, should be `2`) and `update_mode` (was `"full"`, should be `"incremental"`) defaults diverging from the Rust core; added a `k_default_min_points` constant alongside the existing default constants in `fastlowess.hpp`.
- Fixed several Doxygen rendering bugs (homepage showing `concepts.md` instead of `README.md`; blockquotes, admonitions, inline/display math, and `---` after a blockquote rendering as literal/broken text). `README.md` is now the main page, using Doxygen-native syntax (`\f$...\f$` math, blockquote admonitions, explicit `<hr>`).
- Fixed `ci-cpp.yml`'s macOS job warning that the pre-installed `aws/tap` Homebrew tap is untrusted; `brew untap aws/tap` now runs before `brew install llvm cppcheck`, since that tap isn't needed for this build.
- Fixed `ci-cpp.yml`'s Windows job installing `cppcheck` via Chocolatey, whose package is missing its `cfg/std.cfg` library files, causing `make cpp-dev`'s static analysis pass to be silently skipped; it now installs `cppcheck` via `winget` instead (matching the already-working `install-tools` target), with its install directory added to `$GITHUB_PATH`.
- Fixed `Doxyfile`'s `PROJECT_NAME` showing `"fastLowess"` (the separate Rust crate's name) instead of the actual CMake project/library name; changed to `"fastlowess-cpp"`.
- Fixed `Doxyfile`'s `FILE_PATTERNS` missing a space (`*.hpp*.h`), which Doxygen parses as a single malformed glob instead of two separate `*.hpp`/`*.h` patterns; changed to `*.hpp *.h *.md`.

**Julia:**

- Fixed `OnlineLowess`'s `min_points` (was `3`, should be `2`) and `update_mode` (was `"full"`, should be `"incremental"`) keyword-argument defaults diverging from the Rust core.
- Fixed the Documenter homepage: it was a stale, separately maintained `index.md` instead of the README, and the README's centered badge/logo HTML and markdownlint comment rendered as literal text. `make.jl` now regenerates `index.md` from `README.md` on every build.
- Fixed `release-julia-register.yml` extracting the matching version section from the root `CHANGELOG.md`, which includes every binding/crate's entries; it now extracts from the already Julia-filtered `bindings/julia/julia/docs/src/NEWS.md` instead, so the JuliaRegistrator release notes only cover Julia-relevant changes.
- Fixed `make julia-dev` failing with "empty intersection between `fastlowess_jll@X.Y.Z` and project compatibility ..." whenever a locally cached `Manifest.toml` still pinned an older `fastlowess_jll` version after a new one was published: `Pkg.resolve()` treats an already-pinned manifest entry as fixed and won't search the registry for an upgrade, even when the relaxed compat bound requires one. The `dev` target now runs `Pkg.update("fastlowess_jll")` before `Pkg.resolve()` to actively pick up the newly published version.

**Node.js:**

- Fixed the docs homepage never showing the README content ("Get Started" jumped straight to Installation): a new `dev/add-readme-to-docs.js` script embeds `README.md` below the hero (stripping its redundant `# LOWESS Project` H1, since the hero already shows the title), wired into `npm run docs` and `make nodejs-dev`.
- Fixed the docs build always emitting a `[@astrojs/sitemap] The Sitemap integration requires the site astro.config option` warning when the `SITE` environment variable isn't set (e.g. local builds); `astro.config.mjs` now falls back to the production GitHub Pages URL.
- Fixed every link on the "API Reference" page 404ing: TypeDoc preserves original casing (e.g. `classes/Lowess.md`) but Starlight lowercases route slugs and never strips `.md`. A new `dev/lowercase-typedoc-refs.js` script lowercases generated file names and rewrites internal links after `typedoc` runs, wired into `npm run docs`.
- Fixed `custom-weights.md`'s "Zero-weight windows" `:::caution` admonition closing one line early, leaving its second sentence rendered as plain, oddly-indented text below the callout instead of inside it.

**WASM:**

- Same fix as Node.js: `README.md` is now embedded via `dev/add-readme-to-docs.js`, wired into `npm run docs` and `make wasm-dev`.
- Fixed `concepts.md` figures (MkDocs-only `<figure>`/attr_list syntax) not rendering; converted to plain images with italicized captions.
- Fixed inline/display LaTeX math rendering as literal text; wired `remark-math`/`rehype-katex` into `astro.config.mjs`.
- Fixed the same `@astrojs/sitemap` warning as Node.js, with the same fallback in `astro.config.mjs`.
- Fixed the same "API Reference" 404s as Node.js, via the same `dev/lowercase-typedoc-refs.js` script.
- Fixed the same `custom-weights.md` "Zero-weight windows" admonition closing early as Node.js.

**Python:**

- Fixed `StreamingLowess`'s `fraction` default (was `0.3`, should be `0.67`) and `OnlineLowess`'s `fraction` (was `0.2`), `window_capacity` (was `100`), and `update_mode` (was `"full"`) defaults (should be `0.67`, `1000`, and `"incremental"` respectively) diverging from the Rust core and the batch `Lowess` default.
- Fixed the "API Reference" page rendering empty: its `api/index.md` toctree still referenced the pre-rename `python`/`python-streaming`/`python-online` document names; updated to `api`/`api-streaming`/`api-online`, matching the files' current names. Sphinx toctree entries omit the `.md` extension, so this was missed by the earlier rename's link verification.

**R:**

- Fixed `OnlineLowess()`'s `min_points` (was `3L`, should be `2L`) and `update_mode` (was `"full"`, should be `"incremental"`) defaults diverging from the Rust core; updated the roxygen docs and `man/OnlineLowess.Rd` to match.

**Rust:**

- Fixed inline/display LaTeX math rendering as literal text on docs.rs; added a `katex-header.html` that renders it client-side with KaTeX.
- Fixed every cross-reference link across the `lowess`/`fastLowess` crate docs leading nowhere: these pages are embedded into rustdoc via `#![doc = include_str!(...)]`, so plain relative links render verbatim instead of resolving. Converted them to proper intra-doc links (e.g. `crate::doc::concepts`), validated with `cargo doc --all-features -D warnings`.

## 3.1.0

### Added

**Monorepo:**

- Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.
- Added a GitHub workflow for running validation scripts.

**Julia:**

- `release-julia-register.yml` now automatically extracts the matching changelog section and appends it as release notes in the JuliaRegistrator comment, enabling auto-merge on major version bumps.

**R:**

- Added `lenght` gaurds for extra arguments.

**Node.js:**

- Added `npm run lint` to the `Lint` step in `ci-nodejs.yml`, so JavaScript source and test files are linted via `oxlint` on every CI run.

**WASM:**

- Added `npm run lint` to the `Lint` step in `ci-wasm.yml`, so JavaScript source and test files are linted via `oxlint` on every CI run.

**C++:**

- Added clang-tidy and cppcheck installation to Makefile.

### Changed

**Monorepo:**

- Split the monolithic `.github/workflows/ci.yml` into seven per-language workflow files: `ci-rust.yml`, `ci-python.yml`, `ci-julia.yml`, `ci-nodejs.yml`, `ci-wasm.yml`, `ci-cpp.yml`, and `ci-r.yml`. Each file carries the relevant `ci` (multi-OS matrix), `asan`, and `gpu` jobs for its language.
- Each crate/binding sub-Makefile now runs `dev/verify_snippets.py --lang <lang>` for its own language as the final step of `make default`. The root `docs-test` target remains as a convenience to run all languages at once.
- Split `dev/verify_snippets.py` into a lean orchestrator and a `dev/runners/` package. Each language has its own module (`python.py`, `julia.py`, `nodejs.py`, `r.py`, `wasm.py`, `rust.py`, `cpp.py`) containing its `run_<lang>()` function and a `skip_reason()` predicate. Shared types (`Snippet`, `RunResult`) and utilities live in `runners/base.py`; the registry (`RUNNERS`, `SKIP_CHECKS`) is exported from `runners/__init__.py`.

**docs:**

- Moved CHANGELOG and CONTRIBUTING guides to project root.
- Updated README files to be binding/crate specific instead of one generic README for all bindings/crates.

**R:**

- Simplified `bindings/r/Makefile`: replaced `Cargo.toml.orig` save/restore vendoring with `src/vendor-update.sh`; made `[workspace]` permanent in `src/Cargo.toml`; removed Bioconductor dependencies, redundant `cargo fmt --check`, `NAMESPACE` indentation post-processing, and `pkgdown::build_site` from the dev workflow.
- Moved R documentation from ReadTheDocs to GitHub Pages, served by pkgdown at <https://thisisamirv.github.io/lowess-project/r/>. The ReadTheDocs site no longer includes R-specific content.
- Changed R version dependency to 4.4.0 due to issues with installing Bioconducter packages on R < 4.4.0.
- Replaced the multi-step `install.packages` / `BiocManager::install` package installation logic in `bindings/r/Makefile` with a single [`pak`](https://pak.r-lib.org/)-based block. `pak` handles RSPM binary vs source selection automatically (including Linux), skips already-installed packages, and installs CRAN, Bioconductor (`bioc::` prefix), and R-universe packages in one call.
- `make r` (`default:`) now runs `R CMD INSTALL $(R_DIR)` directly; R's `configure` script handles Rust compilation from the committed `vendor.tar.xz`. The full dev workflow moves to `make r-dev`.

**Python:**

- Migrated Python documentation from MkDocs to Sphinx (with MyST-Parser and jupyter-sphinx). Code blocks now execute and embed output automatically via `jupyter-sphinx`.
- `make python` (`default:`) now installs to the user Python environment via `pip install --user`. The full dev workflow (venv setup, formatting, linting, testing, doc-snippet verification) moves to `make python-dev`.

**Julia:**

- Moved Julia documentation from ReadTheDocs to GitHub Pages, served by Documenter.jl at <https://thisisamirv.github.io/lowess-project/julia/stable/>. The ReadTheDocs site no longer includes Julia-specific content. Code blocks use Documenter.jl `@example` sections, which execute and embed output automatically during the docs build.
- `make julia` (`default:`) now builds the Rust library and installs the Julia package via `Pkg.develop`. The full dev workflow moves to `make julia-dev`.

**Node.js:**

- Moved Node.js documentation from ReadTheDocs to GitHub Pages, served by Starlight at <https://thisisamirv.github.io/lowess-project/nodejs/>. The ReadTheDocs site no longer includes Node.js-specific content. `dev/add-nodejs-outputs.js` runs as part of `make nodejs-dev`, executing each JavaScript code block in the docs and injecting its output back into the Markdown source.
- `make nodejs` (`default:`) now builds the native addon and links it globally via `npm link`. The full dev workflow moves to `make nodejs-dev`.
- Updated `oxlint` dependency to 1.80.

**WASM:**

- Moved WASM documentation from ReadTheDocs to GitHub Pages, served by Starlight at <https://thisisamirv.github.io/lowess-project/wasm/>. The ReadTheDocs site no longer includes WASM-specific content. `dev/add-wasm-outputs.js` runs as part of `make wasm-dev`, executing each JavaScript code block in the docs and injecting its output back into the Markdown source.
- `make wasm` (`default:`) now builds both the Node.js and web WASM targets and links the Node.js package globally via `npm link`. The full dev workflow moves to `make wasm-dev`.
- Updated `oxlint` dependency to 1.80.
- Replace the outdated `jetli/wasm-pack-action` workflow with `taiki-e/install-action`.

**C++:**

- Moved C++ documentation from ReadTheDocs to GitHub Pages, served by Doxygen at <https://thisisamirv.github.io/lowess-project/cpp/>. The ReadTheDocs site no longer includes C++-specific content.
- `make cpp` (`default:`) now only runs `cargo build`. The full dev workflow (formatting, linting, cbindgen idempotency, symbol export verification, cmake tests, valgrind, doc-snippet verification) moves to `make cpp-dev`.

**fastLowess:**

- Moved crate documentation from ReadTheDocs to <https://docs.rs/fastLowess>.
- `make fastLowess` (`default:`) now only runs `cargo build`. The full dev workflow moves to `make fastLowess-dev`.

**lowess:**

- Moved crate documentation from ReadTheDocs to <https://docs.rs/lowess>.
- `make lowess` (`default:`) now only runs `cargo build`. The full dev workflow moves to `make lowess-dev`.

### Fixed

**Monorepo:**

- Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare tool names resolved via `PATH`, matching the existing fix in `bindings/r/src/cargo-config.toml`.

**C++:**

- Fixed `make cpp` Windows CI failure (`cannot find -lgcc_eh`): the C++ binding's Makefile detected MinGW via `gcc -dumpmachine` and selected the GNU target, which then used the Rtools cross-compiler from the workspace `.cargo/config.toml`; that compiler delegated to `C:\mingw64\bin\ld.exe`, which lacks `lgcc_eh`. Fixed by always targeting `x86_64-pc-windows-msvc` on Windows, removing the MinGW detection branch entirely.
- Fixed clang-tidy warnings in `bindings/cpp/include/fastlowess.hpp`: replaced all `#if defined(_WIN32)` with `#ifdef _WIN32`, added `#include <cstdio>` for `stdin`/`fileno`/`_fileno`, replaced deprecated `std::getenv("USERPROFILE")` with `_dupenv_s` on Windows, and added `const` to `base` and `cmd` local variables.

**R:**

- Fixed Windows arm64 (R-Universe) build: `ar x` without a member name correctly resolves long-name archive entries (>16 chars stored as `/<offset>`); named extraction silently fails for such entries. Used `objcopy --remove-section=.idata$4` on each extracted `.dll` stub to strip the invalid relocations that lld 19 rejects, then `ar r` to re-insert.
- Fixed `ld.lld` crashing or dropping symbols (`WakeByAddressSingle`, `WaitOnAddress`) on Windows arm64: `--whole-archive` pulls every crate's raw-dylib stub for a given DLL into the link, but different crates' stubs cover different, non-overlapping symbols of that DLL — `--allow-multiple-definition` works on x86_64 but crashes lld's arm64pe backend. Fixed by dropping `--whole-archive` on `gnullvm`; normal archive resolution applies and nothing is lost since `entrypoint.c` already references the extendr init symbol directly.
- Fixed CRAN Windows build (`error: linker not found`): `cargo-config.toml` hardcoded linker/ar as `c:/rtools45/...` absolute paths, which break when Rtools is installed on a different drive. Fixed by using bare tool names resolved via `PATH`.
- Fixed CRAN Windows build (`cannot find -lgcc_eh`): the Rtools gcc lib directory is not writable on CRAN's server, and config-file `rustflags` does not reach build-script linker invocations. `Makevars.win` creates an empty stub via `touch` in `$(TARGET_DIR)/libgcc_mock/` and passes `LIBRARY_PATH` inline on `cargo build`. The path is resolved to an absolute path via `$(pwd)` at shell execution time — a relative path silently fails because Cargo invokes GCC to link build scripts from its own temp directory, not from `src/`.
- Fixed `Lowess(fraction = 0.3, 4)` incorrectly succeeding: `reject_extra_positional_args()` counted unnamed arguments but did not check their position, so a single unnamed arg in any non-first slot passed validation. The check now rejects any unnamed argument that is not in position 1.

**Python:**

- Enforced keyword-only arguments beyond the first positional allowance in `Lowess`, `StreamingLowess`, and `OnlineLowess`, matching R's behaviour: `Lowess(fraction, *, ...)`, `StreamingLowess(fraction, chunk_size, *, ...)`, `OnlineLowess(fraction, window_capacity, min_points, *, ...)`. The `.pyi` stubs were updated with the same `*` separator.

## 3.0.0

### Added

**Python:**

- Exposed a `backend` option (`"cpu"` default, `"gpu"`) on `Lowess`, gated behind an opt-in `gpu` Cargo feature not enabled in published wheels. Run `fastlowess.install_gpu()` to download a prebuilt GPU wheel, or build locally with `maturin develop --features gpu`.

**R:**

- Exposed a `backend` option (`"cpu"` default, `"gpu"`) on `Lowess`, gated behind an opt-in `gpu` Cargo feature not enabled in CRAN/Bioconductor releases. Run `install_gpu()` to download a prebuilt GPU library (requires restarting R), or build locally with `make -f bindings/r/Makefile WITH_GPU=1`.
- Introduced S3 generics `fit()`, `process_chunk()`, `finalize()`, and `add_point()`, replacing the previous list-closure API.
- `bindings/r/Makefile` now auto-installs [Air](https://posit-dev.github.io/air/) if missing, before running `air format`.
- Added a `reject_extra_positional_args()` helper to reject extra unnamed arguments.

**Julia:**

- Exposed a `backend` option (`"cpu"` default, `"gpu"`) on `Lowess`, gated behind an opt-in `gpu` Cargo feature not enabled in published JLL artifacts. Run `install_gpu()` to download a prebuilt GPU library, or build locally with `cargo build --release --features gpu`.

**Node.js:**

- Exposed a `backend` option (`"cpu"` default, `"gpu"`) on `Lowess`, gated behind an opt-in `gpu` Cargo feature not enabled in published npm binaries. Run `await fastlowess.installGpu()` to download a prebuilt GPU addon (requires restarting Node.js), or build locally with `napi build --features gpu`.

**C++:**

- Exposed a `backend` option (`"cpu"` default, `"gpu"`) on `Lowess`, gated behind an opt-in `gpu` Cargo feature not enabled by default. Call `fastlowess::gpu::install()` to download a prebuilt GPU library, or build locally with `cargo build --features gpu`.

**Docs:**

- Added `*See: …*` cross-reference links after every option heading in `docs/api/` files, pointing to the corresponding user-guide page.

### Fixed

**R:**

- Fixed incorrect URLs in R binding docs.

**Julia:**

- Fixed `LowessResult.iterations_used` returning the raw FFI sentinel `-1` instead of `nothing` when robustness iterations were not applicable.

**WASM:**

- Fixed `OnlineLowess.add_point()` returning `undefined` instead of `null` when the sliding window has not yet accumulated enough points.

**Docs:**

- Fixed `docs/api/rust.md` showing Rust enum variant names instead of the string option values that the API actually accepts.

### Changed

**Monorepo:**

- Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
- Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/lowess/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
- Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/lowess/` → `crates/lowess/tests/lowess/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages and `bindings/r/demo/`.

**lowess:**

- Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`, matching `LowessResult`. This is a **breaking change**.
- Updated `wide` to v1.6.

**fastLowess:**

- Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`, matching `LowessResult`. This is a **breaking change**.
- Disabled wgpu's default `dx12` and `gles` features (keeping `vulkan`/`metal`) — both pulled in Windows DLLs not present on every system, causing `--features gpu` builds to fail to even load rather than just failing to find a GPU adapter.
- Exposed GPU backend in `binding_support.rs`.

**Python:**

- Renamed `OnlineOutput`'s `smoothed` and `std_error` properties to `y` and `standard_error`. This is a **breaking change**.

**R:**

- Renamed the `smoothed` and `std_error` fields returned by `OnlineLowess`'s `add_point()` to `y` and `standard_error`. This is a **breaking change**.
- Replaced `dev/style_pkg.R` with [Air](https://posit-dev.github.io/air/) for formatting.
- Removed `dev/fix_rd_style.R`, `dev/prepare_cargo.py`, `dev/patch_vendor_crates.py`, `dev/clean_checksums.py`, and `dev/prepare_cran.sh` — their logic is now inlined directly in `bindings/r/Makefile`, so the R build no longer requires any Python scripts.
- Added `...` to `Lowess()`, `StreamingLowess()`, and `OnlineLowess()` to force named arguments for optional parameters.
- Added `Depends: R (>= 4.6)` to `DESCRIPTION` and a matching CI matrix entry.
- Expanded roxygen2 `@param` docs and added a `See Also` section linking to <https://lowess.readthedocs.io/>.
- Expanded `rfastlowess-intro.Rmd` vignettes.

**Julia:**

- Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`. This is a **breaking change**.
- Removed `dev/format_julia.jl`; formatting is now inlined in `bindings/julia/Makefile`.

**Node.js:**

- Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`. This is a **breaking change**.
- Updated `@napi-rs/cli` to v3.8 and `oxlint` to v1.79.

**WASM:**

- Renamed `OnlineOutput`'s `smoothed` and `std_error` getters to `y` and `standard_error`. This is a **breaking change**.
- Updated `oxlint` to v1.79.

**C++:**

- Renamed `OnlineOutput`'s `smoothed()` and `std_error()` methods to `y()` and `standard_error()`. This is a **breaking change**.

**Docs:**

- Split `StreamingLowess`/`OnlineLowess` content out of each binding's main API reference page into dedicated `{lang}-streaming.md`/`{lang}-online.md` files.
- Moved the `tutorials/` pages into a new `user-guide/use-cases/` section.
- Standardized `docs/api/` code examples across every binding, with expected output comments.
- `dev/verify_snippets.py` now also runs the R code chunks in vignettes.

## 2.0.0

### Added

**lowess and fastLowess:**

- Added `iterations_used: Option<usize>` field to `OnlineOutput<T>`, reporting the number of robustness iterations performed when `UpdateMode::Full` is active. Returns `Some(0)` for the degenerate two-point linear fit and `None` when `UpdateMode::Incremental` is used.
- Added `ParseErrors(Vec<LowessError>)` variant to `LowessError`, which collects all string-parse failures that accumulate in the builder and reports them together when `build()` is called.
- Added `"take_first"` and `"take_last"` as accepted string aliases for `MergeStrategy::TakeFirst` and `MergeStrategy::TakeLast`.
- Added `"resmooth"` as an accepted string alias for `UpdateMode::Full` and `"single"` as an alias for `UpdateMode::Incremental`, aligning string-parse behaviour with the `loess-rs` crate.
- Added `custom_weights(Vec<T>)` builder method on `LowessBuilder` (Batch adapter only). Accepts a vector of non-negative per-observation weights that are multiplied into the distance and robustness weights before each local regression, allowing known-bad points to be suppressed (`0.0`) or high-quality measurements to be emphasised.
- Centralized all `impl FromStr` blocks for the seven option enums (`WeightFunction`, `BoundaryPolicy`, `ScalingMethod`, `RobustnessMethod`, `ZeroWeightFallback`, `MergeStrategy`, `UpdateMode`) directly in `api.rs`, consolidating previously scattered implementations into a single source of truth. Parse and canonical-name helpers are exposed via `lowess::internals::alias` (requires `dev` feature), allowing `fastLowess::binding_support` to delegate all string-to-enum parsing through that path.
- Added module-level `defaults.rs` files within each sub-module (`math/`, `algorithms/`, `adapters/`) to centralize default values close to the types they govern, propagating them from a single source of truth to ensure consistency across bindings and crates.

**Python:**

- Added `OnlineOutput` class to the Python binding. `OnlineLowess.add_point()` now returns `OnlineOutput | None` instead of `float | None`, exposing `smoothed`, `std_error`, `residual`, `robustness_weight`, and `iterations_used`.
- Added `custom_weights` parameter to the `Lowess.fit(x, y, custom_weights=None)` method. Accepts a `list[float]` of non-negative per-observation weights. Batch only.
- Removed `smooth()`, `smooth_streaming()`, and `smooth_online()` convenience function stubs from `_core.pyi`.

**R:**

- Added `custom_weights` parameter to `fit(Lowess, )`. Accepts a numeric vector of non-negative per-observation weights. Batch only.

**Julia:**

- Added `custom_weights` keyword argument to `fit(model, x, y; custom_weights)`. Accepts a `Vector{Float64}` of non-negative per-observation weights. Batch only.

**Node.js:**

- Added `OnlineOutput` object to the Node.js binding. `OnlineLowess.add_point()` now returns `OnlineOutput | null` instead of `number | null`, exposing `smoothed`, `std_error`, `residual`, `robustness_weight`, and `iterations_used`.
- Added `return_se` and `cv_seed` fields to `SmoothOptions`.
- Added `customWeights` as an optional per-call argument to `fit(x, y, customWeights?)` and `fit_async(x, y, customWeights?)`. Accepts a `Float64Array` of non-negative per-observation weights. Includes pre-flight length-mismatch and non-negative validation. Batch only.
- Added JavaScript-layer option key validation: unknown keys in `SmoothOptions`, `StreamingOptions`, or `OnlineOptions` now throw a `TypeError` listing all valid keys, via wrapper classes around the native NAPI exports.

**WebAssembly:**

- Added `custom_weights` field to `LowessOptions` (passed in the options object to `smooth()`). Accepts a `Float64Array` of non-negative per-observation weights. Batch only.

**C++:**

- Added `custom_weights` field to `LowessOptions` and a second overload of `Lowess::fit()` that accepts a `const std::vector<double>& custom_weights` argument. Values must be non-negative and length must match the input data. Batch only.

### Changed

**Monorepo:**

- Renamed all public API method and option names from camelCase to snake_case across every binding and all documentation. This is a **breaking change** for all consumers of the C++, Node.js, and WASM APIs.
- Converted all documentation tables to compact single-space format.
- Updated `.clang-tidy` to configure `lower_case` as the required naming convention for functions and member functions, matching the new snake_case public API.
- Moved `BENCHMARKS.md`, `CHANGELOG.md`, and `CONTRIBUTING.md` from the repository root into `docs/` and added them to the documentation site navigation.
- Added a `[patch.crates-io]` section to the root `Cargo.toml` so all workspace bindings resolve `fastLowess` and `lowess` to the local workspace crates during development, replacing the previously-used registry (crates.io) versions.
- Eliminated all local `parse_*` functions that each binding previously duplicated independently. Option parsing and builder application now delegates to `fastLowess::binding_support`, ensuring consistent aliases, validation messages, and behaviour across every language frontend.
- Replaced direct use of `KFold` / `LOOCV` constructor types in the cross-validation path with `binding_support::apply_cross_validation`.
- Split the Julia release CI workflow into two separate workflows: `release-julia-jll.yml` (triggered on release, opens the Yggdrasil PR) and `release-julia-register.yml` (manual dispatch, triggers JuliaRegistrator once the JLL PR is merged).
- Major documentation improvements.

**lowess and fastLowess:**

- Added `Lowess<T>`, `StreamingLowess<T>`, and `OnlineLowess<T>` type aliases as the primary user-facing constructors (e.g. `StreamingLowess::new().chunk_size(50).build()`). Mode-specific builder methods (`chunk_size`, `overlap`, `window_capacity`, `min_points`, `update_mode`) are now called directly on the type alias rather than after `.adapter()`.
- Made `BatchLowessBuilder`, `StreamingLowessBuilder`, and `OnlineLowessBuilder` internal-only: all public setter methods have been removed from these types. All smoothing configuration now flows through `LowessBuilder<T, Mode>` (exposed via the type aliases above). This is a **breaking change** for any code that called setter methods on an adapter builder directly.
- Changed all enum-typed builder methods to accept strings instead: `weight_function`, `robustness_method`, `scaling_method`, `boundary_policy`, `zero_weight_fallback`, `merge_strategy`, and `update_mode` now take `impl IntoEnum<T>` (accepting both enum variants and strings such as `.weight_function("tricube")`) rather than requiring enum variants to be imported. This is a **breaking change** for any code passing enum variants directly.
- Inlined the `IntoEnum<E>` trait and its macro-generated impls for all enum-typed builder parameters directly into `api.rs` (`lowess`) and `binding_support.rs` (`fastLowess`), eliminating a previously separate `parse` module. This allows builder methods to accept either a typed enum value (e.g. `.weight_function(WeightFunction::Tricube)`) or a string (e.g. `.weight_function("tricube")`) interchangeably.
- Replaced the `cross_validate(CVConfig)` builder method (which required importing `KFold` or `LOOCV` types) with a string-based cross-validation API: `.cv_method("kfold")` / `.cv_method("loocv")`, `.cv_k(n)`, `.cv_fractions(vec![...])`, and `.cv_seed(n)`. `KFold` and `LOOCV` are no longer exported from the prelude. This is a **breaking change** for any code using the old `cross_validate` API.
- Added a `binding_support` module providing shared helpers for all language binding frontends: string-to-enum parse functions (`parse_weight_function`, `parse_robustness_method`, `parse_scaling_method`, `parse_boundary_policy`, `parse_zero_weight_fallback`, `parse_merge_strategy`, `parse_update_mode`), matching canonical-string display functions, `BuilderOptionSet` / `TypedBuilderOptionSet` structs, and `apply_builder_options` / `apply_typed_builder_options` / `apply_cross_validation` helpers. This consolidates previously duplicated logic that was scattered across every binding into a single source of truth.
- Renamed the internal `auto_convergence` struct field to `auto_converge` on `BatchLowessBuilder`, `OnlineLowessBuilder`, `StreamingLowessBuilder`, and the executor config types, making the field name consistent with the existing `auto_converge()` setter method. This is a **breaking change** for any code that accessed these fields directly.
- Changed `build()` to wrap all accumulated string-parse errors in a `LowessError::ParseErrors(Vec<LowessError>)` value instead of surfacing only the first error. This is a **breaking change** for code that matched on `LowessError::InvalidOption` as the error returned from `build()`.
- Made the `IntoEnum<E>` trait `pub(crate)` in both `lowess` and `fastLowess`, restricting it to crate-internal use. Callers do not need to name this trait; builder methods continue to accept both enum variants and string literals unchanged.
- Updated `wide` dependency to v1.5, `wgpu` to v30.0, and `pollster` to v1.0.

**fastLowess:**

- `Lowess`, `StreamingLowess`, and `OnlineLowess` are now dedicated wrapper structs around `LowessBuilder<f64>` with string-accepting forwarding methods, rather than type aliases re-exported from the base `lowess` crate. Each wrapper's `build()` delegates to the corresponding parallel adapter and defaults to parallel execution. This is a **breaking change**: replace `.adapter(Batch).build()` with `.build()`, `Lowess::new().adapter(Streaming)` with `StreamingLowess::new()`, and `Lowess::new().adapter(Online)` with `OnlineLowess::new()`.
- The `fastLowess` prelude now exports only `{Lowess, LowessError, LowessResult, OnlineLowess, StreamingLowess}`, removing `LowessBuilder`, `Adapter::{Batch, Online, Streaming}`, and `Backend::{CPU, GPU}`. This is a **breaking change** for code that relied on those names being in scope via `use fastLowess::prelude::*`.

**C++:**

- Renamed all public member functions to snake_case: `make_error()`, `has_value()`, `r_squared()`, `effective_df()`, `residual_sd()`, `x_value()`, `y_value()`, `x_vector()`, `y_vector()`, `standard_errors()`, `confidence_lower()`, `confidence_upper()`, `prediction_lower()`, `prediction_upper()`, `robustness_weights()`, `fraction_used()`, `iterations_used()`, `process_chunk()`, `add_points()`.
- Replaced `Expected<LowessResult> OnlineLowess::add_points(const std::vector<double>&, const std::vector<double>&)` with `Expected<std::optional<double>> OnlineLowess::add_point(double x, double y)`. The method now processes a single point and returns only that point's smoothed value, or `std::nullopt` if not enough points have been accumulated yet. The underlying C FFI symbol is renamed from `cpp_online_add_points` to `cpp_online_add_point`. This is a **breaking change**.

**Node.js:**

- Renamed all `Diagnostics`, `SmoothOptions`, `StreamingOptions`, and `OnlineOptions` interface fields to snake_case (`r_squared`, `effective_df`, `residual_sd`, `chunk_size`, `merge_strategy`, `window_capacity`, `min_points`, `update_mode`, and all smoothing option fields).
- Renamed binding methods to snake_case: `fit_async`, `process_chunk`, `add_points`.
- Renamed `LowessResultObj` getters to snake_case: `standard_errors`, `confidence_lower`, `confidence_upper`, `prediction_lower`, `prediction_upper`, `robustness_weights`, `cv_scores`, `fraction_used`, `iterations_used`.
- Updated `index.d.ts` to reflect all renamed fields and methods.
- Replaced `add_points(x: Float64Array, y: Float64Array): LowessResultObj` on `OnlineLowess` with `add_point(x: number, y: number): OnlineOutput | null`. The method now processes a single point and returns an `OnlineOutput` object, or `null` if not enough points have been accumulated yet. This is a **breaking change**.
- Changed default `OnlineOptions.window_capacity` from 100 to 1000 and `OnlineOptions.min_points` from 2 to 3, matching the defaults used by the loess binding.
- `OnlineLowess` now forwards all `SmoothOptions` fields to the underlying builder (previously only `fraction`, `iterations`, and `parallel` were forwarded; all other fields were hardcoded to `None`/`false`).
- Updated `napi-rs/cli` dependency to v3.7 and `oxlint` to v1.73.

**WASM:**

- Renamed all JS-facing option keys to snake_case by removing `#[serde(rename = "camelCase")]` attributes from `SmoothOptions`, `StreamingOptions`, and `OnlineOptions`. JSON passed from JavaScript must now use snake_case keys.
- Updated `Diagnostics` getter names to snake_case: `r_squared`, `effective_df`, `residual_sd`.
- Renamed the `update(x: number, y: number)` method on `OnlineLowess` to `add_point(x: number, y: number)`. This is a **breaking change**.
- Updated `oxlint` dependency to v1.73.

**Python:**

- Renamed the `update(x, y)` method on `OnlineLowess` to `add_point(x, y)` and removed the separate array-based `add_points(x, y)` method. `add_point` processes a single point and returns the smoothed value as `float | None`. This is a **breaking change**.
- Updated `pyo3` and `numpy` dependencies to v0.29.

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
- Fixed `.build()` errors incorrectly using `Status::GenericFailure`; they now return `Status::InvalidArg` since build failures originate from invalid configuration (accumulated parse errors), not from runtime execution.
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
