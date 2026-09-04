---
title: "News"
weight: 100
---

<!-- markdownlint-disable MD024 MD025 -->
# fastlowess (Go) 3.2.1

## Added

* Added `dev/bump_version.py --version X.Y.Z` to bump every crate/binding's version files, `CITATION.cff`, and the Spack recipe in one pass (supports `--dry-run`); now also bumps `Project.toml`'s `fastlowess_jll` compat floor (safe pre-publish since `make julia-dev`/CI relax it to an OR-list at test-time).
* Added an optional `commit` input to every release workflow's `workflow_dispatch` trigger, to pin the built commit for manual runs.
* Added an `aarch64-pc-windows-gnullvm` linker entry to the root `.cargo/config.toml`, matching the existing `x86_64-pc-windows-gnu` one; makes local arm64 Windows builds work without a manual env var.
* Added a `dev/check_pinned_versions.py` pin for the docs-site MathJax CDN version.
* Added ARM64 release binaries to `release-go.yml` (native Linux, cross-compiled Windows via `aarch64-pc-windows-gnullvm` + llvm-mingw), with the same macOS x64/`macos-13` mislabeling fix as C++.
* Added an arm64 job to `ci-go.yml` using the same llvm-mingw toolchain, so arm64 support is verified on every push.

## Changed

* Added four new pins to `dev/check_pinned_versions.py`: R's `rextendr`/`roxygen2` versions and the vendored KaTeX CDN version in both Rust crates.
* Changed `check-versions.yml` to open/update a GitHub issue instead of failing CI when a pin goes stale or unreachable.
* Bumped docs-site MathJax CDN version from `3.2.2` to `4.1.3`, updating `dev/check_pinned_versions.py`'s pattern for MathJax 4's CDN layout.

## Fixed

* Fixed a handful of `R²`/`O(n²)` Unicode superscripts the earlier ASCII-fication pass missed (added/edited after it ran) — R tests and several `lowess` crate doc-comments — replaced with `R2`/`O(n^2)`.
* Fixed `release-conda.yml`'s `sed` patterns for the feedstock's new rattler-build `recipe/recipe.yaml` format, removing now-dead R-package-name-fix/Python-dependency-injection/`build_r.sh` steps.
* Fixed `dev/check_links.py` false-flagging valid R vignette links to a sibling's rendered `.html` (R CMD build renders `.Rmd`→`.html`) as broken.
* Fixed every binding's/crate's docs and doc-comments describing `LowessResult.x` (and equivalents) as "Sorted x values"; it's actually returned in the same order as the input `x` (the algorithm sorts internally, then un-sorts every output field back to the original order). Also strengthened Python's `test_unsorted_input` to assert this instead of only checking output length.
* Fixed `.github/dependabot.yml`'s `cargo` entry for `/bindings/r/src`, which could never succeed: its `fastLowess = { path = "vendor/fastLowess" }` path dependency is only committed as `vendor.tar.xz`, never as loose files Dependabot can read. Removed the entry and added `extendr-api`'s version to `dev/check_pinned_versions.py` instead, which also uncovered and fixed a version-comparison bug there: comparing raw tuples treated a shorthand pin like `"0.9"` as older than `"0.9.0"` due to tuple-length tiebreaking; now padded to equal length first.
* Fixed pkg.go.dev showing "License: None detected" and failing tagged/stable checks: `LICENSE-MIT`/`LICENSE-APACHE` lived one directory above the actual Go module. Copied both into `bindings/go/fastlowess`; `release-go.yml` now also pushes a nested-module `vX.Y.Z` tag.
* Fixed `OnlineOptions`'s `MinPoints`/`UpdateMode` defaults (`3`/`"full"`) diverging from every other binding and the Rust core; now `2`/`"incremental"`.
* Fixed `bindings/go/Makefile` and `ffi.go`'s cgo `LDFLAGS` both unconditionally targeting the x64 GNU build on any Windows host, which would have cross-compiled/linked for the wrong architecture on arm64.

# fastlowess (Go) 3.2.0

## Added

* Added `dev/add-readme-to-docs.py`, which auto-detects the Hugo (Go, Java) vs Starlight (Node.js, WASM) docs-site flavor and embeds `README.md` accordingly; wired into the corresponding Makefiles and `package.json` scripts.
* Added `.github/dependabot.yml`, covering every dependency ecosystem in the repo (`github-actions`, `cargo`, `npm`, `pip`, `maven`, `gomod`). Each directory is grouped so all its updates, including majors, land in a single weekly PR.
* Added `dev/check_pinned_versions.py` and a weekly `.github/workflows/check-versions.yml`, which check hardcoded tool/library version pins that Dependabot can't see (Corrosion's CMake `FetchContent` tag, the vendored doxygen-awesome-css theme, the Checkstyle jar, golangci-lint, and Hugo) against their latest GitHub release and fail CI if any are outdated. Read-only: it never opens PRs or edits files itself.
* Added a new Go binding (`bindings/go`), consuming the `fastLowess` Rust core via a dedicated `cgo`-compatible C ABI (`fastlowess-go` crate, mirroring the C++ binding's FFI approach).

## Changed

* Modified `verify_snippets.py` to verify snippets and also add the output of the snippets to the markdown file.
* Added a `large` benchmark category (n = 50000) to `benchmarks/rfastlowess.R` and `benchmarks/stats_lowess.R`, since every existing category ran in well under 100ms. Covers 4 scenarios (`large_delta_0`, `large_delta_0.1`, `large_high_iter`, `large_high_fraction`) stressing `delta`, iteration count, and fraction. `benchmarks/compare.py`'s plot grid grew from 5x2 to 7x2 to fit them.
* Added `.gitattributes`, normalizing all text files to LF line endings (`* text=auto eol=lf`) and marking binary formats (images, archives, compiled libraries, `.rds`/`.RData`, etc.) so Git never treats them as text.
* Consolidated every crate/binding README: merged the "Installation" and "Documentation" sections, replaced GitHub-only alert syntax with plain blockquotes, removed the redundant "API Reference" and "Changelog" sections (each now has its own docs-site page), shortened the "GPU Backend" blurb, and added a "Read more" link to the Concepts page. The top-level repository README is unchanged, since it's only ever viewed on GitHub.
* Renamed the batch adapter's "When to Use" heading to "When to Use Batch Adapter" across every binding/crate's API docs.
* Vendored the [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) theme (v2.4.2) for a modern, sidebar-only cpp Doxygen site with automatic dark mode.
* Added `dev/update_changelogs.py`, which regenerates a per-binding/crate `NEWS.md`/`news.md` from the root `CHANGELOG.md`. Wired into every docs site's navigation, the Rust crates' rustdoc module tree, and every `Makefile` `dev` target.
* Replaced `kernels.md`'s "Choosing a Kernel" mermaid flowchart (every binding/crate) with an equivalent decision table, since Doxygen and rustdoc don't render mermaid and the deeply-nested diamond chain was hard to read even where it did render.
* Replaced `adapter-choice.md`/`adapters.md`'s "Overview" flowchart (mermaid in most bindings/crates, ASCII art in the C++ docs) with an equivalent decision table, unifying on a single rendering-agnostic format across every binding/crate.
* Moved the duplicated "GPU Acceleration" section out of `api.md` (C++, Node.js, Python, `fastLowess`) into each one's dedicated `gpu-backend.md` guide, which also gained "Hardware Requirements"/"Performance Considerations"; `api.md` now just links to it. Removed the section entirely from the `lowess` crate, which has no `gpu` feature.
* Consolidated `parameters.md`/the auto-generated parameter reference across every binding and crate: merged its unique content (fraction/iterations guidance, `delta` defaults, `zero_weight_fallback` behavior) into each `api.md`'s option tables, then removed `parameters.md` and its nav entries/rustdoc module — the option lists and examples it duplicated already live on their own pages.
* Standardized docs across all bindings and crates.
* Replaced the Unicode superscript `²` character with plain ASCII throughout every doc page and doc-comment (`R²` → `R2`; `(y_true - y_pred)²`/`O(window²)` → `^2`), including the `wasm` binding's JSDoc comments, the Julia docstrings in `FastLOWESS.jl`, and the `fastLowess`/`lowess` crates' rustdoc examples. Also changed the `lowess` crate's `Diagnostics` `Display` impl to print `R2` instead of `R²` so the `fastLowess` doc example showing its captured output stays accurate.
* Harmonized the docs-site directory structure across every binding/crate to mirror Go's layout (`introduction/`, `guide/`, `weighting/`, `advanced/`, `use-case/`, `api/`, each grouped under a hub page): Node.js/WASM (Starlight `sidebar`), Python (Sphinx toctree, also removed the dead `mkdocs.yml`), Julia (`Documenter.jl` `pages=[...]`, switched to recursive `walkdir`), and `fastLowess`/`lowess` (nested `#[cfg(doc)]` modules). C++'s Doxygen pages were physically moved to match (`RECURSIVE: YES`), with two hub pages renamed for parity. Java already matched this layout and now carries it into its new Antora site. R's `vignettes/` stays flat (CRAN/pkgdown requirement).

## Fixed

* Fixed `docs.yml` triggering GitHub's "pages build and deployment" once per docs job; per-language jobs now upload artifacts, and a single final `deploy` job pushes to `gh-pages` once per run.
* Fixed `docs.yml`'s reliance on GitHub's legacy branch-based Pages deployment, which auto-triggered an unpinned "pages build and deployment" job on every `gh-pages` push. The former `deploy` job is now `build` (still pushes `_site` to `gh-pages` as a cache); publishing now goes through `actions/upload-pages-artifact` and a new `deploy` job using `actions/deploy-pages`. Requires the repo's Pages source set to "GitHub Actions".
* Fixed every benchmark category in `benchmarks/rfastlowess.R` failing with `attempt to apply non-function`: it called the R6-style `model$fit(x, y)`, but `fit` is an S3 generic (`fit(model, x, y)`), not a field on the `Lowess` object. Also fixed `benchmarks/stats_lowess.R` resolving its `output/` directory relative to the current working directory instead of the script's own location (unlike `rfastlowess.R`, which already did this correctly), so results could land outside `benchmarks/output/` depending on how the script was invoked.
* Fixed the "Handling Outliers" quickstart example (every binding, `lowess`/`fastLowess`) printing nothing: with only 6 points and `fraction = 0.5`, tricube weighting left just 2 effectively-weighted points, which a degree-1 fit reproduces exactly (zero residual). Bumped to `fraction = 0.7`, which correctly downweights the injected outlier.
* Fixed the R `OnlineLowess()` roxygen example printing one line per point (48 lines for a 50-point loop); it now collects the smoothed values and prints only `head(smoothed, 5)`.
* Fixed the R `add_point()` roxygen example always printing `NULL`, since a single call never reaches the default `min_points = 3`; it now uses `min_points = 2L` and shows the second (non-`NULL`) call's result.
* Fixed the R `robustness.Rmd` "Detecting Outliers" example printing 22 lines at the `weight < 0.5` threshold, most of them incidental noise rather than the 3 deliberately injected outliers; tightened to `weight < 0.05`, which isolates the points effectively excluded by the fit.
* Fixed the R `merge.Rmd` "Choosing Chunk Size and Overlap" example constructing a `StreamingLowess` model but never printing anything; it now prints the computed overlap size and its percentage of `chunk_size`.
* Fixed the R `use-case-genomics.Rmd` ChIP-seq example never calling `fit()`, so `result` referenced a stale variable from an earlier chunk and the smoothed line either failed to plot or didn't align with the current example's x-range; added the missing `result <- fit(model, positions, signal_noisy)` call.
* Fixed the R `use-case-real-time.Rmd` "Update Modes" example constructing a `"full"`-mode `OnlineLowess` model but never feeding it data or plotting a result; it now runs the same accumulate-and-plot pattern as the preceding example.
* Fixed the "Detecting Outliers" example's `robustness.md` page (C++, Node.js, WASM, and the `lowess`/`fastLowess` crates) printing an unbounded number of "is likely an outlier" lines; capped output at 5 lines, matching the already-capped Julia and Python versions and the R vignette fix above.
* Fixed the Julia `intervals.md` "Confidence Intervals" and "Standard Errors" examples each looping over all 100 points instead of a short sample; switched to `result.y[1:5]`/`result.confidence_lower[1:5]`/`result.standard_errors[1:5]`-style slicing, matching the already-concise Python version.

# fastlowess (Go) 3.1.0

## Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.
* Added a GitHub workflow for running validation scripts.

## Changed

* Split the monolithic `.github/workflows/ci.yml` into seven per-language workflow files: `ci-rust.yml`, `ci-python.yml`, `ci-julia.yml`, `ci-nodejs.yml`, `ci-wasm.yml`, `ci-cpp.yml`, and `ci-r.yml`. Each file carries the relevant `ci` (multi-OS matrix), `asan`, and `gpu` jobs for its language.
* Each crate/binding sub-Makefile now runs `dev/verify_snippets.py --lang <lang>` for its own language as the final step of `make default`. The root `docs-test` target remains as a convenience to run all languages at once.
* Split `dev/verify_snippets.py` into a lean orchestrator and a `dev/runners/` package. Each language has its own module (`python.py`, `julia.py`, `nodejs.py`, `r.py`, `wasm.py`, `rust.py`, `cpp.py`) containing its `run_<lang>()` function and a `skip_reason()` predicate. Shared types (`Snippet`, `RunResult`) and utilities live in `runners/base.py`; the registry (`RUNNERS`, `SKIP_CHECKS`) is exported from `runners/__init__.py`.
* Moved CHANGELOG and CONTRIBUTING guides to project root.
* Updated README files to be binding/crate specific instead of one generic README for all bindings/crates.

## Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare tool names resolved via `PATH`, matching the existing fix in `bindings/r/src/cargo-config.toml`.

# fastlowess (Go) 3.0.0

## Changed

* Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
* Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/lowess/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
* Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/lowess/` → `crates/lowess/tests/lowess/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages and `bindings/r/demo/`.

# fastlowess (Go) 2.0.0

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

# fastlowess (Go) 1.3.0

## Added

* Added prerequisites for different bindings and platforms to `CONTRIBUTING.md`
* Updated `docs/assets/diagrams/lowess_smoothing_concept.svg` to correctly illustrate LOWESS concepts (robustness iterations, bisquare re-weighting, outlier downweighting) instead of the generic LOESS algorithm it previously depicted.
* Modified `docs/requirements.txt` to update the versions of the documentation dependencies.
* Improved CI tests and coverage.
* Modified Makefile to be truely cross-platform.
* Added sanitizer check for all bindings and crates.

## Fixed

* Fixed R ASAN tests failing to compile vignettes by passing `--no-build-vignettes` to `rcmdcheck`.
* Upgraded ASAN test environment to use modern `rocker/r-devel-san:latest` image and `RDscript` to resolve outdated `readelf` warnings.
* Fixed `Makefile` idempotency checks on Linux by providing a default `/tmp` fallback for the `TEMP` directory variable.
* Fixed accidental root `Cargo.toml` workspace isolation leaks by adding checked-in `pre-commit` and `pre-push` git hook guards that restore `Cargo.toml.bak` when present and fail loudly if required workspace members are still commented out.
* Added a repo-local `.cargo/config.toml` that sets `CC=clang-cl` for `x86_64-pc-windows-msvc`, fixing Criterion 0.8 benchmark builds on Windows when `cc-rs` would otherwise pick `clang.exe` and fail to link `alloca`.

# fastlowess (Go) 1.2.0

## Fixed

* Fixed project logo.

# fastlowess (Go) 0.99.9

## Changed

* Bump rust version to 1.88 for better stability
* Change function-based builder pattern in the bindings to class-based builder pattern, allowing true streaming and online processing
* Improve API docs

# fastlowess (Go) 0.99.7

## Fixed

* Fix README file links
* Fix Makefile bug with R versioning

# fastlowess (Go) 0.99.6

## Fixed

* Fix README file formats and links

# fastlowess (Go) 0.99.5

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

For the full changelog, see:
<https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md>
