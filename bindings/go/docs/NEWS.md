---
title: "News"
weight: 100
---

<!-- markdownlint-disable MD024 MD025 -->
# fastlowess (Go) 3.2.0

## Added

* Added a new Go binding (`bindings/go`), consuming the `fastLowess` Rust core via a dedicated `cgo`-compatible C ABI (`fastlowess-go` crate, mirroring the C++ binding's FFI approach).

## Changed

* Added a `large` benchmark category to `benchmarks/rfastlowess.R` and `benchmarks/stats_lowess.R` (n = 50000, `delta = 0` to disable `stats::lowess`'s interpolation shortcut for a fair, exact-computation comparison), since every existing category completed in well under 100ms. Reliably takes ~13-16s for `stats::lowess` and ~3s for `fastLowess` (serial). Expanded it to 4 scenarios stressing different parameters: `large_delta_0` (the original exact-fit case), `large_delta_0.1` (same workload with `delta` left at its default/auto value, showing the interpolation shortcut's speedup), `large_high_iter` (10 robustness iterations instead of 3), and `large_high_fraction` (n = 20000, `fraction = 0.67`, since a 0.67 fraction at n = 50000 takes over a minute). `benchmarks/compare.py`'s plot grid grew from 5x2 to 7x2 rows to fit the new categories (`large_delta_0`/`large_delta_0.1` share one "large_delta" chart, matching how `fraction_*`/`iterations_*` already group into shared charts).
* Consolidated every crate/binding README: merged the "Installation" and "Documentation" sections, replaced GitHub-only alert syntax with plain blockquotes, removed the redundant "API Reference" and "Changelog" sections (each now has its own docs-site page), shortened the "GPU Backend" blurb, and added a "Read more" link to the Concepts page. The top-level repository README is unchanged, since it's only ever viewed on GitHub.
* Renamed the batch adapter's "When to Use" heading to "When to Use Batch Adapter" across every binding/crate's API docs.
* Vendored the [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) theme (v2.4.2) for a modern, sidebar-only cpp Doxygen site with automatic dark mode.
* Added `dev/update_changelogs.py`, which regenerates a per-binding/crate `NEWS.md`/`news.md` from the root `CHANGELOG.md`. Wired into every docs site's navigation, the Rust crates' rustdoc module tree, and every `Makefile` `dev` target.
* Replaced `kernels.md`'s "Choosing a Kernel" mermaid flowchart (every binding/crate) with an equivalent decision table, since Doxygen and rustdoc don't render mermaid and the deeply-nested diamond chain was hard to read even where it did render.
* Replaced `adapter-choice.md`/`adapters.md`'s "Overview" flowchart (mermaid in most bindings/crates, ASCII art in the C++ docs) with an equivalent decision table, unifying on a single rendering-agnostic format across every binding/crate.
* Moved the duplicated "GPU Acceleration" section (installation, usage, supported features, feature comparison) out of `api.md` in the C++, Node.js, and Python bindings and the `fastLowess` crate into each one's dedicated `gpu-backend.md` guide, which also gained the "Hardware Requirements" and "Performance Considerations" subsections previously only in `api.md`; `api.md` now links to `gpu-backend.md` with a short blurb instead. Removed the same section from the `lowess` crate's `api.md` entirely, since that crate has no `gpu` Cargo feature or GPU backend to document.
* Consolidated `parameters.md`/the auto-generated `@autodocs` parameter reference across every binding and crate (C++, Julia, Node.js, Python, WASM, `fastLowess`, `lowess`): merged its unique content (fraction/iterations choice guidance, `delta`'s per-adapter default, and an inline `zero_weight_fallback` behavior table) into each `api.md`'s builder/options tables (Julia: into the `Lowess`/`StreamingLowess`/`OnlineLowess` docstrings), and removed `parameters.md` itself along with its docs-site navigation entries, `doc::parameters` rustdoc module, and cross-references (now pointing at `api.md`) — the parameter tables, kernel/robustness/boundary/scaling/merge-strategy option lists, and interval/custom-weights code examples it duplicated already live on their own dedicated pages.

## Fixed

* Fixed `docs.yml` triggering GitHub's "pages build and deployment" once per docs job; per-language jobs now upload artifacts, and a single final `deploy` job pushes to `gh-pages` once per run.
* Fixed `docs.yml`'s reliance on GitHub's legacy branch-based Pages deployment, which auto-triggers an unpinned, GitHub-managed "pages build and deployment" job on every `gh-pages` push (surfacing deprecation warnings, e.g. for Node.js 20, that aren't fixable from this repo). The former `deploy` job is now `build`, which still pushes the merged `_site` to `gh-pages` as a cache for future incremental runs, but publishing now goes through `actions/upload-pages-artifact` and a new `deploy` job using the official `actions/deploy-pages`, which this repo pins directly. Requires the repository's Pages source to be switched to "GitHub Actions" in settings.
* Fixed every benchmark category in `benchmarks/rfastlowess.R` failing with `attempt to apply non-function`: it called the R6-style `model$fit(x, y)`, but `fit` is an S3 generic (`fit(model, x, y)`), not a field on the `Lowess` object. Also fixed `benchmarks/stats_lowess.R` resolving its `output/` directory relative to the current working directory instead of the script's own location (unlike `rfastlowess.R`, which already did this correctly), so results could land outside `benchmarks/output/` depending on how the script was invoked.
* Fixed the "Handling Outliers" quickstart example (every binding and the `lowess`/`fastLowess` crates) printing nothing: with only 6 points and `fraction = 0.5`, the local window is small enough that tricube weighting drives the farthest neighbor's weight to ~0, leaving just 2 effectively-weighted points, which a degree-1 fit reproduces exactly (zero residual, no downweighting) — confirmed directly against the `lowess`/`loess` core, not binding-specific. Bumped to `fraction = 0.7`, which correctly downweights the injected outlier.
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
