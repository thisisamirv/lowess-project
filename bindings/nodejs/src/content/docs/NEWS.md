---
title: News
---
<!-- markdownlint-disable MD024 MD025 -->
# fastlowess (Node.js) (development version)

## Changed

* Consolidated every crate/binding README: merged the "Installation" and "Documentation" sections, replaced GitHub-only alert syntax with plain blockquotes, removed the redundant "API Reference" and "Changelog" sections (each now has its own docs-site page), shortened the "GPU Backend" blurb, and added a "Read more" link to the Concepts page. The top-level repository README is unchanged, since it's only ever viewed on GitHub.
* Renamed the batch adapter's "When to Use" heading to "When to Use Batch Adapter" across every binding/crate's API docs.
* Vendored the [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) theme (v2.4.2) for a modern, sidebar-only cpp Doxygen site with automatic dark mode.
* Added `dev/update_changelogs.py`, which regenerates a per-binding/crate `NEWS.md`/`news.md` from the root `CHANGELOG.md`. Wired into every docs site's navigation, the Rust crates' rustdoc module tree, and every `Makefile` `dev` target.
* Moved the duplicated "GPU Acceleration" section (installation, usage, supported features, feature comparison) out of `api.md` in the C++, Node.js, and Python bindings and the `fastLowess` crate into each one's dedicated `gpu-backend.md` guide, which also gained the "Hardware Requirements" and "Performance Considerations" subsections previously only in `api.md`; `api.md` now links to `gpu-backend.md` with a short blurb instead. Removed the same section from the `lowess` crate's `api.md` entirely, since that crate has no `gpu` Cargo feature or GPU backend to document.
* Consolidated `parameters.md`/the auto-generated `@autodocs` parameter reference across every binding and crate (C++, Julia, Node.js, Python, WASM, `fastLowess`, `lowess`): merged its unique content (fraction/iterations choice guidance, `delta`'s per-adapter default, and an inline `zero_weight_fallback` behavior table) into each `api.md`'s builder/options tables (Julia: into the `Lowess`/`StreamingLowess`/`OnlineLowess` docstrings), and removed `parameters.md` itself along with its docs-site navigation entries, `doc::parameters` rustdoc module, and cross-references (now pointing at `api.md`) — the parameter tables, kernel/robustness/boundary/scaling/merge-strategy option lists, and interval/custom-weights code examples it duplicated already live on their own dedicated pages.
* Updated `typedoc-plugin-markdown` to v4.13.
* `make nodejs-dev` now runs `npm update` after `npm install`, so dependencies are kept current.

## Fixed

* Fixed `docs.yml` triggering GitHub's "pages build and deployment" once per docs job; per-language jobs now upload artifacts, and a single final `deploy` job pushes to `gh-pages` once per run.
* Fixed `docs.yml`'s reliance on GitHub's legacy branch-based Pages deployment, which auto-triggers an unpinned, GitHub-managed "pages build and deployment" job on every `gh-pages` push (surfacing deprecation warnings, e.g. for Node.js 20, that aren't fixable from this repo). The former `deploy` job is now `build`, which still pushes the merged `_site` to `gh-pages` as a cache for future incremental runs, but publishing now goes through `actions/upload-pages-artifact` and a new `deploy` job using the official `actions/deploy-pages`, which this repo pins directly. Requires the repository's Pages source to be switched to "GitHub Actions" in settings.
* Fixed the docs homepage never showing the README content ("Get Started" jumped straight to Installation): a new `dev/add-readme-to-docs.js` script embeds `README.md` below the hero (stripping its redundant `# LOWESS Project` H1, since the hero already shows the title), wired into `npm run docs` and `make nodejs-dev`.
* Fixed the docs build always emitting a `[@astrojs/sitemap] The Sitemap integration requires the site astro.config option` warning when the `SITE` environment variable isn't set (e.g. local builds); `astro.config.mjs` now falls back to the production GitHub Pages URL.

# fastlowess (Node.js) 3.1.0

## Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.
* Added a GitHub workflow for running validation scripts.
* Added `npm run lint` to the `Lint` step in `ci-nodejs.yml`, so JavaScript source and test files are linted via `oxlint` on every CI run.

## Changed

* Split the monolithic `.github/workflows/ci.yml` into seven per-language workflow files: `ci-rust.yml`, `ci-python.yml`, `ci-julia.yml`, `ci-nodejs.yml`, `ci-wasm.yml`, `ci-cpp.yml`, and `ci-r.yml`. Each file carries the relevant `ci` (multi-OS matrix), `asan`, and `gpu` jobs for its language.
* Each crate/binding sub-Makefile now runs `dev/verify_snippets.py --lang <lang>` for its own language as the final step of `make default`. The root `docs-test` target remains as a convenience to run all languages at once.
* Split `dev/verify_snippets.py` into a lean orchestrator and a `dev/runners/` package. Each language has its own module (`python.py`, `julia.py`, `nodejs.py`, `r.py`, `wasm.py`, `rust.py`, `cpp.py`) containing its `run_<lang>()` function and a `skip_reason()` predicate. Shared types (`Snippet`, `RunResult`) and utilities live in `runners/base.py`; the registry (`RUNNERS`, `SKIP_CHECKS`) is exported from `runners/__init__.py`.
* Moved CHANGELOG and CONTRIBUTING guides to project root.
* Updated README files to be binding/crate specific instead of one generic README for all bindings/crates.
* Moved Node.js documentation from ReadTheDocs to GitHub Pages, served by Starlight at <https://thisisamirv.github.io/lowess-project/nodejs/>. The ReadTheDocs site no longer includes Node.js-specific content. `dev/add-nodejs-outputs.js` runs as part of `make nodejs-dev`, executing each JavaScript code block in the docs and injecting its output back into the Markdown source.
* `make nodejs` (`default:`) now builds the native addon and links it globally via `npm link`. The full dev workflow moves to `make nodejs-dev`.
* Updated `oxlint` dependency to 1.80.

## Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare tool names resolved via `PATH`, matching the existing fix in `bindings/r/src/cargo-config.toml`.

# fastlowess (Node.js) 3.0.0

## Added

* Exposed a `backend` option (`"cpu"` default, `"gpu"`) on `Lowess`, gated behind an opt-in `gpu` Cargo feature not enabled in published npm binaries. Run `await fastlowess.installGpu()` to download a prebuilt GPU addon (requires restarting Node.js), or build locally with `napi build --features gpu`.

## Changed

* Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
* Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/lowess/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
* Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/lowess/` → `crates/lowess/tests/lowess/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages and `bindings/r/demo/`.
* Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`. This is a **breaking change**.
* Updated `@napi-rs/cli` to v3.8 and `oxlint` to v1.79.

# fastlowess (Node.js) 2.0.0

## Added

* Added `OnlineOutput` object to the Node.js binding. `OnlineLowess.add_point()` now returns `OnlineOutput | null` instead of `number | null`, exposing `smoothed`, `std_error`, `residual`, `robustness_weight`, and `iterations_used`.
* Added `return_se` and `cv_seed` fields to `SmoothOptions`.
* Added `customWeights` as an optional per-call argument to `fit(x, y, customWeights?)` and `fit_async(x, y, customWeights?)`. Accepts a `Float64Array` of non-negative per-observation weights. Includes pre-flight length-mismatch and non-negative validation. Batch only.
* Added JavaScript-layer option key validation: unknown keys in `SmoothOptions`, `StreamingOptions`, or `OnlineOptions` now throw a `TypeError` listing all valid keys, via wrapper classes around the native NAPI exports.

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
* Renamed all `Diagnostics`, `SmoothOptions`, `StreamingOptions`, and `OnlineOptions` interface fields to snake_case (`r_squared`, `effective_df`, `residual_sd`, `chunk_size`, `merge_strategy`, `window_capacity`, `min_points`, `update_mode`, and all smoothing option fields).
* Renamed binding methods to snake_case: `fit_async`, `process_chunk`, `add_points`.
* Renamed `LowessResultObj` getters to snake_case: `standard_errors`, `confidence_lower`, `confidence_upper`, `prediction_lower`, `prediction_upper`, `robustness_weights`, `cv_scores`, `fraction_used`, `iterations_used`.
* Updated `index.d.ts` to reflect all renamed fields and methods.
* Replaced `add_points(x: Float64Array, y: Float64Array): LowessResultObj` on `OnlineLowess` with `add_point(x: number, y: number): OnlineOutput | null`. The method now processes a single point and returns an `OnlineOutput` object, or `null` if not enough points have been accumulated yet. This is a **breaking change**.
* Changed default `OnlineOptions.window_capacity` from 100 to 1000 and `OnlineOptions.min_points` from 2 to 3, matching the defaults used by the loess binding.
* `OnlineLowess` now forwards all `SmoothOptions` fields to the underlying builder (previously only `fraction`, `iterations`, and `parallel` were forwarded; all other fields were hardcoded to `None`/`false`).
* Updated `napi-rs/cli` dependency to v3.7 and `oxlint` to v1.73.

# fastlowess (Node.js) 1.3.0

## Added

* Added prerequisites for different bindings and platforms to `CONTRIBUTING.md`
* Updated `docs/assets/diagrams/lowess_smoothing_concept.svg` to correctly illustrate LOWESS concepts (robustness iterations, bisquare re-weighting, outlier downweighting) instead of the generic LOESS algorithm it previously depicted.
* Modified `docs/requirements.txt` to update the versions of the documentation dependencies.
* Improved CI tests and coverage.
* Modified Makefile to be truely cross-platform.
* Added sanitizer check for all bindings and crates.
* Upgraded `oxlint` to 1.63.
* Upgraded `napi-rs/cli` to 3.6.

## Changed

* Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

## Fixed

* Fixed R ASAN tests failing to compile vignettes by passing `--no-build-vignettes` to `rcmdcheck`.
* Upgraded ASAN test environment to use modern `rocker/r-devel-san:latest` image and `RDscript` to resolve outdated `readelf` warnings.
* Fixed `Makefile` idempotency checks on Linux by providing a default `/tmp` fallback for the `TEMP` directory variable.
* Fixed accidental root `Cargo.toml` workspace isolation leaks by adding checked-in `pre-commit` and `pre-push` git hook guards that restore `Cargo.toml.bak` when present and fail loudly if required workspace members are still commented out.
* Added a repo-local `.cargo/config.toml` that sets `CC=clang-cl` for `x86_64-pc-windows-msvc`, fixing Criterion 0.8 benchmark builds on Windows when `cc-rs` would otherwise pick `clang.exe` and fail to link `alloca`.
* Fixed `make nodejs` on Windows when `/bin/bash` could not launch `npm` from `C:/Program Files/nodejs` by using `npm.cmd`/`npx.cmd` in the `Makefile`.
* Fixed deprecated JavaScript license-audit warnings by replacing the transient `npx license-checker` usage in the `Makefile` with a repo-local Node.js license summary script that still fails on GPL-family licenses.
* Fixed `.build()` errors incorrectly using `Status::GenericFailure`; they now return `Status::InvalidArg` since build failures originate from invalid configuration (accumulated parse errors), not from runtime execution.
* Linted the source code.

# fastlowess (Node.js) 1.2.0

## Added

* Added advanced License Compliance check.
* Added advanced dependency check.
* Added advanced outdated dependency check.
* Added advanced lock file check.
* Added advanced TypeScript check.

## Changed

* Switched from `eslint` to `oxlint` to remove vulnerabilities.

## Fixed

* Fixed project logo.
* Fixed vulnerabilities.

# fastlowess (Node.js) 1.0.0

## Added

* Added `mean` scaling method (Mean Absolute Deviation)
* Added JSDoc documentation to `lib.rs` for napi-rs generation
* Added asynchronous support for batch processing

## Changed

* Updated `eslint/js`, `eslint`, and `globals` packages to their latest versions.

# fastlowess (Node.js) 0.99.9

## Changed

* Bump rust version to 1.88 for better stability
* Change function-based builder pattern in the bindings to class-based builder pattern, allowing true streaming and online processing
* Improve API docs
* Package is now available on npm (fastlowess)

# fastlowess (Node.js) 0.99.8

## Added

* Initial implementation

# fastlowess (Node.js) 0.99.7

## Fixed

* Fix README file links
* Fix Makefile bug with R versioning

# fastlowess (Node.js) 0.99.6

## Fixed

* Fix README file formats and links

# fastlowess (Node.js) 0.99.5

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
