<!-- markdownlint-disable MD024 MD025 -->
# FastLOWESS.jl (development version)

## Changed

* Consolidated every crate/binding README: merged the "Installation" and "Documentation" sections, replaced GitHub-only alert syntax with plain blockquotes, removed the redundant "API Reference" and "Changelog" sections (each now has its own docs-site page), shortened the "GPU Backend" blurb, and added a "Read more" link to the Concepts page. The top-level repository README is unchanged, since it's only ever viewed on GitHub.
* Renamed the batch adapter's "When to Use" heading to "When to Use Batch Adapter" across every binding/crate's API docs.
* Vendored the [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) theme (v2.4.2) for a modern, sidebar-only cpp Doxygen site with automatic dark mode.
* Added `dev/update_changelogs.py`, which regenerates a per-binding/crate `NEWS.md`/`news.md` from the root `CHANGELOG.md`. Wired into every docs site's navigation, the Rust crates' rustdoc module tree, and every `Makefile` `dev` target.
* Moved the duplicated "GPU Acceleration" section (installation, usage, supported features, feature comparison) out of `api.md` in the C++, Node.js, and Python bindings and the `fastLowess` crate into each one's dedicated `gpu-backend.md` guide, which also gained the "Hardware Requirements" and "Performance Considerations" subsections previously only in `api.md`; `api.md` now links to `gpu-backend.md` with a short blurb instead. Removed the same section from the `lowess` crate's `api.md` entirely, since that crate has no `gpu` Cargo feature or GPU backend to document.
* Consolidated `parameters.md`/the auto-generated `@autodocs` parameter reference across every binding and crate (C++, Julia, Node.js, Python, WASM, `fastLowess`, `lowess`): merged its unique content (fraction/iterations choice guidance, `delta`'s per-adapter default, and an inline `zero_weight_fallback` behavior table) into each `api.md`'s builder/options tables (Julia: into the `Lowess`/`StreamingLowess`/`OnlineLowess` docstrings), and removed `parameters.md` itself along with its docs-site navigation entries, `doc::parameters` rustdoc module, and cross-references (now pointing at `api.md`) — the parameter tables, kernel/robustness/boundary/scaling/merge-strategy option lists, and interval/custom-weights code examples it duplicated already live on their own dedicated pages.

## Fixed

* Fixed `docs.yml` triggering GitHub's "pages build and deployment" once per docs job; per-language jobs now upload artifacts, and a single final `deploy` job pushes to `gh-pages` once per run.
* Fixed the Documenter homepage: it was a stale, separately maintained `index.md` instead of the README, and the README's centered badge/logo HTML and markdownlint comment rendered as literal text. `make.jl` now regenerates `index.md` from `README.md` on every build.
* Fixed `release-julia-register.yml` extracting the matching version section from the root `CHANGELOG.md`, which includes every binding/crate's entries; it now extracts from the already Julia-filtered `bindings/julia/julia/docs/src/NEWS.md` instead, so the JuliaRegistrator release notes only cover Julia-relevant changes.
* Fixed `make julia-dev` failing with "empty intersection between `fastlowess_jll@X.Y.Z` and project compatibility ..." whenever a locally cached `Manifest.toml` still pinned an older `fastlowess_jll` version after a new one was published: `Pkg.resolve()` treats an already-pinned manifest entry as fixed and won't search the registry for an upgrade, even when the relaxed compat bound requires one. The `dev` target now runs `Pkg.update("fastlowess_jll")` before `Pkg.resolve()` to actively pick up the newly published version.

# FastLOWESS.jl 3.1.0

## Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.
* Added a GitHub workflow for running validation scripts.
* `release-julia-register.yml` now automatically extracts the matching changelog section and appends it as release notes in the JuliaRegistrator comment, enabling auto-merge on major version bumps.

## Changed

* Split the monolithic `.github/workflows/ci.yml` into seven per-language workflow files: `ci-rust.yml`, `ci-python.yml`, `ci-julia.yml`, `ci-nodejs.yml`, `ci-wasm.yml`, `ci-cpp.yml`, and `ci-r.yml`. Each file carries the relevant `ci` (multi-OS matrix), `asan`, and `gpu` jobs for its language.
* Each crate/binding sub-Makefile now runs `dev/verify_snippets.py --lang <lang>` for its own language as the final step of `make default`. The root `docs-test` target remains as a convenience to run all languages at once.
* Split `dev/verify_snippets.py` into a lean orchestrator and a `dev/runners/` package. Each language has its own module (`python.py`, `julia.py`, `nodejs.py`, `r.py`, `wasm.py`, `rust.py`, `cpp.py`) containing its `run_<lang>()` function and a `skip_reason()` predicate. Shared types (`Snippet`, `RunResult`) and utilities live in `runners/base.py`; the registry (`RUNNERS`, `SKIP_CHECKS`) is exported from `runners/__init__.py`.
* Moved CHANGELOG and CONTRIBUTING guides to project root.
* Updated README files to be binding/crate specific instead of one generic README for all bindings/crates.
* Moved Julia documentation from ReadTheDocs to GitHub Pages, served by Documenter.jl at <https://thisisamirv.github.io/lowess-project/julia/stable/>. The ReadTheDocs site no longer includes Julia-specific content. Code blocks use Documenter.jl `@example` sections, which execute and embed output automatically during the docs build.
* `make julia` (`default:`) now builds the Rust library and installs the Julia package via `Pkg.develop`. The full dev workflow moves to `make julia-dev`.

## Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare tool names resolved via `PATH`, matching the existing fix in `bindings/r/src/cargo-config.toml`.

# FastLOWESS.jl 3.0.0

## Added

* Exposed a `backend` option (`"cpu"` default, `"gpu"`) on `Lowess`, gated behind an opt-in `gpu` Cargo feature not enabled in published JLL artifacts. Run `install_gpu()` to download a prebuilt GPU library, or build locally with `cargo build --release --features gpu`.

## Fixed

* Fixed `LowessResult.iterations_used` returning the raw FFI sentinel `-1` instead of `nothing` when robustness iterations were not applicable.

## Changed

* Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
* Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/lowess/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
* Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/lowess/` → `crates/lowess/tests/lowess/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages and `bindings/r/demo/`.
* Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`. This is a **breaking change**.
* Removed `dev/format_julia.jl`; formatting is now inlined in `bindings/julia/Makefile`.

# FastLOWESS.jl 2.0.0

## Added

* Added `custom_weights` keyword argument to `fit(model, x, y; custom_weights)`. Accepts a `Vector{Float64}` of non-negative per-observation weights. Batch only.

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
* Replaced `add_points(online, x::Vector{Float64}, y::Vector{Float64}) :: LowessResult` with `add_point(online, x::Float64, y::Float64) :: Union{Float64, Nothing}`. The function now processes a single point and returns the smoothed value, or `nothing` if not enough points have been accumulated yet. The underlying C FFI symbol is renamed from `jl_online_lowess_add_points` to `jl_online_lowess_add_point`. This is a **breaking change**.

# FastLOWESS.jl 1.3.0

## Added

* Added prerequisites for different bindings and platforms to `CONTRIBUTING.md`
* Updated `docs/assets/diagrams/lowess_smoothing_concept.svg` to correctly illustrate LOWESS concepts (robustness iterations, bisquare re-weighting, outlier downweighting) instead of the generic LOESS algorithm it previously depicted.
* Modified `docs/requirements.txt` to update the versions of the documentation dependencies.
* Improved CI tests and coverage.
* Modified Makefile to be truely cross-platform.
* Added sanitizer check for all bindings and crates.

## Changed

* Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

## Fixed

* Fixed R ASAN tests failing to compile vignettes by passing `--no-build-vignettes` to `rcmdcheck`.
* Upgraded ASAN test environment to use modern `rocker/r-devel-san:latest` image and `RDscript` to resolve outdated `readelf` warnings.
* Fixed `Makefile` idempotency checks on Linux by providing a default `/tmp` fallback for the `TEMP` directory variable.
* Fixed accidental root `Cargo.toml` workspace isolation leaks by adding checked-in `pre-commit` and `pre-push` git hook guards that restore `Cargo.toml.bak` when present and fail loudly if required workspace members are still commented out.
* Added a repo-local `.cargo/config.toml` that sets `CC=clang-cl` for `x86_64-pc-windows-msvc`, fixing Criterion 0.8 benchmark builds on Windows when `cc-rs` would otherwise pick `clang.exe` and fail to link `alloca`.
* Fixed Windows local Julia runs by exporting an absolute `FASTLOWESS_LIB` path from the `Makefile` and moving DLL discovery in `FastLOWESS.jl` to runtime (`__init__()` plus runtime `ccall`), preventing stale precompiled library paths from being reused.
* Linted the source code.

# FastLOWESS.jl 1.2.0

## Fixed

* Fixed project logo.
* Linted examples.

# FastLOWESS.jl 1.0.0

## Added

* Added `mean` scaling method (Mean Absolute Deviation)

## Changed

* Wrapped all FFI functions in std::panic::catch_unwind. This ensures that if the Rust library panics (e.g., due to an internal assertion), it will be caught and reported as an error to Julia.

# FastLOWESS.jl 0.99.9

## Changed

* Bump rust version to 1.88 for better stability
* Change function-based builder pattern in the bindings to class-based builder pattern, allowing true streaming and online processing
* Improve API docs
* Package is now registered on JuliaRegistries

# FastLOWESS.jl 0.99.8

## Added

* Initial implementation

# FastLOWESS.jl 0.99.7

## Fixed

* Fix README file links
* Fix Makefile bug with R versioning

# FastLOWESS.jl 0.99.6

## Fixed

* Fix README file formats and links

# FastLOWESS.jl 0.99.5

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
