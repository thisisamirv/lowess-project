<!-- markdownlint-disable MD024 MD025 -->
# fastlowess (C++) (development version)

## Changed

* Consolidated every crate/binding README: merged the "Installation" and "Documentation" sections, replaced GitHub-only alert syntax with plain blockquotes, removed the redundant "API Reference" and "Changelog" sections (each now has its own docs-site page), shortened the "GPU Backend" blurb, and added a "Read more" link to the Concepts page. The top-level repository README is unchanged, since it's only ever viewed on GitHub.
* Renamed the batch adapter's "When to Use" heading to "When to Use Batch Adapter" across every binding/crate's API docs.
* Vendored the [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) theme (v2.4.2) for a modern, sidebar-only cpp Doxygen site with automatic dark mode.
* Added `dev/update_changelogs.py`, which regenerates a per-binding/crate `NEWS.md`/`news.md` from the root `CHANGELOG.md`. Wired into every docs site's navigation, the Rust crates' rustdoc module tree, and every `Makefile` `dev` target.
* Moved the duplicated "GPU Acceleration" section (installation, usage, supported features, feature comparison) out of `api.md` in the C++, Node.js, and Python bindings and the `fastLowess` crate into each one's dedicated `gpu-backend.md` guide, which also gained the "Hardware Requirements" and "Performance Considerations" subsections previously only in `api.md`; `api.md` now links to `gpu-backend.md` with a short blurb instead. Removed the same section from the `lowess` crate's `api.md` entirely, since that crate has no `gpu` Cargo feature or GPU backend to document.
* Consolidated `parameters.md`/the auto-generated `@autodocs` parameter reference across every binding and crate (C++, Julia, Node.js, Python, WASM, `fastLowess`, `lowess`): merged its unique content (fraction/iterations choice guidance, `delta`'s per-adapter default, and an inline `zero_weight_fallback` behavior table) into each `api.md`'s builder/options tables (Julia: into the `Lowess`/`StreamingLowess`/`OnlineLowess` docstrings), and removed `parameters.md` itself along with its docs-site navigation entries, `doc::parameters` rustdoc module, and cross-references (now pointing at `api.md`) — the parameter tables, kernel/robustness/boundary/scaling/merge-strategy option lists, and interval/custom-weights code examples it duplicated already live on their own dedicated pages.

## Fixed

* Fixed `docs.yml` triggering GitHub's "pages build and deployment" once per docs job; per-language jobs now upload artifacts, and a single final `deploy` job pushes to `gh-pages` once per run.
* Fixed several Doxygen rendering bugs: the homepage showed `concepts.md` instead of `README.md`; blockquotes, heading+codespan combinations, MkDocs-only admonitions, inline/display math, and `---` after a blockquote all rendered as literal or broken text. `README.md` is now the Doxygen main page, and the affected docs use Doxygen-native syntax (`\f$...\f$`/`\f[...\f]` math, blockquote admonitions, explicit `<hr>`).
* Fixed `ci-cpp.yml`'s macOS job warning that the pre-installed `aws/tap` Homebrew tap is untrusted; `brew untap aws/tap` now runs before `brew install llvm cppcheck`, since that tap isn't needed for this build.
* Fixed `ci-cpp.yml`'s Windows job installing `cppcheck` via Chocolatey, whose package is missing its `cfg/std.cfg` library files, causing `make cpp-dev`'s static analysis pass to be silently skipped; it now installs `cppcheck` via `winget` instead (matching the already-working `install-tools` target), with its install directory added to `$GITHUB_PATH`.

# fastlowess (C++) 3.1.0

## Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.
* Added a GitHub workflow for running validation scripts.
* Added clang-tidy and cppcheck installation to Makefile.

## Changed

* Split the monolithic `.github/workflows/ci.yml` into seven per-language workflow files: `ci-rust.yml`, `ci-python.yml`, `ci-julia.yml`, `ci-nodejs.yml`, `ci-wasm.yml`, `ci-cpp.yml`, and `ci-r.yml`. Each file carries the relevant `ci` (multi-OS matrix), `asan`, and `gpu` jobs for its language.
* Each crate/binding sub-Makefile now runs `dev/verify_snippets.py --lang <lang>` for its own language as the final step of `make default`. The root `docs-test` target remains as a convenience to run all languages at once.
* Split `dev/verify_snippets.py` into a lean orchestrator and a `dev/runners/` package. Each language has its own module (`python.py`, `julia.py`, `nodejs.py`, `r.py`, `wasm.py`, `rust.py`, `cpp.py`) containing its `run_<lang>()` function and a `skip_reason()` predicate. Shared types (`Snippet`, `RunResult`) and utilities live in `runners/base.py`; the registry (`RUNNERS`, `SKIP_CHECKS`) is exported from `runners/__init__.py`.
* Moved CHANGELOG and CONTRIBUTING guides to project root.
* Updated README files to be binding/crate specific instead of one generic README for all bindings/crates.
* Moved C++ documentation from ReadTheDocs to GitHub Pages, served by Doxygen at <https://thisisamirv.github.io/lowess-project/cpp/>. The ReadTheDocs site no longer includes C++-specific content.
* `make cpp` (`default:`) now only runs `cargo build`. The full dev workflow (formatting, linting, cbindgen idempotency, symbol export verification, cmake tests, valgrind, doc-snippet verification) moves to `make cpp-dev`.

## Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare tool names resolved via `PATH`, matching the existing fix in `bindings/r/src/cargo-config.toml`.
* Fixed `make cpp` Windows CI failure (`cannot find -lgcc_eh`): the C++ binding's Makefile detected MinGW via `gcc -dumpmachine` and selected the GNU target, which then used the Rtools cross-compiler from the workspace `.cargo/config.toml`; that compiler delegated to `C:\mingw64\bin\ld.exe`, which lacks `lgcc_eh`. Fixed by always targeting `x86_64-pc-windows-msvc` on Windows, removing the MinGW detection branch entirely.
* Fixed clang-tidy warnings in `bindings/cpp/include/fastlowess.hpp`: replaced all `#if defined(_WIN32)` with `#ifdef _WIN32`, added `#include <cstdio>` for `stdin`/`fileno`/`_fileno`, replaced deprecated `std::getenv("USERPROFILE")` with `_dupenv_s` on Windows, and added `const` to `base` and `cmd` local variables.

# fastlowess (C++) 3.0.0

## Added

* Exposed a `backend` option (`"cpu"` default, `"gpu"`) on `Lowess`, gated behind an opt-in `gpu` Cargo feature not enabled by default. Call `fastlowess::gpu::install()` to download a prebuilt GPU library, or build locally with `cargo build --features gpu`.

## Changed

* Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
* Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/lowess/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
* Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/lowess/` → `crates/lowess/tests/lowess/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages and `bindings/r/demo/`.
* Renamed `OnlineOutput`'s `smoothed()` and `std_error()` methods to `y()` and `standard_error()`. This is a **breaking change**.

# fastlowess (C++) 2.0.0

## Added

* Added `custom_weights` field to `LowessOptions` and a second overload of `Lowess::fit()` that accepts a `const std::vector<double>& custom_weights` argument. Values must be non-negative and length must match the input data. Batch only.

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
* Renamed all public member functions to snake_case: `make_error()`, `has_value()`, `r_squared()`, `effective_df()`, `residual_sd()`, `x_value()`, `y_value()`, `x_vector()`, `y_vector()`, `standard_errors()`, `confidence_lower()`, `confidence_upper()`, `prediction_lower()`, `prediction_upper()`, `robustness_weights()`, `fraction_used()`, `iterations_used()`, `process_chunk()`, `add_points()`.
* Replaced `Expected<LowessResult> OnlineLowess::add_points(const std::vector<double>&, const std::vector<double>&)` with `Expected<std::optional<double>> OnlineLowess::add_point(double x, double y)`. The method now processes a single point and returns only that point's smoothed value, or `std::nullopt` if not enough points have been accumulated yet. The underlying C FFI symbol is renamed from `cpp_online_add_points` to `cpp_online_add_point`. This is a **breaking change**.

## Fixed

* Fixed remaining `yVector()` call in `testBasicSmoothSerial` that was missed during the snake_case rename (now `y_vector()`).

# fastlowess (C++) 1.3.0

## Added

* Added prerequisites for different bindings and platforms to `CONTRIBUTING.md`
* Updated `docs/assets/diagrams/lowess_smoothing_concept.svg` to correctly illustrate LOWESS concepts (robustness iterations, bisquare re-weighting, outlier downweighting) instead of the generic LOESS algorithm it previously depicted.
* Modified `docs/requirements.txt` to update the versions of the documentation dependencies.
* Improved CI tests and coverage.
* Modified Makefile to be truely cross-platform.
* Added sanitizer check for all bindings and crates.
* Added dedicated CMake packaging documentation in `bindings/cpp/CMAKE.md` for Windows installation, `find_package(fastlowess CONFIG REQUIRED)`, and build-tree package discovery.

## Changed

* Removed the legacy snake_case compatibility layer; the public C++ method API now uses camelBack, while variables and constants follow lower_case.
* Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

## Fixed

* Fixed R ASAN tests failing to compile vignettes by passing `--no-build-vignettes` to `rcmdcheck`.
* Upgraded ASAN test environment to use modern `rocker/r-devel-san:latest` image and `RDscript` to resolve outdated `readelf` warnings.
* Fixed `Makefile` idempotency checks on Linux by providing a default `/tmp` fallback for the `TEMP` directory variable.
* Fixed accidental root `Cargo.toml` workspace isolation leaks by adding checked-in `pre-commit` and `pre-push` git hook guards that restore `Cargo.toml.bak` when present and fail loudly if required workspace members are still commented out.
* Added a repo-local `.cargo/config.toml` that sets `CC=clang-cl` for `x86_64-pc-windows-msvc`, fixing Criterion 0.8 benchmark builds on Windows when `cc-rs` would otherwise pick `clang.exe` and fail to link `alloca`.
* Fixed `cbindgen` idempotency check failure by adding automatic installation of the `cbindgen` CLI tool if missing.
* Fixed explicit pointer checks, braces, named constants, and value-semantic result ownerships with compatibility wrappers.
* Fixed all clang-tidy findings.
* Fixed regenerated `fastlowess.h` clang-tidy regressions by normalizing the auto-generated header during the C++ bindings build, so FFI parameter naming and unused generated includes no longer come back after regeneration.
* Fixed `make cpp` on Windows by making C++ symbol-export verification, CMake test execution, DLL runtime resolution, and Unix-specific test steps platform-aware.
* Fixed MSVC `size_t` to `unsigned long` narrowing warnings in the C++ wrapper at the FFI boundary with explicit conversions.
* Fixed C++ CMake package integration by generating and installing `fastlowessConfig.cmake` and related package export files for downstream `find_package` use.

# fastlowess (C++) 1.2.0

## Fixed

* Fixed project logo.
* Fixed memory leak in `OnlineLowess`.

# fastlowess (C++) 1.0.0

## Added

* Added `mean` scaling method (Mean Absolute Deviation)

## Changed

* Replaced exception-based error handling with a type-safe `Expected<T>` result type for all core methods (`fit`, `process_chunk`, `finalize`, `add_points`).
* Refactored the internal FFI layer to use the idiomatic Rust `From` trait for converting result types.
* Updated all C++ examples and tests to use the new `Expected` pattern, aligning the library with modern C++ practices.

# fastlowess (C++) 0.99.9

## Changed

* Bump rust version to 1.88 for better stability
* Change function-based builder pattern in the bindings to class-based builder pattern, allowing true streaming and online processing
* Improve API docs
* Library is now available on conda-forge (libfastlowess)

# fastlowess (C++) 0.99.8

## Added

* Initial implementation

# fastlowess (C++) 0.99.7

## Fixed

* Fix README file links
* Fix Makefile bug with R versioning

# fastlowess (C++) 0.99.6

## Fixed

* Fix README file formats and links

# fastlowess (C++) 0.99.5

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
