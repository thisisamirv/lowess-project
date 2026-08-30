<!-- markdownlint-disable MD024 MD025 -->
# rfastlowess (development version)

## Changed

* Removed the `rfastlowess-package` topic from the pkgdown reference index: its "Main Classes" summary duplicated the `Lowess`/`StreamingLowess`/`OnlineLowess` entries listed right next to it, and roxygen2's auto-generated `rfastlowess`/`rfastlowess-package` aliases showed up as a confusing double entry. The topic is now tagged `@keywords internal` so `?rfastlowess`/`?rfastlowess-package` still work without appearing in the index.
* Unexported the internal `Nullable()` helper (dropped `@export`, its `NAMESPACE` entry, and its `man`/pkgdown Reference page); it was never meant to be called by users. Tests now access it via `getFromNamespace()`, matching the existing `coerce_nullable` convention.
* Added `dev/update_r_news.py` (since generalized into `dev/update_changelogs.py`, see **docs** above), run from `make r-dev`'s Documentation step, which extracts the `**R:**` entries from the root `CHANGELOG.md` and regenerates `bindings/r/NEWS.md` in CRAN's conventional per-version format.

# rfastlowess 3.1.0

## Added

* Added `lenght` gaurds for extra arguments.

## Changed

* Simplified `bindings/r/Makefile`: replaced `Cargo.toml.orig` save/restore vendoring with `src/vendor-update.sh`; made `[workspace]` permanent in `src/Cargo.toml`; removed Bioconductor dependencies, redundant `cargo fmt --check`, `NAMESPACE` indentation post-processing, and `pkgdown::build_site` from the dev workflow.
* Moved R documentation from ReadTheDocs to GitHub Pages, served by pkgdown at <https://thisisamirv.github.io/loess-project/r/>. The ReadTheDocs site no longer includes R-specific content.
* Changed R version dependency to 4.4.0 due to issues with installing Bioconducter packages on R < 4.4.0.
* Replaced the multi-step `install.packages` / `BiocManager::install` package installation logic in `bindings/r/Makefile` with a single [`pak`](https://pak.r-lib.org/)-based block. `pak` handles RSPM binary vs source selection automatically (including Linux), skips already-installed packages, and installs CRAN, Bioconductor (`bioc::` prefix), and R-universe packages in one call.
* `make r` (`default:`) now runs `R CMD INSTALL $(R_DIR)` directly; R's `configure` script handles Rust compilation from the committed `vendor.tar.xz`. The full dev workflow moves to `make r-dev`.

## Fixed

* Fixed Windows arm64 (R-Universe) build: `ar x` without a member name correctly resolves long-name archive entries (>16 chars stored as `/<offset>`); named extraction silently fails for such entries. Used `objcopy --remove-section=.idata$4` on each extracted `.dll` stub to strip the invalid relocations that lld 19 rejects, then `ar r` to re-insert.
* Fixed `ld.lld` crashing or dropping symbols (`WakeByAddressSingle`, `WaitOnAddress`) on Windows arm64: `--whole-archive` pulls every crate's raw-dylib stub for a given DLL into the link, but different crates' stubs cover different, non-overlapping symbols of that DLL — `--allow-multiple-definition` works on x86_64 but crashes lld's arm64pe backend. Fixed by dropping `--whole-archive` on `gnullvm`; normal archive resolution applies and nothing is lost since `entrypoint.c` already references the extendr init symbol directly.
* Fixed CRAN Windows build (`error: linker not found`): `cargo-config.toml` hardcoded linker/ar as `c:/rtools45/...` absolute paths, which break when Rtools is installed on a different drive. Fixed by using bare tool names resolved via `PATH`.
* Fixed CRAN Windows build (`cannot find -lgcc_eh`): the Rtools gcc lib directory is not writable on CRAN's server, and config-file `rustflags` does not reach build-script linker invocations. `Makevars.win` creates an empty stub via `touch` in `$(TARGET_DIR)/libgcc_mock/` and passes `LIBRARY_PATH` inline on `cargo build`. The path is resolved to an absolute path via `$(pwd)` at shell execution time — a relative path silently fails because Cargo invokes GCC to link build scripts from its own temp directory, not from `src/`.
* Fixed `Lowess(fraction = 0.3, 4)` incorrectly succeeding: `reject_extra_positional_args()` counted unnamed arguments but did not check their position, so a single unnamed arg in any non-first slot passed validation. The check now rejects any unnamed argument that is not in position 1.

# rfastlowess 3.0.0

## Added

* Exposed a `backend` option (`"cpu"` default, `"gpu"`) on `Lowess`, gated behind an opt-in `gpu` Cargo feature not enabled in CRAN/Bioconductor releases. Run `install_gpu()` to download a prebuilt GPU library (requires restarting R), or build locally with `make -f bindings/r/Makefile WITH_GPU=1`.
* Introduced S3 generics `fit()`, `process_chunk()`, `finalize()`, and `add_point()`, replacing the previous list-closure API.
* `bindings/r/Makefile` now auto-installs [Air](https://posit-dev.github.io/air/) if missing, before running `air format`.
* Added a `reject_extra_positional_args()` helper to reject extra unnamed arguments.

## Fixed

* Fixed incorrect URLs in R binding docs.

## Changed

* Renamed the `smoothed` and `std_error` fields returned by `OnlineLowess`'s `add_point()` to `y` and `standard_error`. This is a **breaking change**.
* Replaced `dev/style_pkg.R` with [Air](https://posit-dev.github.io/air/) for formatting.
* Removed `dev/fix_rd_style.R`, `dev/prepare_cargo.py`, `dev/patch_vendor_crates.py`, `dev/clean_checksums.py`, and `dev/prepare_cran.sh` — their logic is now inlined directly in `bindings/r/Makefile`, so the R build no longer requires any Python scripts.
* Added `...` to `Lowess()`, `StreamingLowess()`, and `OnlineLowess()` to force named arguments for optional parameters.
* Added `Depends: R (>= 4.6)` to `DESCRIPTION` and a matching CI matrix entry.
* Expanded roxygen2 `@param` docs and added a `See Also` section linking to <https://lowess.readthedocs.io/>.
* Expanded `rfastlowess-intro.Rmd` vignettes.

# rfastlowess 2.0.0

## Added

* Added `custom_weights` parameter to `fit(Lowess, )`. Accepts a numeric vector of non-negative per-observation weights. Batch only.

## Changed

* Replaced `$add_points(x, y)` (vector inputs returning a list result) on `OnlineLowess` with `$add_point(x, y)` (scalar inputs returning `numeric` or `NULL`). The method now processes one point at a time and returns `NULL` until enough points have been accumulated. This is a **breaking change**.

# rfastlowess 1.3.0

## Added

* Upgraded `rextendr` scaffold to 0.5.0: bumped `Config/rextendr/version` in `DESCRIPTION` and updated `entrypoint.c` to register the extendr panic hook (`register_extendr_panic_hook()`), so Rust panics now surface as R errors instead of crashing the session.
* Added `dev/fix_rd_style.R` post-processing script to automatically normalize Rd file indentation (to 4 spaces) and wrap long lines (> 80 characters), ensuring compliance with CRAN/pkgcheck stylistic notes.
* Added `bindings/r/_pkgdown.yml` configuration and updated the `Makefile` to use `pkgdown::build_site()`, satisfying the `pkgcheck` requirement for a dedicated documentation website.
* Added automatic copying of shared tests from the project root into the R package within `dev/prepare_cran.sh`, ensuring `R CMD build` is self-contained.
* Added direct `extendr` wrapper coverage tests plus extra validation-path tests in the R package, lifting `covr::package_coverage()` to 100% and clearing the package-level `pkgcheck`/`goodpractice` coverage complaints.

## Changed

* Refactored the R binding validation helpers to reuse `validate_common_args()` and `coerce_nullable()` in production code, split `validate_params()` into smaller helper validators, and consolidated duplicated constructor parameter documentation with `@inheritParams` before regenerating the Rd files.
* Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.
* Abides by new rOpenSci standards.

## Fixed

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

# rfastlowess 1.2.0

## Added

* Added new tests.
* Added a reference to the `CONTRIBUTING.md` file.
* Added new examples for the `print` and `plot` methods.
* Added test coverage evaluation.
* Added missing srr tags.

## Changed

* Removed the `coerce_params` dead code.
* Spread srr tags to different files and removed extra tags from `srr-stats-standards.R`.

# rfastlowess 1.0.0

## Added

* Added `mean` scaling method (Mean Absolute Deviation)
* Implemented `print` and `plot` methods for `LowessResult` objects
* Added srr tags

## Changed

* Return results as `LowessResult` S3 objects instead of raw vectors

# rfastlowess 0.99.9

## Changed

* Package is now available on conda-forge (r-rfastlowess)

# rfastlowess 0.99.2

## Changed

* Prepared package for rOpenSci Software Peer Review
* Renamed main functions to avoid conflicts with base R:
* Updated documentation to reflect new API and rOpenSci guidelines

## Added

* Documentation website using `pkgdown` with automated deployment
* Comprehensive function documentation with examples and cross-references
* URL validation using `urlchecker` in CI workflow
* Rigorous parameter validation for all exported functions
* Expanded test suite achieving >96% coverage
* Codecov CI workflow and badge

## Fixed

* Documentation URLs
* Package startup messages
* `pkgcheck` workflow to run on host runner

# rfastlowess 0.99.1

## Changed

* Modified package for Bioconductor submission

# rfastlowess 0.99.0

## Added

* Support for new features in `fastLowess` v0.4.0, including `NoBoundary` boundary policy and `MAD`/`MAR` scaling methods

## Changed

* Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
* Updated documentation

# rfastlowess 0.4.0

## Added

* Support for new features in `fastLowess` v0.4.0

## Changed

* Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
* Updated documentation

# rfastlowess 0.3.0

## Added

* Option to install from R-universe without Rust

## Changed

* Updated to `fastLowess` v0.3.0 and `lowess` v0.6.0
* Updated cross-validation API

## Fixed

* Automated vendor checksum fixing for CI builds

# rfastlowess 0.2.0

## Added

* Support for new features in `fastLowess` v0.2.0

## Changed

* Improved documentation

# rfastlowess 0.1.0

## Added

* R binding for `fastLowess`

For the full changelog, see:
<https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md>
