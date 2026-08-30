<!-- markdownlint-disable MD024 MD025 -->
# fastlowess (C++) (development version)

## Fixed

* Fixed the Doxygen site homepage showing `docs/concepts.md` instead of `README.md`. `Doxyfile` now includes `README.md` in `INPUT` and sets it as `USE_MDFILE_AS_MAINPAGE`.
* Fixed Doxygen rendering the "View the full documentation" blockquote as raw `<a>` tag text instead of a styled link, by dropping the markdown heading (`###`) nested inside the blockquote.
* Fixed Doxygen rendering headings that mixed a heading level with an inline code span (e.g. `` ### `fastlowess::Lowess` ``) as literal `<tt>...</tt>` tag text in `api.md`, `api-streaming.md`, and `api-online.md`; the backticks were dropped from those headings.
* Fixed Doxygen rendering MkDocs-only `!!! note/warning/tip "title"` admonitions as literal `!!! ...` text across the cpp docs; converted every occurrence to a plain `> **Title:** ...` blockquote.
* Fixed Doxygen rendering inline/display LaTeX math (`$...$`/`$$...$$`) as literal text across the cpp docs; converted every occurrence to Doxygen's `\f$...\f$`/`\f[...\f]` syntax.
* Fixed Doxygen leaking a stray `</blockquote>` tag when a `---` thematic break immediately followed a `>` blockquote; replaced those specific separators with an explicit `<hr>` tag across the cpp docs.

# fastlowess (C++) 3.1.0

## Added

* Added clang-tidy and cppcheck installation to Makefile.

## Changed

* Moved C++ documentation from ReadTheDocs to GitHub Pages, served by Doxygen at <https://thisisamirv.github.io/lowess-project/cpp/>. The ReadTheDocs site no longer includes C++-specific content.
* `make cpp` (`default:`) now only runs `cargo build`. The full dev workflow (formatting, linting, cbindgen idempotency, symbol export verification, cmake tests, valgrind, doc-snippet verification) moves to `make cpp-dev`.

## Fixed

* Fixed `make cpp` Windows CI failure (`cannot find -lgcc_eh`): the C++ binding's Makefile detected MinGW via `gcc -dumpmachine` and selected the GNU target, which then used the Rtools cross-compiler from the workspace `.cargo/config.toml`; that compiler delegated to `C:\mingw64\bin\ld.exe`, which lacks `lgcc_eh`. Fixed by always targeting `x86_64-pc-windows-msvc` on Windows, removing the MinGW detection branch entirely.
* Fixed clang-tidy warnings in `bindings/cpp/include/fastlowess.hpp`: replaced all `#if defined(_WIN32)` with `#ifdef _WIN32`, added `#include <cstdio>` for `stdin`/`fileno`/`_fileno`, replaced deprecated `std::getenv("USERPROFILE")` with `_dupenv_s` on Windows, and added `const` to `base` and `cmd` local variables.

# fastlowess (C++) 3.0.0

## Added

* Exposed a `backend` option (`"cpu"` default, `"gpu"`) on `Lowess`, gated behind an opt-in `gpu` Cargo feature not enabled by default. Call `fastlowess::gpu::install()` to download a prebuilt GPU library, or build locally with `cargo build --features gpu`.

## Changed

* Renamed `OnlineOutput`'s `smoothed()` and `std_error()` methods to `y()` and `standard_error()`. This is a **breaking change**.

# fastlowess (C++) 2.0.0

## Added

* Added `custom_weights` field to `LowessOptions` and a second overload of `Lowess::fit()` that accepts a `const std::vector<double>& custom_weights` argument. Values must be non-negative and length must match the input data. Batch only.

## Changed

* Renamed all public member functions to snake_case: `make_error()`, `has_value()`, `r_squared()`, `effective_df()`, `residual_sd()`, `x_value()`, `y_value()`, `x_vector()`, `y_vector()`, `standard_errors()`, `confidence_lower()`, `confidence_upper()`, `prediction_lower()`, `prediction_upper()`, `robustness_weights()`, `fraction_used()`, `iterations_used()`, `process_chunk()`, `add_points()`.
* Replaced `Expected<LowessResult> OnlineLowess::add_points(const std::vector<double>&, const std::vector<double>&)` with `Expected<std::optional<double>> OnlineLowess::add_point(double x, double y)`. The method now processes a single point and returns only that point's smoothed value, or `std::nullopt` if not enough points have been accumulated yet. The underlying C FFI symbol is renamed from `cpp_online_add_points` to `cpp_online_add_point`. This is a **breaking change**.

## Fixed

* Fixed remaining `yVector()` call in `testBasicSmoothSerial` that was missed during the snake_case rename (now `y_vector()`).

# fastlowess (C++) 1.3.0

## Added

* Added dedicated CMake packaging documentation in `bindings/cpp/CMAKE.md` for Windows installation, `find_package(fastlowess CONFIG REQUIRED)`, and build-tree package discovery.

## Changed

* Removed the legacy snake_case compatibility layer; the public C++ method API now uses camelBack, while variables and constants follow lower_case.
* Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

## Fixed

* Fixed `cbindgen` idempotency check failure by adding automatic installation of the `cbindgen` CLI tool if missing.
* Fixed explicit pointer checks, braces, named constants, and value-semantic result ownerships with compatibility wrappers.
* Fixed all clang-tidy findings.
* Fixed regenerated `fastlowess.h` clang-tidy regressions by normalizing the auto-generated header during the C++ bindings build, so FFI parameter naming and unused generated includes no longer come back after regeneration.
* Fixed `make cpp` on Windows by making C++ symbol-export verification, CMake test execution, DLL runtime resolution, and Unix-specific test steps platform-aware.
* Fixed MSVC `size_t` to `unsigned long` narrowing warnings in the C++ wrapper at the FFI boundary with explicit conversions.
* Fixed C++ CMake package integration by generating and installing `fastlowessConfig.cmake` and related package export files for downstream `find_package` use.

# fastlowess (C++) 1.2.0

## Fixed

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

* Library is now available on conda-forge (libfastlowess)

# fastlowess (C++) 0.99.8

## Added

* Initial implementation

For the full changelog, see:
<https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md>
