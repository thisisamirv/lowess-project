# Contributing to lowess-project

We welcome contributions via bug reports, feature requests, documentation improvements, and code changes.

## Issues

Before opening a new issue, please search existing ones.

- **Bugs**: Include a minimal reproducible example, environment details (OS, Rust/Python/R version), and expected vs actual behavior.
- **Features**: Describe the use case and provide examples of the proposed API or behavior.

## Ideas for Contribution

The Batch adapter already covers a comprehensive set of options (kernels, robustness, cross-validation, intervals, GPU execution, custom weights, etc.), but there's still room to grow. Contributions in any of these areas are especially welcome:

**Batch:**

- **Out-of-sample prediction**: `fit(x, y)` only returns smoothed values at the input `x` positions; there's no way to evaluate the fitted curve at new x-values not in the training set (like R's `predict.loess(model, newdata)`). Would need `fit()` to retain the local-fit structure for a later `predict(new_x)` call.
- **Expose local slope/derivative**: each point's local WLS fit already computes a slope internally ([regression.rs](crates/lowess/src/algorithms/regression.rs)), but only the fitted `y` is kept. A `return_derivative` option exposing that per-point slope would enable rate-of-change/turning-point analysis with minimal new computation.
- **Adaptive/automatic fraction selection**: CV-based bandwidth selection currently requires hand-picking a `cv_fractions` grid. A continuous search (e.g. golden-section over `(0, 1]` minimizing CV error or AICc) would remove the hardest tuning decision.
- **STL-style seasonal-trend decomposition**: Cleveland's STL is built from repeated LOWESS passes (trend + seasonal + remainder); none of the current adapters offer this, despite already having the necessary primitives.
- **Bootstrap-based intervals**: an alternative to the existing analytic hat-matrix SE-based confidence/prediction intervals, useful when the residual-normality assumption is questionable.

**Streaming:**

- **Concurrent chunk processing**: `parallel` currently only parallelizes the local regression fits *within* a single chunk; independent chunks still go through `process_chunk()` one at a time. A `process_chunks(&[...])` helper that processes chunks concurrently (carefully handling overlap-merge ordering) could meaningfully speed up large offline chunked jobs.
- **Checkpointable/resumable state**: `StreamingBuffer`/the builder state have no serialization support, so a long-running streaming session can't be saved and resumed across process restarts — useful for production ETL pipelines processing files far larger than memory over multiple runs.
- **Iterator-based convenience wrapper**: a `process_stream(impl Iterator<Item = (Vec<T>, Vec<T>)>)` helper that repeatedly calls `process_chunk`/`finalize` internally, reducing boilerplate for the common "read chunks from a file/socket" pattern.

**Online:**

- **Populate `standard_error`**: `OnlineOutput.standard_error` exists in the struct but is always `None` — the fast `Incremental` path never computes it, and the `Full` path builds its `LowessConfig` with `return_variance: None`, so it's never populated there either. Wiring up real standard errors (at least for `Full` mode) would give real-time uncertainty for dashboards.
- **Time/x-range-based window eviction**: `window_capacity` is a point-count cap; for irregularly-sampled real-time data (e.g. sensor gaps), a "keep points within the last N x-units" policy would be a useful alternative.
- **Configurable warm-up behavior**: `add_point()` returns `None` until `min_points` is reached; an option to return an early, lower-confidence estimate immediately (e.g. a running mean) instead of a gap could help dashboards that don't want to show blanks.

If you'd like to work on one of these, please open an issue first to discuss the approach and API shape.

## Development Setup

The project uses a root `Makefile` that delegates to per-component `Makefile`s in each crate and binding directory.

### Prerequisites

To develop across all platforms, you will need the following tools installed. You only need to install the prerequisites for the specific bindings you are working on.

**Core (Rust)**:

- **Rust**: 1.89.0+ (stable)
- **Cargo Tools**: `cargo fmt`, `cargo clippy`, `cargo llvm-cov` (for coverage)

**Python**:

- **Python**: 3.8+
- **Packages**: `pip install maturin ruff pytest numpy matplotlib` (or allow the Makefile to create a virtual environment)

**R**:

- **R**: 4.4.0+ (with `Rscript` in PATH). Note: On Windows, you must ensure the R `bin\x64` directory (e.g., `C:\Program Files\R\R-4.x.x\bin\x64`) is added to your system `Path` via `sysdm.cpl` so that test binaries can locate `R.dll`.
- **Rtools**: Required on Windows for C/C++ compilation. You must manually add it to your PATH to use `make` (e.g., in PowerShell: `$env:PATH = "C:\rtools45\usr\bin;C:\rtools45\x86_64-w64-mingw32.static.posix\bin;" + $env:PATH`)
- **Windows Rust Target**: `rustup target add x86_64-pc-windows-gnu` (R on Windows requires the GNU MinGW toolchain)
- **LaTeX Distribution**: Required for building PDF manual during `R CMD check --as-cran`. Install TinyTeX (`install.packages('tinytex'); tinytex::install_tinytex()`) or MiKTeX (Windows) or MacTeX (macOS) or TeX Live (Linux)
- **System Dependencies**:
  - **All platforms**: `pandoc`
  - **Linux/Ubuntu**: `libcurl4-openssl-dev`, `libssl-dev`, `libxml2-dev`, `libfontconfig1-dev`, `libharfbuzz-dev`, `libfribidi-dev`, `libfreetype6-dev`, `libpng-dev`, `libtiff5-dev`, `libjpeg-dev`, `libprotobuf-dev`, `protobuf-compiler`, `libuv1-dev`, `libgit2-dev`, `libssh2-1-dev`, `libmagick++-dev`
  - **macOS**: System libraries are typically available; install Xcode Command Line Tools if needed
  - **Windows**: System libraries are typically bundled with Rtools
- **air**: R formatter — auto-installed by `make r-dev` if not found (via the official installer script)
- *The Makefile automatically installs the following R packages into `bindings/r/.r-lib/`:*
  - *CRAN (required)*: `styler`, `testthat`, `rmarkdown`, `knitr`, `lintr`, `roxygen2`, `pkgdown`, `remotes`
  - *CRAN (optional)*: `covr`, `prettycode`, `toml`, `V8`, `visNetwork`
  - *ropensci (optional)*: `srr`, `pkgcheck`, `pkgstats`

**Julia**:

- **Julia**: 1.11+ (with `julia` in PATH)
- *Note: The Makefile automatically handles Julia package dependencies like JuliaFormatter, Aqua, and JET.*

**Node.js**:

- **Node.js & npm**: v22+ recommended (with `npx` in PATH)

**WebAssembly**:

- **wasm-pack**: Install via `cargo install wasm-pack` or the official installer
- **Node.js & npm**: Required for testing WASM output

**C++**:

- **Compiler**: `g++` or `clang++` with C++17 support
- **Tools**: `cmake`, `make`, `clang-tidy`, `cppcheck`, `valgrind`
- **cbindgen**: Install via `cargo install cbindgen` (for header generation)

**Go**:

- **Go**: 1.23+ (with `go` in PATH)
- **cgo**: `CGO_ENABLED=1` and a C compiler (GCC/Clang; on Windows, a MinGW-w64 toolchain, since Go's `cgo` invokes `gcc` rather than MSVC's `cl.exe`)
- **golangci-lint**: auto-installed by `make go-dev` if not found (via the official install script)

**Java**:

- **JDK**: 25+ (with `java`/`javac` in PATH, and `JAVA_HOME` pointing at it — some JDK installers add older JDKs earlier on `PATH`, which can silently shadow a newer `JAVA_HOME`)
- **Maven**: 3.9+ (with `mvn` in PATH)
- **Checkstyle**: auto-installed by `make java-dev` if not found (downloads the standalone jar)

### Clone and Branch

```bash
git clone https://github.com/thisisamirv/lowess-project.git
cd lowess-project
git checkout -b feature/your-feature
```

## Make Targets

The project is organized as a Cargo workspace with separate targets for each component:

### Rust Crates

```bash
# lowess crate (core algorithms)
make lowess          # Build only (cargo build)
make lowess-dev      # Format, lint, build, test, examples, doc snippets
make lowess-coverage # Run coverage
make lowess-clean    # Clean build artifacts

# fastLowess crate (high-level API with adapters)
make fastLowess          # Build only (cargo build)
make fastLowess-dev      # Format, lint, build, test, doc snippets
make fastLowess-coverage # Run coverage
make fastLowess-clean    # Clean build artifacts
```

### Python Bindings

```bash
make python          # Install to user environment (pip install --user)
make python-dev      # Format, lint, build, test, doc snippets (uses .venv)
make python-coverage # Run coverage
make python-clean    # Clean build artifacts
```

### R Bindings

```bash
make r          # Install the package (R CMD INSTALL)
make r-dev      # Vendor, format, lint, check (--as-cran), test, doc snippets
make r-coverage # Run coverage
make r-clean    # Clean build artifacts
```

### Julia Bindings

```bash
make julia       # Build Rust shared library and register package locally
make julia-dev   # Format, lint, build, test, export verification, ABI check, doc snippets
make julia-clean # Clean build artifacts
```

### Node.js Bindings

```bash
make nodejs       # Build and link (npm install + build + npm link)
make nodejs-dev   # Format, lint, audit, build, test, doc snippets
make nodejs-clean # Clean build artifacts
```

### WebAssembly Bindings

```bash
make wasm       # Build WASM targets and link (wasm-pack + npm link)
make wasm-dev   # Format, lint, build, test, doc snippets
make wasm-clean # Clean build artifacts
```

### C++ Bindings

```bash
make cpp       # Build only (cargo build --profile release-c)
make cpp-dev   # Format, lint, cbindgen, cmake tests, valgrind, doc snippets
make cpp-clean # Clean build artifacts
```

### Go Bindings

```bash
make go       # Build Rust FFI crate + `go build ./...`
make go-dev   # Format, lint (golangci-lint), build, export verification, test, doc snippets
make go-clean # Clean build artifacts
```

### Java Bindings

```bash
make java       # Build Rust JNI crate + `mvn package`
make java-dev   # Lint (Checkstyle), build, export verification, test, doc snippets
make java-clean # Clean build artifacts
```

### Full Workspace

```bash
make all          # Build all components
make all-dev      # Full quality-check workflow for all components
make all-coverage # Run coverage for lowess, fastLowess, python, and r
make all-clean    # Clean all build artifacts
make docs-test    # Run doc snippet tests across all languages
```

## Workspace Structure

This monorepo is a Cargo workspace. A `[workspace.package]` block in the root `Cargo.toml` defines shared values, but each crate currently declares all fields explicitly rather than using `field.workspace = true` inheritance:

```toml
# Root Cargo.toml
[workspace.package]
authors = ["Amir Valizadeh <thisisamirv@gmail.com>"]
edition = "2024"
license = "MIT OR Apache-2.0"
rust-version = "1.89"
repository = "https://github.com/thisisamirv/lowess-project"
# ... and more shared metadata
```

Each crate defines its own version and all metadata independently:

```toml
# Individual crate Cargo.toml
[package]
name = "lowess"
version = "4.0.0"
authors = ["Amir Valizadeh <thisisamirv@gmail.com>"]
edition = "2024"
rust-version = "1.89"
license = "MIT OR Apache-2.0"
# ...
```

**Notes:**

- Each crate carries its own version for independent publishing
- Shared values (edition, license, MSRV, repository) are kept consistent manually across crates

## Project Structure

```text
lowess-project/
├── crates/
│   ├── lowess/           # Core LOWESS algorithms (no_std compatible)
│   └── fastLowess/       # High-level API with adapters (Rayon + Ndarray)
├── bindings/
│   ├── python/           # PyO3 bindings (fastlowess package)
│   ├── r/                # extendr bindings (rfastlowess package)
│   ├── julia/            # C-API + Julia wrapper (FastLOWESS.jl)
│   ├── nodejs/           # NAPI-RS bindings
│   ├── wasm/             # wasm-bindgen bindings
│   ├── cpp/              # C++17 wrapper
│   ├── go/               # cgo bindings
│   └── java/             # JNI bindings
├── validation/           # R vs lowess parity validation
├── benchmarks/           # Performance benchmarks (Criterion)
└── Makefile              # Build automation
```

Documentation lives alongside each crate/binding rather than in a shared top-level directory: `crates/*/src/doc.rs` (rustdoc, via `#[cfg(doc)]` modules) and `bindings/*/docs`/`docs-site` (Sphinx, Documenter.jl, Doxygen, Antora, Starlight, Hugo, or pkgdown, depending on the language).

## Pull Requests

1. **Focus**: Keep PRs small and focused on a single change.
2. **Tests**: Add or update tests for any logic changes.
3. **Linting**: Ensure `make <component>` passes for affected components.
4. **Documentation**: Update docstrings and docs as needed.

### Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- Use `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, or `chore` types.
- Scopes are optional but helpful (e.g., `lowess`, `fastLowess`, `python`, `r`, `julia`, `nodejs`, `wasm`, `cpp`, `go`, `java`, `docs`).

Examples:

```text
feat(python): add streaming adapter support
fix(lowess): correct boundary padding calculation
docs: update installation instructions
```

## Testing

Tests are organized by component:

```bash
# Rust tests
cargo test -p lowess --features dev
cargo test -p fastLowess --features dev

# Python tests
pytest bindings/python/tests/

# R tests (package must be installed first via make r)
R_LIBS_USER=bindings/r/.r-lib Rscript -e \
  "Sys.setenv(NOT_CRAN='true'); testthat::test_dir('bindings/r/tests/testthat', package = 'rfastlowess')"

# Julia tests
julia --project=bindings/julia/julia bindings/julia/tests/test_FastLOWESS.jl

# Node.js tests
cd bindings/nodejs && npm test

# WASM tests
wasm-pack test --node bindings/wasm

# C++ tests (Linux; requires prior: cargo build -p fastlowess-cpp --profile release-c)
mkdir -p bindings/cpp/tests/build
cd bindings/cpp/tests/build
cmake -DFASTLOWESS_LIB="$(pwd)/../../../target/release-c/libfastlowess_cpp.so" \
      -DFASTLOWESS_LIB_DIR="$(pwd)/../../../target/release-c" ..
make && ./test_fastlowess_suite

# Go tests (requires prior: cargo build -p fastlowess-go --profile release-c)
cd bindings/go/fastlowess && go test ./... -v

# Java tests (requires prior: cargo build -p fastlowess-java)
cd bindings/java && mvn test
```

## License

By contributing, you agree that your work will be licensed under the project's dual license (MIT OR Apache-2.0).
