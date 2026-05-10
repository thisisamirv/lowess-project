# Contributing to lowess-project

We welcome contributions via bug reports, feature requests, documentation improvements, and code changes.

## Issues

Before opening a new issue, please search existing ones.

- **Bugs**: Include a minimal reproducible example, environment details (OS, Rust/Python/R version), and expected vs actual behavior.
- **Features**: Describe the use case and provide examples of the proposed API or behavior.

## Development Setup

The project uses a `Makefile` to standardize development tasks.

### Prerequisites

To develop across all platforms, you will need the following tools installed. You only need to install the prerequisites for the specific bindings you are working on.

**Core (Rust)**:

- **Rust**: 1.88.0+ (stable)
- **Cargo Tools**: `cargo fmt`, `cargo clippy`, `cargo llvm-cov` (for coverage)

**Python**:

- **Python**: 3.8+
- **Packages**: `pip install maturin ruff pytest numpy matplotlib` (or allow the Makefile to create a virtual environment)

**R**:

- **R**: 4.2+ (with `Rscript` in PATH). Note: On Windows, you must ensure the R `bin\x64` directory (e.g., `C:\Program Files\R\R-4.x.x\bin\x64`) is added to your system `Path` via `sysdm.cpl` so that test binaries can locate `R.dll`.
- **Rtools**: Required on Windows for C/C++ compilation. You must manually add it to your PATH to use `make` (e.g., in PowerShell: `$env:PATH = "C:\rtools45\usr\bin;C:\rtools45\x86_64-w64-mingw32.static.posix\bin;" + $env:PATH`)
- **Windows Rust Target**: `rustup target add x86_64-pc-windows-gnu` (R on Windows requires the GNU MinGW toolchain)
- **LaTeX Distribution**: Required for building PDF manual during `R CMD check --as-cran`. Install TinyTeX (`install.packages('tinytex'); tinytex::install_tinytex()`) or MiKTeX (Windows) or MacTeX (macOS) or TeX Live (Linux)
- **System Dependencies**:
  - **All platforms**: `pandoc`
  - **Linux/Ubuntu**: `libcurl4-openssl-dev`, `libssl-dev`, `libxml2-dev`, `libfontconfig1-dev`, `libharfbuzz-dev`, `libfribidi-dev`, `libfreetype6-dev`, `libpng-dev`, `libtiff5-dev`, `libjpeg-dev`, `libprotobuf-dev`, `protobuf-compiler`, `libuv1-dev`, `libgit2-dev`, `libssh2-1-dev`, `libmagick++-dev`
  - **macOS**: System libraries are typically available; install Xcode Command Line Tools if needed
  - **Windows**: System libraries are typically bundled with Rtools
- *Note: The Makefile automatically installs R-level development dependencies (BiocManager, styler, lintr, roxygen2, pkgdown, testthat, etc.)*

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

**Documentation**:

- **Python**: `python3` (the Makefile automatically creates a virtual environment and installs `mkdocs`)

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
make lowess          # Format, lint, build, test, examples
make lowess-coverage # Run coverage
make lowess-clean    # Clean build artifacts

# fastLowess crate (high-level API with adapters)
make fastLowess          # Format, lint, build, test, examples
make fastLowess-coverage # Run coverage
make fastLowess-clean    # Clean build artifacts
```

### Python Bindings

```bash
make python          # Format, lint, build, test, examples
make python-coverage # Run coverage
make python-clean    # Clean build artifacts
```

### R Bindings

```bash
make r          # Vendor, build, check, test
make r-clean    # Clean build artifacts
```

### Full Workspace

```bash
make all       # Run checks for all components (lowess, fastLowess, python, r)
make all-clean # Clean all build artifacts
make docs      # Build MkDocs documentation
make docs-serve # Serve documentation locally
```

## Workspace Structure

This monorepo uses **Cargo workspace inheritance** for centralized configuration:

```toml
# Root Cargo.toml
[workspace.package]
version = "0.99.3"
authors = ["Amir Valizadeh <thisisamirv@gmail.com>"]
edition = "2024"
license = "MIT OR Apache-2.0"
rust-version = "1.89"
readme = "README.md"
# ... and more

[workspace.dependencies]
lowess = { version = "0.99", path = "crates/lowess" }
fastLowess = { version = "0.99", path = "crates/fastLowess" }
# ... shared dependencies
```

All member crates inherit from workspace:

```toml
# Individual crate Cargo.toml
[package]
name = "lowess"
version = { workspace = true }
authors = { workspace = true }
readme = { workspace = true }
# ...
```

**Benefits:**

- Single source of truth for versions and metadata
- Update once → all crates bump together
- Consistent MSRV across all packages

## Project Structure

```text
lowess-project/
├── crates/
│   ├── lowess/           # Core LOWESS algorithms (no_std compatible)
│   └── fastLowess/       # High-level API with adapters (Rayon + Ndarray)
├── bindings/
│   ├── python/           # PyO3 bindings (fastlowess package)
│   └── r/                # extendr bindings (rfastlowess package)
├── tests/                # Tests
├── docs/                 # MkDocs documentation
└── Makefile              # Build automation
```

## Pull Requests

1. **Focus**: Keep PRs small and focused on a single change.
2. **Tests**: Add or update tests for any logic changes.
3. **Linting**: Ensure `make <component>` passes for affected components.
4. **Documentation**: Update docstrings and docs as needed.

### Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- Use `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, or `chore` types.
- Scopes are optional but helpful (e.g., `lowess`, `python`, `r`, `docs`).

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
cargo test -p lowess
cargo test -p fastLowess

# Python tests
pytest tests/python/

# R tests (via make r)
```

## License

By contributing, you agree that your work will be licensed under the project's dual license (MIT OR Apache-2.0).
