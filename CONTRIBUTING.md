# Contributing to lowess-project

We welcome contributions via bug reports, feature requests, documentation improvements, and code changes.

## Issues

Before opening a new issue, please search existing ones.

- **Bugs**: Include a minimal reproducible example, environment details (OS, Rust/Python/R version), and expected vs actual behavior.
- **Features**: Describe the use case and provide examples of the proposed API or behavior.

## Development Setup

The project uses a `Makefile` to standardize development tasks.

### Prerequisites

- **Rust**: 1.85.0+ (stable)
- **Python**: 3.8+ with `pip`
- **R**: 4.2+ (for R bindings)
- **maturin**: `pip install maturin` (for Python bindings)
- **ruff**: `pip install ruff` (for Python linting)

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
rust-version = "1.85.0"
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
