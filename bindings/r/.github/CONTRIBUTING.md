# Contributing to rfastlowess (R)

We welcome contributions via bug reports, feature requests, documentation improvements, and code changes.

## Issues

Before opening a new issue, please search existing ones.

- **Bugs**: Include a minimal reproducible example (reprex), environment details (`sessionInfo()` in R), and expected vs actual behavior.
- **Features**: Describe the use case and provide examples of the proposed R API or behavior.

## Development Setup

The project uses a `Makefile` to standardize development tasks and `extendr` to build the Rust bindings for R.

```bash
# Clone and branch
git clone https://github.com/thisisamirv/rfastlowess.git
cd rfastlowess
git checkout -b feature/your-feature

# Run all checks (the main command you need)
make check
```

### Common Makefile Commands

| Command                     | Description                             |
|-----------------------------|-----------------------------------------|
| `make check`                | Run all checks                          |
| `make clean`                | Clean all build artifacts               |
| `make install-dev-packages` | Install required development R packages |

### Prerequisites

- **Rust**: Latest stable (1.85.0+)
- **R**: 4.0+
- **R Packages**: `devtools`, `roxygen2`, `lintr`, `styler`, `prettycode`, `covr`, `codemetar`, `BiocCheck`, `urlchecker`, `pkgdown` (Install all via `make install-dev-packages`)

## Pull Requests

1. **Focus**: Keep PRs small and focused on a single change.
2. **Tests**: Add or update demonstrations in `demo/` or examples in `R/` files.
3. **Linting**: Ensure `make check` passes. We follow [Conventional Commits](https://www.conventionalcommits.org/).
4. **Documentation**: Update roxygen2 comments in `R/` files and internal docs in `src/lib.rs`.

### Commitment Guidelines

- Use `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, or `chore` types.
- Scopes are optional but helpful (e.g., `bindings`, `docs`, `ci`).

## Project Structure

`fastLowess-R` bridges the high-performance Rust core with an R-native API:

1. **Rust Core (`src/`)**: Contains the `extendr` bindings in `lib.rs` and vendored Rust dependencies.
2. **R Package (`R/`)**: Contains the R wrapper functions, documentation templates, and user-facing API.
3. **Demos (`demo/`)**: Functional demonstration scripts used for verification.

Most of the heavy lifting is done by the `fastLowess` Rust crate (<https://github.com/thisisamirv/fastLowess>).

### Dependency Management

Rust dependencies are vendored into `src/vendor/` to allow for CRAN-compliant, offline builds. To update these from crates.io, use:

```bash
make vendor-update
```

This command automates re-vendoring and cleans up checksums for CRAN portability.

## Testing

Verification is performed via Rust tests, demonstration scripts, and the standard `R CMD check`:

- `make test-rust`: Runs Rust unit tests.
- `demo/*.R`: Functional tests for batch, online, and streaming smoothing.
- `R CMD check`: Verified via `make check-CMD` or `devtools::check()`.

Please do not add tests directly to the package root. Add them to `demo/` or as `@examples` in the R documentation.

### Benchmarks and Validation

Integration accuracy is verified via the demo scripts. While performance benchmarks reside in the `fastLowess` Rust crate, we ensure the R bindings maintain consistency with the core engine.

## License

By contributing, you agree that your work will be licensed under the project's existing license.
