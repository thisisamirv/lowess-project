# Contributing to fastlowess

We welcome contributions via bug reports, feature requests, documentation improvements, and code changes.

## Issues

Before opening a new issue, please search existing ones.

- **Bugs**: Include a minimal reproducible example, environment details (OS, Python version), and expected vs actual behavior.
- **Features**: Describe the use case and provide examples of the proposed API or behavior.

## Development Setup

The project uses a `Makefile` to standardize development tasks and `maturin` to build the Python bindings.

```bash
# Clone and branch
git clone https://github.com/thisisamirv/fastLowess-py.git
cd fastLowess-py
git checkout -b feature/your-feature

# Common commands
make develop    # Build and install in development mode
make check      # Run ALL checks: fmt, lint, build, test, docs, examples, install
make test       # Run Python and Rust test suites
make fmt        # Format code (Rust + Python)
make clippy     # Run Rust linter and Python linter
make doc        # Build Rust docs and Python (Sphinx) docs
make examples   # Run all example scripts
```

### Prerequisites

- **Rust**: Latest stable (1.85.0+)
- **Python**: 3.9+ with `pip`
- **maturin**: `pip install maturin`
- **ruff**: `pip install ruff` (for Python linting and formatting)
- **Sphinx**: `pip install sphinx` (optional, for documentation)

## Pull Requests

1. **Focus**: Keep PRs small and focused on a single change.
2. **Tests**: Add or update tests for any logic changes (usually in `tests/`).
3. **Linting**: Ensure `make check` passes. We follow [Conventional Commits](https://www.conventionalcommits.org/).
4. **Documentation**: Update docstrings in `fastlowess/__init__.py` and internal docs in `src/lib.rs`.

### Commitment Guidelines

- Use `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, or `chore` types.
- Scopes are optional but helpful (e.g., `bindings`, `docs`, `ci`).

## Project Structure

`fastLowess-py` bridges the high-performance Rust core with a Pythonic API:

1. **Rust Core (`src/lib.rs`)**: Contains the `PyO3` bindings and orchestrates calls to the `fastLowess` Rust crate.
2. **Python Package (`fastLowess/`)**: Contains the package initialization, high-level documentation, and metadata.
3. **Tests (`tests/`)**: High-level integration tests written in Python using `pytest`.

Most of the heavy lifting is done by the `fastLowess` Rust crate (<https://github.com/thisisamirv/fastLowess>).

## Testing

Tests are primarily written in Python to verify the user-facing API:

- `tests/test_fastlowess.py`: Functional tests using `pytest` and `numpy`.

Running tests:

```bash
make test-python  # Runs pytest
make test-rust    # Runs cargo test
```

Please do not add tests directly to the scripts in `examples/`. Add them to the `tests/` directory instead.

### Benchmarks and Validation

Correction and performance are validated against Python's `statsmodels`. While the core benchmarks reside in the `fastLowess` Rust crate, integration accuracy is verified here in `tests/`.

## License

By contributing, you agree that your work will be licensed under the project's existing license (AGPL-3.0).
