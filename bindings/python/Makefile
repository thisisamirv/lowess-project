# Run all local checks (formatting, linting, building, tests, docs)
check: fmt clippy build maturin install test doc examples
	@echo "All checks completed successfully!"

# Formatting
fmt: fmt-rust fmt-py fmt-fix fmt-fix-rust fmt-fix-py
	@echo "Formatting check complete!"

fmt-rust:
	@echo "Checking Rust code formatting..."
	@cargo fmt --all -- --check

fmt-py:
	@echo "Checking Python code formatting with ruff..."
	@ruff format --check fastlowess/ tests/ || true

fmt-fix: fmt-fix-rust fmt-fix-py
	@echo "Formatting complete!"

fmt-fix-rust:
	@echo "Formatting Rust code..."
	@cargo fmt --all

fmt-fix-py:
	@echo "Formatting Python code with ruff..."
	@ruff format fastlowess/ tests/

# Linter
clippy: clippy-default clippy-serial lint-py lint-py-fix

clippy-default:
	@echo "Running clippy (default / parallel)..."
	@cargo clippy --all-targets -- -D warnings

clippy-serial:
	@echo "Running clippy (serial / no parallel)..."
	@cargo clippy --all-targets --no-default-features -- -D warnings
	@echo "Clippy check complete!"

lint-py:
	@echo "Linting Python code with ruff..."
	@ruff check fastlowess/ tests/
	@echo "Python lint complete!"

lint-py-fix:
	@echo "Fixing Python lint issues with ruff..."
	@ruff check --fix fastlowess/ tests/
	@echo "Python lint fix complete!"

# Build
build: build-default build-serial

build-default:
	@echo "Building crate (default / parallel)..."
	@cargo build

build-serial:
	@echo "Building crate (serial / no parallel)..."
	@cargo build --no-default-features
	@echo "Build complete!"

# Maturin (Python package)
maturin: develop develop-release wheel wheel-release

develop:
	@echo "Building and installing Python package (development mode)..."
	@maturin develop
	@echo "Development install complete!"

develop-release:
	@echo "Building and installing Python package (release mode)..."
	@maturin develop --release
	@echo "Development install (release) complete!"

wheel:
	@echo "Building Python wheel..."
	@maturin build
	@echo "Wheel build complete!"

wheel-release:
	@echo "Building Python wheel (release)..."
	@maturin build --release
	@echo "Wheel build (release) complete!"

# Test
test: test-rust test-python

test-rust:
	@echo "Running Rust tests..."
	@cargo test
	@echo "Rust tests complete!"

test-python:
	@echo "Running Python tests..."
	@python -m pytest tests -v
	@echo "Python tests complete!"

# Documentation
doc: doc-default doc-serial doc-py
	@echo "Documentation build complete!"

doc-default:
	@echo "Building documentation (default / parallel)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

doc-serial:
	@echo "Building documentation (serial / no parallel)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --no-default-features

# Python (Sphinx) Documentation
doc-py:
	@echo "Building Sphinx documentation..."
	@pip install -r docs/requirements.txt
	@cd docs && make html
	@echo "Sphinx documentation build complete! Output is in docs/build/html/index.html"

# Examples
examples: example_batch example_online example_streaming

example_batch:
	@echo "Running batch example..."
	@python examples/batch_smoothing.py
	@echo "Batch example complete!"

example_online:
	@echo "Running online example..."
	@python examples/online_smoothing.py
	@echo "Online example complete!"

example_streaming:
	@echo "Running streaming example..."
	@python examples/streaming_smoothing.py
	@echo "Streaming example complete!"

# Installation
install:
	@echo "Installing Python package..."
	@pip3 install target/wheels/fastlowess-*.whl
	@echo "Python package installed!"

# Clean
clean: clean-rust clean-python
	@echo "Clean complete!"

clean-rust:
	@echo "Performing cargo clean..."
	@cargo clean
	@rm -rf Cargo.lock
	@rm -rf target

clean-python:
	@echo "Cleaning Python build artifacts..."
	@rm -rf coverage_html
	@rm -rf benchmarks
	@rm -rf validation
	@rm -rf .benchmarks
	@rm -rf target/wheels
	@rm -rf .pytest_cache
	@rm -rf __pycache__
	@rm -rf fastlowess/__pycache__
	@rm -rf tests/__pycache__
	@rm -rf *.egg-info
	@rm -rf .ruff_cache
	@rm -rf *.so
