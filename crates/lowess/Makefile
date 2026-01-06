# Run all local checks (formatting, linting, building, tests, docs)
check: fmt clippy build test doc examples
	@echo "All checks completed successfully!"

# Coverage (requires cargo-llvm-cov and llvm)
coverage:
	@echo "Running coverage report (text)..."
	@LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov --features std,dev
	@echo "Coverage report complete!"

# Formatting
fmt:
	@echo "Checking code formatting..."
	@cargo fmt --all -- --check
	@echo "Formatting check complete!"

# Linter
clippy: clippy-std clippy-no-std clippy-dev

clippy-std:
	@echo "Running clippy (std)..."
	@cargo clippy --all-targets --features std -- -D warnings

clippy-no-std:
	@echo "Running clippy (no-std)..."
	@cargo clippy --all-targets --no-default-features -- -D warnings
	@echo "Clippy check complete!"

clippy-dev:
	@echo "Running clippy (dev)..."
	@cargo clippy --all-targets --features dev -- -D warnings
	@echo "Clippy check complete!"

# Build
build: build-std build-no-std build-dev

build-std:
	@echo "Building crate (std)..."
	@cargo build --features std

build-no-std:
	@echo "Building crate (no-std)..."
	@cargo build --no-default-features
	@echo "Build complete!"

build-dev:
	@echo "Building crate (dev)..."
	@cargo build --features dev
	@echo "Build complete!"

# Test
test: test-std test-no-std

test-std:
	@echo "Running tests (std)..."
	@cargo test --workspace --features std,dev

test-no-std:
	@echo "Running tests (no-std)..."
	@cargo test --no-default-features --features dev
	@echo "Tests complete!"

# Documentation
doc: doc-std doc-no-std doc-dev
	@echo "Documentation build complete!"

doc-std:
	@echo "Building documentation (std)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features std

doc-no-std:
	@echo "Building documentation (no-std)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --no-default-features

doc-dev:
	@echo "Building documentation (dev)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features dev

# Examples
examples: example_batch example_online example_streaming

example_batch:
	@echo "Building examples..."
	@cargo run --example batch_smoothing
	@echo "Examples build complete!"

example_online:
	@echo "Building examples..."
	@cargo run --example online_smoothing
	@echo "Examples build complete!"

example_streaming:
	@echo "Building examples..."
	@cargo run --example streaming_smoothing
	@echo "Examples build complete!"

# Clean
clean:
	@echo "Cleaning..."
	@cargo clean
	@rm -rf Cargo.lock
	@rm -rf target
	@rm -rf coverage_html
	@rm -rf benchmarks
	@rm -rf validation
	@echo "Clean complete!"
