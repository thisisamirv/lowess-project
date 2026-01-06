# Run all local checks (formatting, linting, building, tests, docs)
check: fmt clippy build test doc examples
	@echo "All checks completed successfully!"

# Coverage (requires cargo-llvm-cov and llvm)
coverage:
	@echo "Running coverage report (text)..."
	@LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov --features dev
	@echo "Coverage report complete!"

# Formatting
fmt:
	@echo "Checking code formatting..."
	@cargo fmt --all -- --check
	@echo "Formatting check complete!"

# Linter
clippy: clippy-cpu clippy-gpu
	@echo "All clippy checks completed successfully!"

clippy-cpu:
	@echo "Running clippy (cpu)..."
	@cargo clippy --all-targets --features cpu -- -D warnings
	@echo "Clippy check complete!"

clippy-gpu:
	@echo "Running clippy (gpu)..."
	@cargo clippy --all-targets --features gpu -- -D warnings
	@echo "Clippy check complete!"

# Build
build: build-cpu build-gpu

build-cpu:
	@echo "Building crate (cpu)..."
	@cargo build --features cpu
	@echo "Build complete!"

build-gpu:
	@echo "Building crate (gpu)..."
	@cargo build --features gpu
	@echo "Build complete!"

# Test
test: test-cpu test-gpu


test-cpu:
	@echo "Running tests (cpu)..."
	@cargo test --features cpu
	@echo "Tests complete!"

test-gpu:
	@echo "Running tests (gpu)..."
	@cargo test --features gpu
	@echo "Tests complete!"

# Documentation
doc: doc-cpu doc-gpu

doc-cpu:
	@echo "Building documentation (cpu)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features cpu
	@echo "Documentation build complete!"

doc-gpu:
	@echo "Building documentation (gpu)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features gpu
	@echo "Documentation build complete!"

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
	@echo "Performing cargo clean..."
	@cargo clean
	@rm -rf target
	@rm -rf Cargo.lock
	@rm -rf benchmarks
	@rm -rf validation
	@echo "Clean complete!"
