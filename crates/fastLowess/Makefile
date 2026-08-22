THIS_MK := $(lastword $(MAKEFILE_LIST))

.SHELLFLAGS := -ec

UNAME_S := $(shell uname -s)

ifeq ($(OS),Windows_NT)
	HOST_PLATFORM  := windows
	PATH_SEPARATOR := ;
	STAT_SIZE_CMD  := stat -c%s
	NPM := npm.cmd
	NPX := npx.cmd
else ifeq ($(UNAME_S),Darwin)
	HOST_PLATFORM  := macos
	PATH_SEPARATOR := :
	STAT_SIZE_CMD  := stat -f%z
	NPM := npm
	NPX := npx
else
	HOST_PLATFORM  := linux
	PATH_SEPARATOR := :
	STAT_SIZE_CMD  := stat -c%s
	NPM := npm
	NPX := npx
endif

PYTHON ?= python3
PYO3_PYTHON ?= $(PYTHON)
NODE ?= node

TEMP ?= /tmp
ifeq ($(OS),Windows_NT)
	TEMP := /tmp
endif

FEATURE_SET ?= all
RUN_GPU_TESTS ?= auto

FASTLOWESS_PKG := fastLowess
FASTLOWESS_DIR := crates/fastLowess
FASTLOWESS_FEATURES := cpu gpu dev

ifeq ($(RUN_GPU_TESTS),auto)
	ifeq ($(HOST_PLATFORM),linux)
		EFFECTIVE_RUN_GPU_TESTS := true
	else
		EFFECTIVE_RUN_GPU_TESTS := false
	endif
else
	EFFECTIVE_RUN_GPU_TESTS := $(RUN_GPU_TESTS)
endif

.PHONY: default dev clean coverage ensure-llvm-cov

default:
	@cargo build -q -p $(FASTLOWESS_PKG)

dev:
	@echo "Running $(FASTLOWESS_PKG) crate checks..."
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(FASTLOWESS_PKG) -- --check
	@echo "Formatting complete!"
	@echo "=============================================================================="
	@echo "2. Linting & Building..."
	@echo "=============================================================================="
	@for feature in $(FASTLOWESS_FEATURES) no-default-features; do \
		if [ "$$feature" = "no-default-features" ]; then \
			echo "Checking $(FASTLOWESS_PKG) (no-default-features)..."; \
			cargo clippy -q -p $(FASTLOWESS_PKG) --all-targets --no-default-features -- -D warnings || exit 1; \
			cargo build -q -p $(FASTLOWESS_PKG) --no-default-features || exit 1; \
			RUSTDOCFLAGS="-D warnings" cargo doc -q -p $(FASTLOWESS_PKG) --no-deps --no-default-features || exit 1; \
		else \
			echo "Checking $(FASTLOWESS_PKG) ($$feature)..."; \
			cargo clippy -q -p $(FASTLOWESS_PKG) --all-targets --features $$feature -- -D warnings || exit 1; \
			cargo build -q -p $(FASTLOWESS_PKG) --features $$feature || exit 1; \
			RUSTDOCFLAGS="-D warnings" cargo doc -q -p $(FASTLOWESS_PKG) --no-deps --features $$feature || exit 1; \
		fi; \
	done
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@echo "Testing (no-default-features)..."
	@cargo test -q -p $(FASTLOWESS_PKG) --test $(FASTLOWESS_PKG) --no-default-features
	@for feature in $(FASTLOWESS_FEATURES); do \
		echo "Testing ($$feature)..."; \
		if [ "$$feature" = "gpu" ]; then \
			if [ "$(EFFECTIVE_RUN_GPU_TESTS)" = "true" ]; then \
				cargo test -q -p $(FASTLOWESS_PKG) --test $(FASTLOWESS_PKG) --features $$feature -- --test-threads=1 || exit 1; \
			else \
				echo "Skipping GPU tests on $(HOST_PLATFORM); set RUN_GPU_TESTS=true to force them."; \
			fi; \
		else \
			cargo test -q -p $(FASTLOWESS_PKG) --test $(FASTLOWESS_PKG) --features $$feature || exit 1; \
		fi; \
	done
	@echo "=============================================================================="
	@echo "Doc snippet tests (rust)..."
	@echo "=============================================================================="
	@$(PYTHON) dev/verify_snippets.py --lang rust --timeout 120
	@echo "All $(FASTLOWESS_PKG) crate checks completed successfully!"

ensure-llvm-cov:
	@cargo llvm-cov --version > /dev/null 2>&1 || (echo "Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov && cargo llvm-cov install-llvm-tools)

coverage: ensure-llvm-cov
	@echo "Running $(FASTLOWESS_PKG) coverage..."
	@LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov -p $(FASTLOWESS_PKG) --test $(FASTLOWESS_PKG) --all-features \
		--ignore-filename-regex 'crates[/\\]lowess[/\\]|bindings[/\\]|benchmarks[/\\]|examples[/\\]|tests[/\\]'

clean:
	@echo "Cleaning $(FASTLOWESS_PKG) crate..."
	@cargo clean -p $(FASTLOWESS_PKG)
	@git clean -fdX -- $(FASTLOWESS_DIR)
	@echo "$(FASTLOWESS_PKG) clean complete!"