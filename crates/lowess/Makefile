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

PYTHON      ?= python3
PYO3_PYTHON ?= $(PYTHON)
NODE        ?= node

TEMP ?= /tmp
ifeq ($(OS),Windows_NT)
	TEMP := /tmp
endif

FEATURE_SET ?= all

LOWESS_PKG      := lowess
LOWESS_DIR      := crates/lowess
LOWESS_FEATURES := std dev
LOWESS_EXAMPLES := batch_smoothing online_smoothing streaming_smoothing

.PHONY: default clean coverage ensure-llvm-cov examples

default:
	@echo "Running $(LOWESS_PKG) crate checks..."
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(LOWESS_PKG) -- --check
	@echo "Formatting complete!"
	@echo "=============================================================================="
	@echo "2. Linting & Building..."
	@echo "=============================================================================="
	@if [ "$(FEATURE_SET)" = "all" ]; then \
		echo "Checking $(LOWESS_PKG) (std)..."; \
		cargo clippy -q -p $(LOWESS_PKG) --all-targets -- -D warnings || exit 1; \
		cargo build -q -p $(LOWESS_PKG) || exit 1; \
		RUSTDOCFLAGS="-D warnings" cargo doc -q -p $(LOWESS_PKG) --no-deps || exit 1; \
		echo "Checking $(LOWESS_PKG) (no-default-features)..."; \
		cargo clippy -q -p $(LOWESS_PKG) --all-targets --no-default-features -- -D warnings || exit 1; \
		cargo build -q -p $(LOWESS_PKG) --no-default-features || exit 1; \
		RUSTDOCFLAGS="-D warnings" cargo doc -q -p $(LOWESS_PKG) --no-deps --no-default-features || exit 1; \
		for feature in $(LOWESS_FEATURES); do \
			echo "Checking $(LOWESS_PKG) ($$feature)..."; \
			cargo clippy -q -p $(LOWESS_PKG) --all-targets --features $$feature -- -D warnings || exit 1; \
			cargo build -q -p $(LOWESS_PKG) --features $$feature || exit 1; \
			RUSTDOCFLAGS="-D warnings" cargo doc -q -p $(LOWESS_PKG) --no-deps --features $$feature || exit 1; \
		done; \
	else \
		cargo clippy -q -p $(LOWESS_PKG) --all-targets --features $(FEATURE_SET) -- -D warnings || exit 1; \
		cargo build -q -p $(LOWESS_PKG) --features $(FEATURE_SET) || exit 1; \
		RUSTDOCFLAGS="-D warnings" cargo doc -q -p $(LOWESS_PKG) --no-deps --features $(FEATURE_SET) || exit 1; \
	fi
	@cargo clippy -p $(LOWESS_PKG) --examples --all-features -- -D warnings || exit 1
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@echo "Testing (no-default-features)..."
	@cargo test -q -p $(LOWESS_PKG) --test $(LOWESS_PKG) --no-default-features
	@for feature in $(LOWESS_FEATURES); do \
		echo "Testing ($$feature)..."; \
		cargo test -q -p $(LOWESS_PKG) --test $(LOWESS_PKG) --features $$feature || exit 1; \
	done
	@echo "=============================================================================="
	@"$(MAKE)" -f $(THIS_MK) examples
	@echo "All $(LOWESS_PKG) crate checks completed successfully!"

examples:
	@echo "Running $(LOWESS_PKG) examples..."
	@echo "============================================================================="
	@echo "Running examples (no-default-features)..."
	@for example in $(LOWESS_EXAMPLES); do \
		cargo run -q -p $(LOWESS_PKG) --example $$example --no-default-features || exit 1; \
	done
	@for feature in $(LOWESS_FEATURES); do \
		echo "Running examples ($$feature)..."; \
		for example in $(LOWESS_EXAMPLES); do \
			cargo run -q -p $(LOWESS_PKG) --example $$example --features $$feature || exit 1; \
		done; \
	done
	@echo "============================================================================="

ensure-llvm-cov:
	@cargo llvm-cov --version > /dev/null 2>&1 || (echo "Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov && cargo llvm-cov install-llvm-tools)

coverage: ensure-llvm-cov
	@echo "Running $(LOWESS_PKG) coverage..."
	@LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov -p $(LOWESS_PKG) --test $(LOWESS_PKG) --all-features \
		--ignore-filename-regex 'crates[/\\]fastLowess[/\\]|bindings[/\\]|benchmarks[/\\]|examples[/\\]|tests[/\\]'

clean:
	@echo "Cleaning $(LOWESS_PKG) crate..."
	@cargo clean -p $(LOWESS_PKG)
	@rm -rf $(LOWESS_DIR)/coverage_html
	@rm -rf $(LOWESS_DIR)/Cargo.lock
	@rm -rf $(LOWESS_DIR)/benchmarks
	@rm -rf $(LOWESS_DIR)/validation
	@echo "$(LOWESS_PKG) clean complete!"
