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

PY_PKG      := fastLowess-py
PY_DIR      := bindings/python
PY_VENV     := .venv
PY_TEST_DIR := bindings/python/tests

ifeq ($(OS),Windows_NT)
	PY_ACTIVATE    := $(PY_VENV)/Scripts/activate
	PY_VENV_PYTHON := $(PY_VENV)/Scripts/python.exe
else
	PY_ACTIVATE    := $(PY_VENV)/bin/activate
	PY_VENV_PYTHON := $(PY_VENV)/bin/python
endif

.PHONY: default clean coverage ensure-llvm-cov examples

default:
	@echo "Running $(PY_PKG) checks..."
	@echo "=============================================================================="
	@echo "0. Environment Setup..."
	@echo "=============================================================================="
	@if [ ! -d "$(PY_VENV)" ]; then $(PYTHON) -m venv $(PY_VENV); fi
	@. $(PY_ACTIVATE) && python -c "import shutil, site; from pathlib import Path; [shutil.rmtree(path, ignore_errors=True) for base in site.getsitepackages() for path in Path(base).glob('~ip*')]"
	@. $(PY_ACTIVATE) && python -m pip cache purge >/dev/null 2>&1 || true
	@echo "Installing Python packages (pip, pytest, numpy, maturin, ruff)..."
	@. $(PY_ACTIVATE) && python -m pip install -q --no-cache-dir --upgrade pip >/dev/null
	@. $(PY_ACTIVATE) && python -m pip install -q --no-cache-dir pytest numpy maturin ruff matplotlib >/dev/null
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(PY_PKG) -- --check
	@. $(PY_ACTIVATE) && ruff format $(PY_DIR)/python/ $(PY_TEST_DIR)/ $(PY_DIR)/examples/
	@echo "=============================================================================="
	@echo "2. Linting..."
	@echo "=============================================================================="
	@VIRTUAL_ENV= PYO3_PYTHON=$(PYO3_PYTHON) cargo clippy -q -p $(PY_PKG) --all-targets -- -D warnings
	@. $(PY_ACTIVATE) && ruff check $(PY_DIR)/python/ $(PY_TEST_DIR)/ $(PY_DIR)/examples/
	@echo "=============================================================================="
	@echo "3. Building..."
	@echo "=============================================================================="
	@. $(PY_ACTIVATE) && cd $(PY_DIR) && maturin develop -q
	@echo "=============================================================================="
	@echo "4. Testing..."
	@echo "=============================================================================="
	@VIRTUAL_ENV= PYO3_PYTHON=$(PYO3_PYTHON) cargo test -q -p $(PY_PKG)
	@. $(PY_ACTIVATE) && python -m pytest $(PY_TEST_DIR) -q
	@"$(MAKE)" -f $(THIS_MK) examples
	@echo "$(PY_PKG) checks completed successfully!"

examples:
	@echo "Running $(PY_PKG) examples..."
	@echo "============================================================================="
	@. $(PY_ACTIVATE) && pip install -q matplotlib
	@. $(PY_ACTIVATE) && python $(PY_DIR)/examples/batch_smoothing.py
	@. $(PY_ACTIVATE) && python $(PY_DIR)/examples/streaming_smoothing.py
	@. $(PY_ACTIVATE) && python $(PY_DIR)/examples/online_smoothing.py
	@echo "============================================================================="

ensure-llvm-cov:
	@cargo llvm-cov --version > /dev/null 2>&1 || (echo "Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov && cargo llvm-cov install-llvm-tools)

coverage: ensure-llvm-cov
	@echo "Running $(PY_PKG) coverage..."
	@cd $(PY_DIR) && LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov -p $(PY_PKG) --all-targets

clean:
	@echo "Cleaning $(PY_PKG)..."
	@cargo clean -p $(PY_PKG)
	@rm -rf $(PY_DIR)/coverage_html
	@rm -rf $(PY_DIR)/benchmarks
	@rm -rf $(PY_DIR)/validation
	@rm -rf $(PY_DIR)/.benchmarks
	@rm -rf $(PY_DIR)/target/wheels
	@rm -rf $(PY_DIR)/.pytest_cache
	@rm -rf $(PY_DIR)/__pycache__
	@rm -rf $(PY_DIR)/examples/plots/
	@rm -rf $(PY_DIR)/python/fastlowess/__pycache__
	@rm -rf $(PY_DIR)/python/fastlowess/*so
	@rm -rf $(PY_TEST_DIR)/__pycache__
	@rm -rf $(PY_DIR)/*.egg-info
	@rm -rf $(PY_DIR)/.ruff_cache
	@rm -rf $(PY_DIR)/*.so
	@rm -rf $(PY_DIR)/Cargo.lock
	@echo "$(PY_PKG) clean complete!"
