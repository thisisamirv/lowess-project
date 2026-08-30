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

PY_PKG := fastLowess-py
PY_DIR := bindings/python
PY_VENV := .venv
PY_TEST_DIR := bindings/python/tests

ifeq ($(OS),Windows_NT)
	PY_ACTIVATE    := $(PY_VENV)/Scripts/activate
	PY_VENV_PYTHON := $(PY_VENV)/Scripts/python.exe
else
	PY_ACTIVATE    := $(PY_VENV)/bin/activate
	PY_VENV_PYTHON := $(PY_VENV)/bin/python
endif

PY_DOCS_DIR := $(PY_DIR)/docs
PY_DOCS_BUILD := $(PY_DOCS_DIR)/_build

.PHONY: default dev docs docs-dev clean coverage ensure-llvm-cov

default:
	@$(PYTHON) -m pip install -q --user $(PY_DIR)

dev:
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
	@. $(PY_ACTIVATE) && ruff format $(PY_DIR)/python/ $(PY_TEST_DIR)/
	@echo "=============================================================================="
	@echo "2. Linting..."
	@echo "=============================================================================="
	@VIRTUAL_ENV= PYO3_PYTHON=$(PYO3_PYTHON) cargo clippy -q -p $(PY_PKG) --all-targets -- -D warnings
	@. $(PY_ACTIVATE) && ruff check $(PY_DIR)/python/ $(PY_TEST_DIR)/
	@echo "=============================================================================="
	@echo "3. Building..."
	@echo "=============================================================================="
	@. $(PY_ACTIVATE) && cd $(PY_DIR) && maturin develop -q
	@echo "=============================================================================="
	@echo "4. Testing..."
	@echo "=============================================================================="
	@VIRTUAL_ENV= PYO3_PYTHON=$(PYO3_PYTHON) cargo test -q -p $(PY_PKG)
	@. $(PY_ACTIVATE) && python -m pytest $(PY_TEST_DIR) -q
	@echo "=============================================================================="
	@echo "Doc snippet tests (python)..."
	@echo "=============================================================================="
	@$(PYTHON) dev/verify_snippets.py --lang python --timeout 120
	@$(PYTHON) dev/update_changelogs.py

ensure-llvm-cov:
	@cargo llvm-cov --version > /dev/null 2>&1 || (echo "Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov && cargo llvm-cov install-llvm-tools)

coverage: ensure-llvm-cov
	@echo "Running $(PY_PKG) coverage..."
	@cd $(PY_DIR) && LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov -p $(PY_PKG) --all-targets

clean:
	@echo "Cleaning $(PY_PKG)..."
	@cargo clean -p $(PY_PKG)
	@git clean -fdX -- $(PY_DIR)
	@echo "$(PY_PKG) clean complete!"
docs:
	@echo "Building $(PY_PKG) docs (Sphinx)..."
	@. $(PY_ACTIVATE) && python -m pip install -q --no-cache-dir sphinx furo myst-parser jupyter-sphinx sphinx-copybutton sphinxcontrib-mermaid ipykernel >/dev/null
	@. $(PY_ACTIVATE) && sphinx-build -b html $(PY_DOCS_DIR) $(PY_DOCS_BUILD)/html
	@echo "Docs built at $(PY_DOCS_BUILD)/html/index.html"

docs-dev:
	@echo "Serving $(PY_PKG) docs (Sphinx autobuild)..."
	@. $(PY_ACTIVATE) && python -m pip install -q --no-cache-dir sphinx-autobuild >/dev/null
	@. $(PY_ACTIVATE) && sphinx-autobuild $(PY_DOCS_DIR) $(PY_DOCS_BUILD)/html --open-browser