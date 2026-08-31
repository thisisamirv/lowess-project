# ==============================================================================
# Configuration
# ==============================================================================
FEATURE_SET ?= all
RUN_GPU_TESTS ?= auto

# Make shell commands fail on error
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

ifeq ($(RUN_GPU_TESTS),auto)
	ifeq ($(HOST_PLATFORM),linux)
		EFFECTIVE_RUN_GPU_TESTS := true
	else
		EFFECTIVE_RUN_GPU_TESTS := false
	endif
else
	EFFECTIVE_RUN_GPU_TESTS := $(RUN_GPU_TESTS)
endif

# lowess crate
LOWESS_PKG := lowess
LOWESS_DIR := crates/lowess
LOWESS_FEATURES := std dev

# fastLowess crate
FASTLOWESS_PKG := fastLowess
FASTLOWESS_DIR := crates/fastLowess
FASTLOWESS_FEATURES := cpu gpu dev

# Python bindings
PY_PKG := fastLowess-py
PY_DIR := bindings/python
PY_VENV := .venv
ifeq ($(OS),Windows_NT)
	PY_ACTIVATE    := $(PY_VENV)/Scripts/activate
	PY_VENV_PYTHON := $(PY_VENV)/Scripts/python.exe
else
	PY_ACTIVATE    := $(PY_VENV)/bin/activate
	PY_VENV_PYTHON := $(PY_VENV)/bin/python
endif

# R bindings
R_PKG_NAME := rfastlowess
R_DIR := bindings/r
R_LIB_DIR := $(R_DIR)/.r-lib

# Julia bindings
JL_DIR := bindings/julia

ifeq ($(HOST_PLATFORM),windows)
	JL_SHARED_LIB := target/release/fastlowess_jl.dll
else ifeq ($(HOST_PLATFORM),macos)
	JL_SHARED_LIB := target/release/libfastlowess_jl.dylib
else
	JL_SHARED_LIB := target/release/libfastlowess_jl.so
endif

# Node.js bindings
NODE_DIR := bindings/nodejs

# WebAssembly bindings
WASM_DIR := bindings/wasm

# C++ bindings
CPP_DIR := bindings/cpp
CPP_CARGO_PROFILE := --profile release-c
CPP_LIBRARY_DIR := target/release-c

ifeq ($(OS),Windows_NT)
	_CPP_GCC_MACHINE := $(shell gcc -dumpmachine 2>/dev/null)
	ifneq ($(findstring mingw,$(_CPP_GCC_MACHINE)),)
		CPP_LIBRARY_DIR := target/x86_64-pc-windows-gnu/release-c
	else
		CPP_LIBRARY_DIR := target/x86_64-pc-windows-msvc/release-c
	endif
endif

# ==============================================================================
# lowess crate
# ==============================================================================
lowess:
	@"$(MAKE)" -f crates/lowess/Makefile FEATURE_SET="$(FEATURE_SET)"

lowess-dev:
	@"$(MAKE)" -f crates/lowess/Makefile dev FEATURE_SET="$(FEATURE_SET)"

lowess-coverage:
	@"$(MAKE)" -f crates/lowess/Makefile coverage

lowess-clean:
	@"$(MAKE)" -f crates/lowess/Makefile clean

ensure-llvm-cov:
	@cargo llvm-cov --version > /dev/null 2>&1 || (echo "Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov && cargo llvm-cov install-llvm-tools)

# ==============================================================================
# fastLowess crate
# ==============================================================================
fastLowess:
	@"$(MAKE)" -f crates/fastLowess/Makefile \
		FEATURE_SET="$(FEATURE_SET)" \
		RUN_GPU_TESTS="$(RUN_GPU_TESTS)"

fastLowess-dev:
	@"$(MAKE)" -f crates/fastLowess/Makefile dev \
		FEATURE_SET="$(FEATURE_SET)" \
		RUN_GPU_TESTS="$(RUN_GPU_TESTS)"

fastLowess-coverage:
	@"$(MAKE)" -f crates/fastLowess/Makefile coverage

fastLowess-clean:
	@"$(MAKE)" -f crates/fastLowess/Makefile clean

# ==============================================================================
# Python bindings
# ==============================================================================
python:
	@"$(MAKE)" -f bindings/python/Makefile \
		PYTHON="$(PYTHON)" PYO3_PYTHON="$(PYO3_PYTHON)"

python-dev:
	@"$(MAKE)" -f bindings/python/Makefile dev \
		PYTHON="$(PYTHON)" PYO3_PYTHON="$(PYO3_PYTHON)"

python-coverage:
	@"$(MAKE)" -f bindings/python/Makefile coverage PYTHON="$(PYTHON)" PYO3_PYTHON="$(PYO3_PYTHON)"

python-clean:
	@"$(MAKE)" -f bindings/python/Makefile clean

# ==============================================================================
# R bindings
# ==============================================================================
r:
	@"$(MAKE)" -f bindings/r/Makefile

r-dev:
	@"$(MAKE)" -f bindings/r/Makefile dev

r-coverage:
	@"$(MAKE)" -f bindings/r/Makefile coverage

r-clean:
	@"$(MAKE)" -f bindings/r/Makefile clean PYTHON="$(PYTHON)"

# ==============================================================================
# Julia bindings
# ==============================================================================
julia:
	@"$(MAKE)" -f bindings/julia/Makefile PYTHON="$(PYTHON)"

julia-dev:
	@"$(MAKE)" -f bindings/julia/Makefile dev PYTHON="$(PYTHON)"

julia-clean:
	@"$(MAKE)" -f bindings/julia/Makefile clean

# ==============================================================================
# Node.js bindings
# ==============================================================================
nodejs:
	@"$(MAKE)" -f bindings/nodejs/Makefile

nodejs-dev:
	@"$(MAKE)" -f bindings/nodejs/Makefile dev

nodejs-clean:
	@"$(MAKE)" -f bindings/nodejs/Makefile clean

# ==============================================================================
# WebAssembly bindings
# ==============================================================================
wasm:
	@"$(MAKE)" -f bindings/wasm/Makefile

wasm-dev:
	@"$(MAKE)" -f bindings/wasm/Makefile dev

wasm-clean:
	@"$(MAKE)" -f bindings/wasm/Makefile clean

# ==============================================================================
# C++ bindings
# ==============================================================================
cpp:
	@"$(MAKE)" -f bindings/cpp/Makefile

cpp-dev:
	@"$(MAKE)" -f bindings/cpp/Makefile dev

cpp-clean:
	@"$(MAKE)" -f bindings/cpp/Makefile clean

# ==============================================================================
# Go bindings
# ==============================================================================
go:
	@"$(MAKE)" -f bindings/go/Makefile

go-dev:
	@"$(MAKE)" -f bindings/go/Makefile dev

go-clean:
	@"$(MAKE)" -f bindings/go/Makefile clean

# ==============================================================================
# Development checks
# ==============================================================================
check-msrv:
	@echo "Checking MSRV..."
	@$(PYTHON) dev/check_msrv.py

# ==============================================================================
# Documentation
# ==============================================================================
docs-test:
	@echo "Running doc snippet tests..."
	@if [ -f "$(PY_VENV_PYTHON)" ]; then \
		$(PY_VENV_PYTHON) dev/verify_snippets.py --timeout 120; \
	else \
		$(PYTHON) dev/verify_snippets.py --timeout 120; \
	fi

# ==============================================================================
# All targets
# ==============================================================================
all: lowess fastLowess python r julia nodejs wasm cpp go check-msrv
	@echo "All checks completed successfully!"

all-dev: lowess-dev fastLowess-dev python-dev r-dev julia-dev nodejs-dev wasm-dev cpp-dev go-dev check-msrv
	@echo "All dev checks completed successfully!"

all-coverage: lowess-coverage fastLowess-coverage python-coverage r-coverage
	@echo "All coverage completed!"

all-clean: r-clean lowess-clean fastLowess-clean python-clean julia-clean nodejs-clean wasm-clean cpp-clean go-clean
	@echo "Cleaning project root..."
	@cargo clean
	@$(PYTHON) dev/kill_locked_venv.py $(PY_VENV)
	@git clean -fdX .
	@echo "All clean completed!"

.PHONY: lowess lowess-dev lowess-coverage lowess-clean fastLowess fastLowess-dev fastLowess-coverage fastLowess-clean python python-dev python-coverage python-clean r r-dev r-coverage r-clean julia julia-dev julia-clean julia-update-commit nodejs nodejs-dev nodejs-clean wasm wasm-dev wasm-clean cpp cpp-dev cpp-clean go go-dev go-clean check-msrv docs-test all all-dev all-coverage all-clean ensure-llvm-cov