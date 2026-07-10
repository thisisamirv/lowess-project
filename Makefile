# ==============================================================================
# Configuration
# ==============================================================================
FEATURE_SET ?= all
RUN_GPU_TESTS ?= auto

# Make shell commands fail on error
.SHELLFLAGS := -ec

UNAME_S := $(shell uname -s)

ifeq ($(OS),Windows_NT)
	HOST_PLATFORM := windows
	PATH_SEPARATOR := ;
	STAT_SIZE_CMD := stat -c%s
else ifeq ($(UNAME_S),Darwin)
	HOST_PLATFORM := macos
	PATH_SEPARATOR := :
	STAT_SIZE_CMD := stat -f%z
else
	HOST_PLATFORM := linux
	PATH_SEPARATOR := :
	STAT_SIZE_CMD := stat -c%s
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

# Python interpreter
PYTHON ?= python
PYO3_PYTHON ?= $(PYTHON)
NODE ?= node

# lowess crate
LOWESS_PKG := lowess
LOWESS_DIR := crates/lowess
LOWESS_FEATURES := std dev
LOWESS_EXAMPLES := batch_smoothing online_smoothing streaming_smoothing

# fastLowess crate
FASTLOWESS_PKG := fastLowess
FASTLOWESS_DIR := crates/fastLowess
FASTLOWESS_FEATURES := cpu gpu dev
FASTLOWESS_EXAMPLES := fast_batch_smoothing fast_online_smoothing fast_streaming_smoothing

# Python bindings
PY_PKG := fastLowess-py
PY_DIR := bindings/python
PY_VENV := .venv
PY_TEST_DIR := tests/python
ifeq ($(OS),Windows_NT)
	PY_ACTIVATE := $(PY_VENV)/Scripts/activate
	PY_VENV_PYTHON := $(PY_VENV)/Scripts/python.exe
else
	PY_ACTIVATE := $(PY_VENV)/bin/activate
	PY_VENV_PYTHON := $(PY_VENV)/bin/python
endif

# R bindings
R_PKG_NAME := rfastlowess
R_PKG_VERSION = $(shell grep "^Version:" bindings/r/DESCRIPTION | sed 's/Version: //')
R_PKG_TARBALL = $(R_PKG_NAME)_$(R_PKG_VERSION).tar.gz
R_DIR := bindings/r
R_LIB_DIR := $(R_DIR)/.r-lib
R_CARGO_TARGET :=
R_CROSS_ENV :=
ifeq ($(OS),Windows_NT)
    R_CARGO_TARGET := --target x86_64-pc-windows-gnu
    R_CROSS_ENV := PATH="/c/rtools45/x86_64-w64-mingw32.static.posix/bin:$$PATH"
endif

# Julia bindings
JL_PKG := fastlowess-jl
JL_DIR := bindings/julia
JL_TEST_DIR := tests/julia

# Node.js bindings
NODE_PKG := fastlowess-node
NODE_DIR := bindings/nodejs
NODE_TEST_DIR := tests/nodejs

ifeq ($(HOST_PLATFORM),windows)
	NPM := npm.cmd
	NPX := npx.cmd
else
	NPM := npm
	NPX := npx
endif

# WebAssembly bindings
WASM_PKG := fastlowess-wasm
WASM_DIR := bindings/wasm
WASM_TEST_DIR := tests/wasm

# C++ bindings
CPP_PKG := fastlowess-cpp
CPP_DIR := bindings/cpp
CPP_CARGO_TARGET :=
# Use the dedicated release-c profile (panic=abort) so no GCC unwind symbols
# end up in the static archive or DLL, making the artifact compatible with
# MSVC, Clang, and GCC consumers without any MinGW runtime dependency.
CPP_CARGO_PROFILE := --profile release-c
CPP_LIBRARY_DIR := target/release-c

ifeq ($(OS),Windows_NT)
	# Detect whether MinGW GCC is the active C++ toolchain (e.g. rtools45, MSYS2).
	# gcc -dumpmachine reports a mingw triple when MinGW is first in PATH.
	# Use the GNU Rust target in that case so the static lib uses MinGW ABI
	# (avoids __chkstk / type_info vtable link errors from MSVC-only symbols).
	# Otherwise keep the MSVC Rust target for native MSVC builds.
	_CPP_GCC_MACHINE := $(shell gcc -dumpmachine 2>/dev/null)
	ifneq ($(findstring mingw,$(_CPP_GCC_MACHINE)),)
		_CPP_WIN_TOOLCHAIN := mingw
		CPP_CARGO_TARGET := --target x86_64-pc-windows-gnu
		CPP_LIBRARY_DIR := target/x86_64-pc-windows-gnu/release-c
	else
		_CPP_WIN_TOOLCHAIN := msvc
		CPP_CARGO_TARGET := --target x86_64-pc-windows-msvc
		CPP_LIBRARY_DIR := target/x86_64-pc-windows-msvc/release-c
	endif
endif

# Julia native library paths and symbol scanners
ifeq ($(HOST_PLATFORM),windows)
	JL_SHARED_LIB := target/release/fastlowess_jl.dll
	JL_EXPORT_SCAN := objdump -p $(JL_SHARED_LIB)
else ifeq ($(HOST_PLATFORM),macos)
	JL_SHARED_LIB := target/release/libfastlowess_jl.dylib
	JL_EXPORT_SCAN := nm -gU $(JL_SHARED_LIB)
else
	JL_SHARED_LIB := target/release/libfastlowess_jl.so
	JL_EXPORT_SCAN := nm -D $(JL_SHARED_LIB)
endif

JL_SHARED_LIB_ABS := $(abspath $(JL_SHARED_LIB))

ifeq ($(HOST_PLATFORM),windows)
	CPP_SHARED_LIB := $(CPP_LIBRARY_DIR)/fastlowess_cpp.dll
	CPP_EXPORT_SCAN := objdump -p $(CPP_SHARED_LIB)
	ifeq ($(_CPP_WIN_TOOLCHAIN),mingw)
		# MinGW: GNU import library (.dll.a)
		CPP_TEST_LIB := $(CPP_LIBRARY_DIR)/libfastlowess_cpp.dll.a
		# Use MinGW Makefiles generator if mingw32-make is available (single-config,
		# binary lands in the build dir).  Otherwise fall back to the default CMake
		# generator on this host (typically Visual Studio, multi-config, binary in
		# Release/ sub-directory).
		_HAVE_MINGW32_MAKE := $(shell mingw32-make --version 2>/dev/null | head -c 3)
		ifneq ($(_HAVE_MINGW32_MAKE),)
			CPP_CMAKE_GENERATOR := -G "MinGW Makefiles"
			CPP_TEST_BUILD := cmake --build .
			CPP_TEST_RUN := PATH="../../../$(CPP_LIBRARY_DIR)$(PATH_SEPARATOR)$$PATH" ./test_fastlowess_suite.exe
		else
			CPP_CMAKE_GENERATOR :=
			CPP_TEST_BUILD := cmake --build . --config Release
			CPP_TEST_RUN := PATH="../../../$(CPP_LIBRARY_DIR)$(PATH_SEPARATOR)$$PATH" ./Release/test_fastlowess_suite.exe
		endif
	else
		# MSVC: import library (.lib); multi-config generator puts binary in Release/
		CPP_TEST_LIB := $(CPP_LIBRARY_DIR)/fastlowess_cpp.lib
		CPP_CMAKE_GENERATOR :=
		CPP_TEST_BUILD := cmake --build . --config Release
		CPP_TEST_RUN := PATH="../../../$(CPP_LIBRARY_DIR)$(PATH_SEPARATOR)$$PATH" ./Release/test_fastlowess_suite.exe
	endif
else ifeq ($(HOST_PLATFORM),macos)
	CPP_SHARED_LIB := $(CPP_LIBRARY_DIR)/libfastlowess_cpp.dylib
	CPP_TEST_LIB := $(CPP_SHARED_LIB)
	CPP_EXPORT_SCAN := nm -gU $(CPP_SHARED_LIB)
	CPP_CMAKE_GENERATOR :=
	CPP_TEST_BUILD := make
	CPP_TEST_RUN := ./test_fastlowess_suite
else
	CPP_SHARED_LIB := $(CPP_LIBRARY_DIR)/libfastlowess_cpp.so
	CPP_TEST_LIB := $(CPP_SHARED_LIB)
	CPP_EXPORT_SCAN := nm -D $(CPP_SHARED_LIB)
	CPP_CMAKE_GENERATOR :=
	CPP_TEST_BUILD := make
	CPP_TEST_RUN := ./test_fastlowess_suite
endif

CPP_LIBRARY_DIR_ABS := $(abspath $(CPP_LIBRARY_DIR))
CPP_TEST_LIB_ABS := $(abspath $(CPP_TEST_LIB))

ifeq ($(HOST_PLATFORM),windows)
	CPP_EXAMPLE_RUN_ENV := PATH="$(CPP_LIBRARY_DIR)$(PATH_SEPARATOR)$$PATH"
else ifeq ($(HOST_PLATFORM),macos)
	CPP_EXAMPLE_RUN_ENV := DYLD_LIBRARY_PATH=$(CPP_LIBRARY_DIR)
else
	CPP_EXAMPLE_RUN_ENV := LD_LIBRARY_PATH=$(CPP_LIBRARY_DIR)
endif

# Examples directory
EXAMPLES_DIR := examples

# Documentation
DOCS_VENV := docs-venv

# Temporary directory for build checks
TEMP ?= /tmp
ifeq ($(OS),Windows_NT)
    TEMP := /tmp
endif

# ==============================================================================
# lowess crate
# ==============================================================================
ISOLATE ?= true
all: ISOLATE := false

lowess:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py crates/lowess -- "$(MAKE)" _lowess_impl; \
	else \
		"$(MAKE)" _lowess_impl; \
	fi

_lowess_impl:
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
	@cargo clippy -p examples --examples --all-features -- -D warnings || exit 1
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@echo "Testing (no-default-features)..."
	@cargo test -q -p lowess-project-tests --test $(LOWESS_PKG) --no-default-features
	@for feature in $(LOWESS_FEATURES); do \
		echo "Testing ($$feature)..."; \
		cargo test -q -p lowess-project-tests --test $(LOWESS_PKG) --features $$feature || exit 1; \
	done
	@echo "=============================================================================="
	@echo "All $(LOWESS_PKG) crate checks completed successfully!"

ensure-llvm-cov:
	@cargo llvm-cov --version > /dev/null 2>&1 || (echo "Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov && cargo llvm-cov install-llvm-tools)

lowess-coverage: ensure-llvm-cov
	@echo "Running $(LOWESS_PKG) coverage..."
	@LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov --workspace --test $(LOWESS_PKG) --all-features \
		--ignore-filename-regex 'crates[/\\]fastLowess[/\\]|bindings[/\\]|benchmarks[/\\]|examples[/\\]|tests[/\\]'

lowess-clean:
	@echo "Cleaning $(LOWESS_PKG) crate..."
	@cargo clean -p $(LOWESS_PKG)
	@rm -rf $(LOWESS_DIR)/coverage_html
	@rm -rf $(LOWESS_DIR)/Cargo.lock
	@rm -rf $(LOWESS_DIR)/benchmarks
	@rm -rf $(LOWESS_DIR)/validation
	@echo "$(LOWESS_PKG) clean complete!"

# ==============================================================================
# fastLowess crate
# ==============================================================================
fastLowess:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py crates/fastLowess -- "$(MAKE)" _fastLowess_impl; \
	else \
		"$(MAKE)" _fastLowess_impl; \
	fi

_fastLowess_impl:
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
	@cargo test -q -p lowess-project-tests --test $(FASTLOWESS_PKG) --no-default-features
	@for feature in $(FASTLOWESS_FEATURES); do \
		echo "Testing ($$feature)..."; \
		if [ "$$feature" = "gpu" ]; then \
			if [ "$(EFFECTIVE_RUN_GPU_TESTS)" = "true" ]; then \
				cargo test -q -p lowess-project-tests --test $(FASTLOWESS_PKG) --features $$feature -- --test-threads=1 || exit 1; \
			else \
				echo "Skipping GPU tests on $(HOST_PLATFORM); set RUN_GPU_TESTS=true to force them."; \
			fi; \
		else \
			cargo test -q -p lowess-project-tests --test $(FASTLOWESS_PKG) --features $$feature || exit 1; \
		fi; \
	done
	@echo "=============================================================================="
	@echo "All $(FASTLOWESS_PKG) crate checks completed successfully!"

fastLowess-coverage: ensure-llvm-cov
	@echo "Running $(FASTLOWESS_PKG) coverage..."
	@LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov --workspace --test $(FASTLOWESS_PKG) --all-features \
		--ignore-filename-regex 'crates[/\\]lowess[/\\]|bindings[/\\]|benchmarks[/\\]|examples[/\\]|tests[/\\]'

fastLowess-clean:
	@echo "Cleaning $(FASTLOWESS_PKG) crate..."
	@cargo clean -p $(FASTLOWESS_PKG)
	@rm -rf $(FASTLOWESS_DIR)/coverage_html
	@rm -rf $(FASTLOWESS_DIR)/benchmarks
	@rm -rf $(FASTLOWESS_DIR)/validation
	@echo "$(FASTLOWESS_PKG) clean complete!"

# ==============================================================================
# Python bindings
# ==============================================================================
python:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py bindings/python -- "$(MAKE)" _python_impl; \
	else \
		"$(MAKE)" _python_impl; \
	fi

_python_impl:
	@echo "Running $(PY_PKG) checks..."
	@echo "=============================================================================="
	@echo "0. Environment Setup..."
	@echo "=============================================================================="
	@if [ ! -d "$(PY_VENV)" ]; then $(PYTHON) -m venv $(PY_VENV); fi
	@. $(PY_ACTIVATE) && python -c "import shutil, site; from pathlib import Path; [shutil.rmtree(path, ignore_errors=True) for base in site.getsitepackages() for path in Path(base).glob('~ip*')]"
	@. $(PY_ACTIVATE) && python -m pip cache purge >/dev/null 2>&1 || true
	@echo "Installing Python packages (pip, pytest, numpy, maturin, ruff)..."
	@. $(PY_ACTIVATE) && python -m pip install -q --no-cache-dir --upgrade pip >/dev/null
	@. $(PY_ACTIVATE) && python -m pip install -q --no-cache-dir pytest numpy maturin ruff >/dev/null
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(PY_PKG) -- --check
	@. $(PY_ACTIVATE) && ruff format $(PY_DIR)/python/ $(PY_TEST_DIR)/ examples/python/
	@echo "=============================================================================="
	@echo "2. Linting..."
	@echo "=============================================================================="
	@VIRTUAL_ENV= PYO3_PYTHON=$(PYO3_PYTHON) cargo clippy -q -p $(PY_PKG) --all-targets -- -D warnings
	@. $(PY_ACTIVATE) && ruff check $(PY_DIR)/python/ $(PY_TEST_DIR)/ examples/python/
	@echo "=============================================================================="
	@echo "3. Building..."
	@echo "=============================================================================="
	@. $(PY_ACTIVATE) && cd $(PY_DIR) && maturin develop -q
	@echo "=============================================================================="
	@echo "4. Testing..."
	@echo "=============================================================================="
	@VIRTUAL_ENV= PYO3_PYTHON=$(PYO3_PYTHON) cargo test -q -p $(PY_PKG)
	@. $(PY_ACTIVATE) && python -m pytest $(PY_TEST_DIR) -q
	@echo "$(PY_PKG) checks completed successfully!"

python-coverage: ensure-llvm-cov
	@echo "Running $(PY_PKG) coverage..."
	@cd $(PY_DIR) && LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov -p $(PY_PKG) --all-targets

python-clean:
	@echo "Cleaning $(PY_PKG)..."
	@cargo clean -p $(PY_PKG)
	@rm -rf $(PY_DIR)/coverage_html
	@rm -rf $(PY_DIR)/benchmarks
	@rm -rf $(PY_DIR)/validation
	@rm -rf $(PY_DIR)/.benchmarks
	@rm -rf $(PY_DIR)/target/wheels
	@rm -rf $(PY_DIR)/.pytest_cache
	@rm -rf $(PY_DIR)/__pycache__
	@rm -rf examples/python/plots/
	@rm -rf $(PY_DIR)/python/fastlowess/__pycache__
	@rm -rf $(PY_DIR)/python/fastlowess/*so
	@rm -rf $(PY_TEST_DIR)/__pycache__
	@rm -rf $(PY_DIR)/*.egg-info
	@rm -rf $(PY_DIR)/.ruff_cache
	@rm -rf $(PY_DIR)/*.so
	@rm -rf $(PY_DIR)/Cargo.lock
	@echo "$(PY_PKG) clean complete!"

# ==============================================================================
# R bindings
# ==============================================================================
r:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py bindings/r -- "$(MAKE)" _r_impl; \
	else \
		"$(MAKE)" _r_impl; \
	fi

_r_impl:
	@echo "Running $(R_PKG_NAME) checks..."
	@# Sync version from Cargo.toml to DESCRIPTION
	@VERSION=$$(grep "^version =" $(R_DIR)/src/Cargo.toml | head -n1 | sed 's/version = "\(.*\)"/\1/'); \
	sed -i.bak "s/^Version: .*/Version: $$VERSION/" $(R_DIR)/DESCRIPTION; \
	rm -f $(R_DIR)/DESCRIPTION.bak; \
	echo "Synced DESCRIPTION version to $$VERSION"
	@if [ -f $(R_DIR)/src/Cargo.toml.orig ]; then \
		mv $(R_DIR)/src/Cargo.toml.orig $(R_DIR)/src/Cargo.toml; \
	elif [ ! -f $(R_DIR)/src/Cargo.toml ] && [ -f $(R_DIR)/src/Cargo.toml.test ]; then \
		cp $(R_DIR)/src/Cargo.toml.test $(R_DIR)/src/Cargo.toml; \
	fi
	@echo "=============================================================================="
	@echo "1. Patching Cargo.toml for isolated build..."
	@echo "=============================================================================="
	@cp $(R_DIR)/src/Cargo.toml $(R_DIR)/src/Cargo.toml.orig
	@# Extract values from root Cargo.toml [workspace.package] section and update R binding's Cargo.toml
	@# Metadata sync disabled by user request
	@# (Only cleaning up workspace/patch/vendor directives below)
	@sed -i.bak '/^\[workspace\]/d' $(R_DIR)/src/Cargo.toml; \
	sed -i.bak '/^\[patch\.crates-io\]/d' $(R_DIR)/src/Cargo.toml; \
	sed -i.bak '/^lowess = { path = "vendor\/lowess" }/d' $(R_DIR)/src/Cargo.toml; \
	rm -f $(R_DIR)/src/Cargo.toml.bak; \
	rm -rf $(R_DIR)/*.Rcheck $(R_DIR)/*.BiocCheck $(R_DIR)/src/target $(R_DIR)/target $(R_DIR)/src/vendor; \
	echo "" >> $(R_DIR)/src/Cargo.toml
	@mkdir -p $(R_DIR)/src/.cargo && cp $(R_DIR)/src/cargo-config.toml $(R_DIR)/src/.cargo/config.toml
	@echo "Patched $(R_DIR)/src/Cargo.toml"
	@echo "=============================================================================="
	@echo "2. Installing development packages (R & Python)..."
	@echo "=============================================================================="
	@echo "Installing Python build packages (tomli, tomli_w)..."
	@$(PYTHON) -m pip install -q tomli tomli_w >/dev/null 2>&1 || true
	@echo "Installing R required packages (BiocManager, styler, testthat, rmarkdown, knitr, lintr, roxygen2, pkgdown, remotes)..."
	@echo "This may take a while depending on your internet connection and system performance."
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "suppressMessages({ lib <- Sys.getenv('R_LIBS_USER'); dir.create(lib, recursive = TRUE, showWarnings = FALSE); .libPaths(c(lib, .libPaths())); options(repos = c(CRAN = 'https://cloud.r-project.org'), warn = 1); required_pkgs <- c('BiocManager', 'styler', 'testthat', 'rmarkdown', 'knitr', 'lintr', 'roxygen2', 'pkgdown', 'remotes'); missing <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE, lib.loc = lib)]; if (length(missing) > 0L) { tryCatch({ install.packages(missing, lib = lib, type = ifelse(Sys.info()[['sysname']] == 'Darwin', 'both', 'source'), INSTALL_opts = '--no-test-load', quiet = TRUE, dependencies = NA, Ncpus = parallel::detectCores()); still_missing <- missing[!vapply(missing, requireNamespace, logical(1), quietly = TRUE, lib.loc = lib)]; if (length(still_missing) > 0L) stop('Required R packages not available: ', paste(still_missing, collapse = ', '), call. = FALSE) }, error = function(err) stop('Failed to install required packages: ', conditionMessage(err), call. = FALSE)) }; optional_pkgs <- c('covr', 'prettycode', 'toml', 'V8', 'visNetwork'); missing_opt <- optional_pkgs[!vapply(optional_pkgs, requireNamespace, logical(1), quietly = TRUE, lib.loc = lib)]; if (length(missing_opt) > 0L) { tryCatch(install.packages(missing_opt, lib = lib, quiet = TRUE, dependencies = NA, Ncpus = parallel::detectCores()), error = function(err) invisible(NULL)) } })" >/dev/null
	@echo "Installing R srr package..."
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "suppressMessages({ lib <- Sys.getenv('R_LIBS_USER'); if (!requireNamespace('srr', quietly = TRUE, lib.loc = lib)) { options(repos = c(ropenscireviewtools = 'https://ropensci-review-tools.r-universe.dev', CRAN = 'https://cloud.r-project.org')); tryCatch(suppressWarnings(install.packages('srr', lib = lib, quiet = TRUE)), error = function(err) invisible(NULL)) } })" >/dev/null
	@echo "Installing R Bioconductor packages (BiocStyle, BiocCheck)..."
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "suppressMessages({ lib <- Sys.getenv('R_LIBS_USER'); if (!requireNamespace('BiocManager', quietly = TRUE, lib.loc = lib)) stop('Required R package not available: BiocManager', call. = FALSE); bioc_pkgs <- c('BiocStyle'); missing_bioc <- bioc_pkgs[!vapply(bioc_pkgs, requireNamespace, logical(1), quietly = TRUE, lib.loc = lib)]; if (length(missing_bioc) > 0L) { tryCatch({ BiocManager::install(missing_bioc, lib = lib, update = FALSE, ask = FALSE, force = TRUE, quiet = TRUE); still_missing <- missing_bioc[!vapply(missing_bioc, requireNamespace, logical(1), quietly = TRUE, lib.loc = lib)]; if (length(still_missing) > 0L) stop('Required Bioconductor packages not available: ', paste(still_missing, collapse = ', '), call. = FALSE) }, error = function(err) stop('Failed to install Bioconductor packages: ', conditionMessage(err), call. = FALSE)) }; optional_bioc <- c('BiocCheck'); missing_opt_bioc <- optional_bioc[!vapply(optional_bioc, requireNamespace, logical(1), quietly = TRUE, lib.loc = lib)]; if (length(missing_opt_bioc) > 0L) { tryCatch(suppressWarnings(BiocManager::install(missing_opt_bioc, lib = lib, update = FALSE, ask = FALSE, force = TRUE, quiet = TRUE)), error = function(err) invisible(NULL)) } })" >/dev/null
	@echo "Installing R ropensci packages (pkgcheck, pkgstats)..."
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "suppressMessages({ lib <- Sys.getenv('R_LIBS_USER'); options(repos = c('https://ropensci.r-universe.dev', 'https://cloud.r-project.org')); install_optional <- function(pkg) { if (requireNamespace(pkg, quietly = TRUE, lib.loc = lib)) return(invisible(TRUE)); tryCatch(suppressWarnings(install.packages(pkg, lib = lib, quiet = TRUE)), error = function(err) invisible(NULL)); invisible(TRUE) }; invisible(vapply(c('pkgcheck', 'pkgstats'), install_optional, logical(1))) })" >/dev/null
	@echo "R development packages installed!"
	@echo "=============================================================================="
	@echo "3. Vendoring..."
	@echo "=============================================================================="
	@echo "Updating and re-vendoring crates.io dependencies..."
	@# Step 1: Clean R package Cargo.toml for vendoring
	@dev/prepare_cargo.py clean $(R_DIR)/src/Cargo.toml -q
	@# Step 2: Prepare vendor directory with local crates
	@rm -rf $(R_DIR)/src/vendor $(R_DIR)/src/vendor.tar.xz
	@mkdir -p $(R_DIR)/src/vendor
	@cp -rL crates/fastLowess $(R_DIR)/src/vendor/
	@cp -rL crates/lowess $(R_DIR)/src/vendor/
	@rm -rf $(R_DIR)/src/vendor/fastLowess/target $(R_DIR)/src/vendor/lowess/target
	@rm -f $(R_DIR)/src/vendor/fastLowess/Cargo.lock $(R_DIR)/src/vendor/lowess/Cargo.lock
	@rm -f $(R_DIR)/src/vendor/fastLowess/README.md $(R_DIR)/src/vendor/fastLowess/CHANGELOG.md
	@rm -f $(R_DIR)/src/vendor/lowess/README.md $(R_DIR)/src/vendor/lowess/CHANGELOG.md
	@# Step 3: Patch local crates (remove workspace inheritance, strip GPU deps)
	@dev/patch_vendor_crates.py Cargo.toml $(R_DIR)/src/vendor -q
	@# Step 4: Create dummy checksum files for local crates
	@echo '{"files":{},"package":null}' > $(R_DIR)/src/vendor/lowess/.cargo-checksum.json
	@echo '{"files":{},"package":null}' > $(R_DIR)/src/vendor/fastLowess/.cargo-checksum.json
	@# Step 5: Add workspace isolation to R package
	@dev/prepare_cargo.py isolate $(R_DIR)/src/Cargo.toml -q
	@# Step 6: Vendor crates.io dependencies
	@(cd $(R_DIR)/src && cargo vendor -q --no-delete vendor)
	@# Step 7: Regenerate checksums after vendoring
	@dev/clean_checksums.py -q $(R_DIR)/src/vendor
	@echo "Creating vendor.tar.xz archive (including Cargo.lock)..."
	@(cd $(R_DIR)/src && (tar --sort=name --mtime='1970-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner --xz --create --file=vendor.tar.xz vendor Cargo.lock 2>/dev/null || tar --xz --create --file=vendor.tar.xz vendor Cargo.lock))
	@rm -rf $(R_DIR)/src/vendor
	@echo "Vendor update complete. Archive: $(R_DIR)/src/vendor.tar.xz"
	@if [ -f $(R_DIR)/src/vendor.tar.xz ] && [ ! -d $(R_DIR)/src/vendor ]; then \
		echo "Extending vendor.tar.xz..."; \
		(cd $(R_DIR)/src && tar -xf vendor.tar.xz) && \
		rm -f $(R_DIR)/src/vendor/*/CITATION.cff && \
		rm -f $(R_DIR)/src/vendor/*/CITATION; \
	fi
	@echo "=============================================================================="
	@echo "4. Building..."
	@echo "=============================================================================="
	@(cd $(R_DIR)/src && $(R_CROSS_ENV) cargo build -q --release $(R_CARGO_TARGET) || (mv Cargo.toml.orig Cargo.toml && exit 1))
	@rm -rf $(R_DIR)/src/.cargo
	@echo "=============================================================================="
	@echo "4a. Formatting..."
	@echo "=============================================================================="
	@cd $(R_DIR)/src && cargo fmt -q
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript dev/style_pkg.R $(R_DIR) || true
	@cd $(R_DIR)/src && cargo fmt -- --check || (echo "Run 'cargo fmt' to fix"; exit 1)
	@cd $(R_DIR)/src && $(R_CROSS_ENV) cargo clippy -q $(R_CARGO_TARGET) -- -D warnings
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "lib <- Sys.getenv('R_LIBS_USER'); if (!requireNamespace('lintr', quietly = TRUE, lib.loc = lib)) stop('Required R package not available: lintr', call. = FALSE); my_linters <- lintr::linters_with_defaults(indentation_linter = lintr::indentation_linter(indent = 4L), object_name_linter = NULL, commented_code_linter = NULL, object_usage_linter = NULL); lints <- c(lintr::lint_dir('$(R_DIR)/R', linters = my_linters), lintr::lint_dir('tests/r/testthat', linters = my_linters), lintr::lint_dir('examples/r', linters = my_linters)); print(lints); if (length(lints) > 0L) quit(status = 1)"
	@echo "=============================================================================="
	@echo "4b. Documentation..."
	@echo "=============================================================================="
	@rm -rf $(R_DIR)/*.Rcheck
	@cd $(R_DIR)/src && RUSTDOCFLAGS="-D warnings" $(R_CROSS_ENV) cargo doc -q --no-deps $(R_CARGO_TARGET)
	@cd $(R_DIR) && R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "lib <- Sys.getenv('R_LIBS_USER'); options(repos = c(CRAN = 'https://cloud.r-project.org')); for (pkg in c('roxygen2', 'srr')) { if (!requireNamespace(pkg, quietly = TRUE, lib.loc = lib)) suppressWarnings(try(install.packages(pkg, lib = lib, quiet = TRUE), silent = TRUE)) }; if (!requireNamespace('roxygen2', quietly = TRUE, lib.loc = lib)) stop('Required R package not available for roxygen regeneration: roxygen2', call. = FALSE); has_srr <- requireNamespace('srr', quietly = TRUE, lib.loc = lib); roclets <- if (has_srr) c('namespace', 'rd', 'srr::srr_stats_roclet') else c('namespace', 'rd'); roxygen2::roxygenise(package.dir = '.', roclets = roclets, load_code = roxygen2::load_pkgload)"
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript dev/fix_rd_style.R
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "if (!requireNamespace('rmarkdown', quietly = TRUE, lib.loc = Sys.getenv('R_LIBS_USER')) || !rmarkdown::pandoc_available()) { message('\nERROR: Pandoc is required to build R Markdown vignettes but is not available.\nPlease install Pandoc (https://pandoc.org/installing.html) and ensure it is in your PATH.\n'); quit(status = 1) }"
	@cd $(R_DIR) && R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "lib <- Sys.getenv('R_LIBS_USER'); options(repos = c(CRAN = 'https://cloud.r-project.org')); for (pkg in c('pkgdown', 'rmarkdown', 'knitr')) { if (!requireNamespace(pkg, quietly = TRUE, lib.loc = lib)) install.packages(pkg, lib = lib, quiet = TRUE) }; if (!requireNamespace('pkgdown', quietly = TRUE, lib.loc = lib) || !requireNamespace('rmarkdown', quietly = TRUE, lib.loc = lib) || !requireNamespace('knitr', quietly = TRUE, lib.loc = lib)) stop('Required R packages not available for pkgdown site build', call. = FALSE); pkgdown::build_site(quiet = TRUE, install = TRUE)"
	@rm -f $(R_DIR)/.gitignore
	@echo "=============================================================================="
	@echo "4c. Building..."
	@echo "=============================================================================="
	@cd $(R_DIR) && ../../dev/prepare_cran.sh
	@cd $(R_DIR) && R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) R CMD build .
	@echo "=============================================================================="
	@echo "5. Installing..."
	@echo "=============================================================================="
	@rm -rf $(R_DIR)/00LOCK-$(R_PKG_NAME)
	@mkdir -p $(R_LIB_DIR)
	@echo "Installing R runtime dependencies (BiocManager, BiocGenerics, testthat)..."
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "suppressMessages({ lib <- Sys.getenv('R_LIBS_USER'); options(repos = c(CRAN = 'https://cloud.r-project.org')); if (!requireNamespace('BiocManager', quietly = TRUE, lib.loc = lib)) install.packages('BiocManager', lib = lib, quiet = TRUE); if (!requireNamespace('BiocGenerics', quietly = TRUE, lib.loc = lib)) BiocManager::install('BiocGenerics', lib = lib, ask = FALSE, update = FALSE, quiet = TRUE); if (!requireNamespace('testthat', quietly = TRUE, lib.loc = lib)) install.packages('testthat', lib = lib, quiet = TRUE) })" >/dev/null
	@cd $(R_DIR) && R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) R CMD INSTALL -l .r-lib $(R_PKG_TARBALL)
	@echo "=============================================================================="
	@echo "8. Testing..."
	@echo "=============================================================================="
	@mkdir -p $(R_DIR)/src/.cargo && cp $(R_DIR)/src/cargo-config.toml $(R_DIR)/src/.cargo/config.toml
	@if [ -n "$(R_CARGO_TARGET)" ]; then \
	    GCC_LIBDIR="c:/rtools45/x86_64-w64-mingw32.static.posix/lib/gcc/x86_64-w64-mingw32.static.posix/14.3.0"; \
	    test -f "$$GCC_LIBDIR/libgcc_eh.a" || x86_64-w64-mingw32.static.posix-ar rcs "$$GCC_LIBDIR/libgcc_eh.a"; \
	fi
	@cd $(R_DIR)/src && $(R_CROSS_ENV) cargo test -q --no-run $(R_CARGO_TARGET)
	@rm -rf $(R_DIR)/src/.cargo
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "Sys.setenv(NOT_CRAN='true'); testthat::test_dir('tests/r/testthat', package = 'rfastlowess')"
	@echo "=============================================================================="
	@echo "9. Submission checks..."
	@echo "=============================================================================="
	@cd $(R_DIR) && R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) R_MAKEVARS_USER=$(CURDIR)/dev/Makevars.check R CMD check --as-cran --no-manual $(R_PKG_TARBALL) || true
	@cd $(R_DIR) && R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "if (requireNamespace('BiocCheck', quietly=TRUE, lib.loc = Sys.getenv('R_LIBS_USER'))) BiocCheck::BiocCheck('$(R_PKG_TARBALL)', new_package=FALSE)" || true
	@echo "Package size (Limit: 5MB):"
	@ls -lh $(R_DIR)/$(R_PKG_TARBALL) || true
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "lib <- Sys.getenv('R_LIBS_USER'); if (!requireNamespace('pkgstats', quietly = TRUE, lib.loc = lib) || !requireNamespace('pkgcheck', quietly = TRUE, lib.loc = lib)) { message('Skipping pkgcheck because pkgstats/pkgcheck are not available in R_LIBS_USER'); quit(status = 0) }; library(pkgstats, lib.loc = lib); library(pkgcheck, lib.loc = lib); token <- Sys.getenv('GITHUB_TOKEN', ''); if (nzchar(token) && !nzchar(Sys.getenv('GITHUB_PAT', ''))) Sys.setenv(GITHUB_PAT = token); tryCatch(pkgcheck(use_cache = FALSE), error = function(err) { msg <- conditionMessage(err); if (grepl('GitHub API error', msg, ignore.case = TRUE) || grepl('rate limit exceeded', msg, ignore.case = TRUE) || grepl('timeout', msg, ignore.case = TRUE) || grepl('could not resolve host', msg, ignore.case = TRUE)) { message('Skipping pkgcheck due to external GitHub/network failure: ', msg); return(invisible(NULL)); }; stop(err) })"
	@if [ -f $(R_DIR)/src/Cargo.toml.orig ]; then mv $(R_DIR)/src/Cargo.toml.orig $(R_DIR)/src/Cargo.toml; fi

	@echo "All $(R_PKG_NAME) checks completed successfully!"

r-coverage:
	@echo "Calculating $(R_PKG_NAME) coverage..."
	@cd $(R_DIR) && NOT_CRAN=true Rscript -e "\
	  if (!requireNamespace('covr', quietly = TRUE)) { message('covr missing'); quit(status=0) }; \
	  cov <- covr::package_coverage(); \
	  covr::zero_coverage(cov); \
	  print(cov)"

r-clean:
	@echo "Cleaning $(R_PKG_NAME)..."
	@if [ -d $(R_DIR)/src/target ]; then \
		rm -rf $(R_DIR)/src/target 2>/dev/null || \
		(command -v docker >/dev/null && docker run --rm -v "$(PWD)/$(R_DIR)":/pkg ghcr.io/r-universe-org/build-wasm:latest rm -rf /pkg/src/target) || \
		echo "Warning: Failed to clean src/target"; \
	fi
	@(cd $(R_DIR)/src && cargo clean 2>/dev/null || true)
	@rm -rf $(R_DIR)/src/vendor $(R_DIR)/target $(R_LIB_DIR)
	@rm -rf $(R_DIR)/$(R_PKG_NAME).Rcheck $(R_DIR)/..Rcheck $(R_DIR)/$(R_PKG_NAME).BiocCheck
	@rm -f $(R_DIR)/$(R_PKG_NAME)_*.tar.gz
	@rm -rf $(R_DIR)/src/*.o $(R_DIR)/src/*.so $(R_DIR)/src/*.dll $(R_DIR)/src/Cargo.toml.orig $(R_DIR)/src/Cargo.lock $(R_DIR)/Cargo.lock
	@rm -rf $(R_DIR)/doc $(R_DIR)/Meta $(R_DIR)/vignettes/*.html $(R_DIR)/README.html
	@$(PYTHON) -c "from pathlib import Path; [path.unlink() for path in Path(r'$(R_DIR)').rglob('*.Rout')]"
	@Rscript -e "try(remove.packages('$(R_PKG_NAME)'), silent = TRUE)" || true
	@rm -rf $(R_DIR)/src/Makevars $(R_DIR)/rfastlowess*.tgz
	@rm -rf $(R_DIR)/benchmarks $(R_DIR)/validation $(R_DIR)/docs
	@echo "$(R_PKG_NAME) clean complete!"

# ==============================================================================
# Julia bindings
# ==============================================================================
julia:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py bindings/julia -- "$(MAKE)" _julia_impl; \
	else \
		"$(MAKE)" _julia_impl; \
	fi

_julia_impl:
	@echo "Running $(JL_PKG) checks..."
	@# Backup and adjust Project.toml for local testing
	@PROJECT_TOML="$(JL_DIR)/julia/Project.toml"; \
	BACKUP="$$PROJECT_TOML.bak"; \
	cp "$$PROJECT_TOML" "$$BACKUP"; \
	trap 'if [ -f "$$BACKUP" ]; then mv "$$BACKUP" "$$PROJECT_TOML"; echo "=============================================================================="; echo "Restored $$PROJECT_TOML"; fi' EXIT; \
	echo "=============================================================================="; \
	echo "0. Local environment setup (relaxing JLL constraint)..."; \
	echo "=============================================================================="; \
	LATEST=$$(julia -e 'using Pkg; Pkg.activate(temp=true); try Pkg.add("fastlowess_jll"); pkgs = Pkg.dependencies(); v = [p.version for (_,p) in pkgs if p.name == "fastlowess_jll"]; isempty(v) ? print("1.0.0") : print(first(v)) catch; print("1.0.0") end' | cut -d'+' -f1); \
	CURRENT=$$(grep "^version =" "$$PROJECT_TOML" | cut -d"\"" -f2); \
	julia -e "using TOML; path = \"$$PROJECT_TOML\"; p = TOML.parsefile(path); if haskey(get(p, \"compat\", Dict()), \"fastlowess_jll\"); p[\"compat\"][\"fastlowess_jll\"] = \"$$LATEST, $$CURRENT\"; open(path, \"w\") do io; TOML.print(io, p); end; end"; \
	echo "Modified $$PROJECT_TOML (fastlowess_jll = \"$$LATEST, $$CURRENT\") using TOML parser."; \
	"$(MAKE)" _julia_checks_internal

_julia_checks_internal:
	@echo "=============================================================================="
	@echo "0. Commit hash update..."
	@echo "=============================================================================="
	@git fetch origin main 2>/dev/null || true
	@COMMIT=$$(git rev-parse origin/main 2>/dev/null) && \
		COMMIT="$$COMMIT" $(PYTHON) -c 'from pathlib import Path; import os, re; path = Path("dev/build_tarballs_julia.jl"); text = path.read_text(encoding="utf-8"); commit = os.environ["COMMIT"]; new_text, _ = re.subn(r"GitSource\\(\"[^\"]*\",\\s*\"[a-f0-9]+\"\\)", f"GitSource(\"https://github.com/thisisamirv/lowess-project.git\", \"{commit}\")", text); path.write_text(new_text, encoding="utf-8")' && \
		echo "Commit: $$COMMIT"
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cd $(JL_DIR) && cargo fmt -- --check
	@echo "Formatting complete!"
	@echo "=============================================================================="
	@echo "2. Linting..."
	@echo "=============================================================================="
	@cd $(JL_DIR) && cargo clippy -q --all-targets -- -D warnings
	@echo "Linting Julia files..."
	@julia dev/format_julia.jl || true
	@julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format(["bindings/julia/julia", "tests/julia", "examples/julia"], verbose=true, overwrite=false) ? exit(0) : exit(1)'
	@echo "=============================================================================="
	@echo "3. Building..."
	@echo "=============================================================================="
	@cd $(JL_DIR) && cargo build -q --release
	@cd $(JL_DIR) && RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps
	@echo "=============================================================================="
	@echo "4. Testing Rust library..."
	@echo "=============================================================================="
	@cd $(JL_DIR) && cargo test -q
	@echo "=============================================================================="
	@echo "5. Verifying library exports..."
	@echo "=============================================================================="
	@$(JL_EXPORT_SCAN) 2>/dev/null | grep -q jl_lowess_new || \
		(echo "Error: jl_lowess_new not exported"; exit 1)
	@$(JL_EXPORT_SCAN) 2>/dev/null | grep -q jl_streaming_lowess_new || \
		(echo "Error: jl_streaming_lowess_new not exported"; exit 1)
	@$(JL_EXPORT_SCAN) 2>/dev/null | grep -q jl_online_lowess_new || \
		(echo "Error: jl_online_lowess_new not exported"; exit 1)
	@$(JL_EXPORT_SCAN) 2>/dev/null | grep -q jl_lowess_free_result || \
		(echo "Error: jl_lowess_free_result not exported"; exit 1)
	@echo "All exports verified!"
	@echo "=============================================================================="
	@echo "5b. ABI size check (limit: 5 MB)..."
	@echo "=============================================================================="
	@[ $$($(STAT_SIZE_CMD) $(JL_SHARED_LIB)) -le 5242880 ] || \
		(echo "Error: $(JL_SHARED_LIB) exceeds 5 MB ABI size limit"; exit 1)
	@echo "ABI size OK."
	@echo "=============================================================================="
	@echo "6. Testing Julia bindings..."
	@echo "=============================================================================="
	@export FASTLOWESS_LIB=$(JL_SHARED_LIB_ABS) && \
	julia --project=$(JL_DIR)/julia -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()' && \
	julia --project=$(JL_DIR)/julia tests/julia/test_FastLOWESS.jl
	@echo "=============================================================================="
	@echo "7. Aqua.jl package quality..."
	@echo "=============================================================================="
	@export FASTLOWESS_LIB=$(JL_SHARED_LIB_ABS) && \
	julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path="$(JL_DIR)/julia"); \
		Pkg.add("Aqua"); using Aqua, FastLOWESS; \
		Aqua.test_all(FastLOWESS; ambiguities=false, stale_deps=(ignore=[:Aqua],))'
	@echo "=============================================================================="
	@echo "8. JET.jl type-inference check..."
	@echo "=============================================================================="
	@export FASTLOWESS_LIB=$(JL_SHARED_LIB_ABS) && \
	julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path="$(JL_DIR)/julia"); \
		Pkg.add("JET"); using JET, FastLOWESS; \
		report = JET.report_package(FastLOWESS); \
		show(stderr, MIME("text/plain"), report); println(stderr); \
		if length(JET.get_reports(report)) > 0; @error "JET found type errors"; exit(1); end'
	@echo "$(JL_PKG) checks completed successfully!"
	@echo ""
	@echo "To use in Julia:"
	@echo "  julia> using Pkg"
	@echo "  julia> Pkg.develop(path=\"$(JL_DIR)/julia\")"
	@echo "  julia> using FastLOWESS"

julia-clean:
	@echo "Cleaning $(JL_PKG)..."
	@cargo clean -p $(JL_PKG)
	@rm -rf $(JL_DIR)/target
	@rm -rf $(JL_DIR)/julia/Manifest.toml
	@rm -rf $(JL_DIR)/Cargo.lock
	@echo "$(JL_PKG) clean complete!"

# ==============================================================================
# Node.js bindings
# ==============================================================================
nodejs:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py bindings/nodejs -- "$(MAKE)" _nodejs_impl; \
	else \
		"$(MAKE)" _nodejs_impl; \
	fi

_nodejs_impl:
	@echo "Running $(NODE_PKG) checks..."
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(NODE_PKG) -- --check
	@echo "=============================================================================="
	@echo "2. Linting & Building..."
	@echo "=============================================================================="
	@cargo clippy -q -p $(NODE_PKG) --all-targets -- -D warnings
	@echo "Linting Node.js files..."
	@cd $(NODE_DIR) && $(NPM) install
	@cd $(NODE_DIR) && $(NPM) audit || true
	@$(NODE) dev/check_js_licenses.mjs $(NODE_DIR) --fail-on-gpl
	@cd $(NODE_DIR) && $(NPX) -y depcheck --ignores="fastlowess-*,oxlint"
	@cd $(NODE_DIR) && ($(NPM) outdated | grep -v "fastlowess-" || true)
	@cd $(NODE_DIR) && $(NPM) ci --dry-run
	@cd $(NODE_DIR) && $(NPX) -y -p typescript tsc index.d.ts --noEmit --allowJs
	@$(NPX) -y oxlint $(NODE_DIR)/index.js tests/nodejs/test_fastlowess.js examples/nodejs/*.js
	@cd $(NODE_DIR) && $(NPM) run build
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@cd $(NODE_DIR) && $(NPM) test
	@echo "$(NODE_PKG) checks completed successfully!"

nodejs-clean:
	@echo "Cleaning $(NODE_PKG)..."
	@cargo clean -p $(NODE_PKG)
	@rm -rf $(NODE_DIR)/node_modules $(NODE_DIR)/fastlowess.*.node
	@echo "$(NODE_PKG) clean complete!"

# ==============================================================================
# WebAssembly bindings
# ==============================================================================
wasm:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py bindings/wasm -- "$(MAKE)" _wasm_impl; \
	else \
		"$(MAKE)" _wasm_impl; \
	fi

_wasm_impl:
	@echo "Running $(WASM_PKG) checks..."
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(WASM_PKG) -- --check
	@echo "=============================================================================="
	@echo "2. Linting & Building..."
	@echo "=============================================================================="
	@cargo clippy -q -p $(WASM_PKG) --all-targets -- -D warnings
	@echo "Linting WASM JS files..."
	@cd $(WASM_DIR) && $(NPM) install -q
	@cd $(WASM_DIR) && $(NPM) audit || true
	@$(NODE) dev/check_js_licenses.mjs $(WASM_DIR) --fail-on-gpl
	@cd $(WASM_DIR) && $(NPX) -y depcheck --ignores="oxlint"
	@cd $(WASM_DIR) && ($(NPM) outdated | grep -v "fastlowess-" || true)
	@cd $(WASM_DIR) && $(NPM) ci --dry-run
	@$(NPX) -y oxlint $(WASM_DIR)/src/*.js tests/wasm/*.js
	@if ! command -v wasm-pack >/dev/null 2>&1; then \
		echo "wasm-pack not found. Installing..."; \
		cargo install wasm-pack; \
	fi
	@cd $(WASM_DIR) && wasm-pack build --target nodejs --out-dir pkg
	@echo "Checking WASM size (Limit: 2MB)..."
	@[ $$($(STAT_SIZE_CMD) $(WASM_DIR)/pkg/fastlowess_wasm_bg.wasm) -le 2097152 ] || (echo "Error: WASM size exceeded 2MB"; exit 1)
	@echo "Building for Web (Examples)..."
	@cd $(WASM_DIR) && wasm-pack build --target web --out-dir pkg-web
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@cd $(WASM_DIR) && wasm-pack test --node
	@echo "Running JS tests..."
	@node --test tests/wasm/test_fastlowess_wasm.js
	@echo "=============================================================================="
	@echo "$(WASM_PKG) checks completed successfully!"

wasm-clean:
	@echo "Cleaning $(WASM_PKG)..."
	@cargo clean -p $(WASM_PKG)
	@rm -rf $(WASM_DIR)/pkg $(WASM_DIR)/pkg-web $(WASM_DIR)/node_modules
	@echo "$(WASM_PKG) clean complete!"

# ==============================================================================
# C++ bindings
# ==============================================================================
cpp:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py bindings/cpp -- "$(MAKE)" _cpp_impl; \
	else \
		"$(MAKE)" _cpp_impl; \
	fi

_cpp_impl:
	@echo "Running $(CPP_PKG) checks..."
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(CPP_PKG) -- --check
	@echo "=============================================================================="
	@echo "2. Linting & Building..."
	@echo "=============================================================================="
	@cargo clippy -q -p $(CPP_PKG) --all-targets $(CPP_CARGO_TARGET) -- -D warnings
	@echo "Linting C++ files..."
	@if command -v clang-tidy >/dev/null 2>&1; then \
		clang_tidy_log="$(TEMP)/clang-tidy-cpp.log"; \
		clang-tidy bindings/cpp/include/fastlowess.hpp tests/cpp/test_fastlowess.cpp examples/cpp/*.cpp -- -I bindings/cpp/include -std=c++17 > "$$clang_tidy_log" 2>&1; \
		clang_tidy_status=$$?; \
		grep -Ev '^(\[[0-9]+/[0-9]+\] Processing file |[0-9]+ warnings generated\.|Suppressed [0-9]+ warnings \([0-9]+ in non-user code\)\.|Use -header-filter=\.\* or leave it as default to display errors from all non-system headers\.|Use -system-headers to display errors from system headers as well\.)' "$$clang_tidy_log" || true; \
		rm -f "$$clang_tidy_log"; \
		if [ $$clang_tidy_status -ne 0 ]; then \
			echo "C++ linting failed"; \
			exit $$clang_tidy_status; \
		fi; \
	else \
		echo "clang-tidy not found. Skipping C++ lint pass."; \
	fi
	@echo "Running cppcheck..."
	@if command -v cppcheck >/dev/null 2>&1; then \
		cppcheck_log="$(TEMP)/cppcheck-cpp.log"; \
		if cppcheck --error-exitcode=1 --enable=warning,performance,portability \
			--suppress=missingInclude --suppress=missingIncludeSystem \
			-I $(CPP_DIR)/include \
			$(CPP_DIR)/include/fastlowess.hpp tests/cpp/test_fastlowess.cpp examples/cpp/ > "$$cppcheck_log" 2>&1; then \
			cat "$$cppcheck_log"; \
		else \
			cppcheck_status=$$?; \
			cat "$$cppcheck_log"; \
			if grep -q "Failed to load library configuration file 'std.cfg'" "$$cppcheck_log"; then \
				echo "cppcheck installation is broken. Skipping static analysis pass."; \
			else \
				rm -f "$$cppcheck_log"; \
				exit $$cppcheck_status; \
			fi; \
		fi; \
		rm -f "$$cppcheck_log"; \
	else \
		echo "cppcheck not found. Skipping static analysis pass."; \
	fi
	@if [ -n "$(CPP_CARGO_TARGET)" ]; then \
		_rust_target=$$(echo "$(CPP_CARGO_TARGET)" | sed 's/--target //'); \
		rustup target add "$$_rust_target" 2>/dev/null || true; \
	fi
	@cargo build -q -p $(CPP_PKG) $(CPP_CARGO_PROFILE) $(CPP_CARGO_TARGET)
	@echo "C header generated at $(CPP_DIR)/include/fastlowess.h"
	@echo "=============================================================================="
	@echo "2b. cbindgen idempotency check..."
	@echo "=============================================================================="
	@if ! command -v cbindgen >/dev/null 2>&1; then \
		echo "cbindgen not found. Installing latest version..."; \
		cargo install cbindgen --force; \
	fi
	@cbindgen --config $(CPP_DIR)/cbindgen.toml --crate $(CPP_PKG) --output $(TEMP)/fastlowess_new.h 2>/dev/null && \
		diff -q $(CPP_DIR)/include/fastlowess.h $(TEMP)/fastlowess_new.h > /dev/null || \
		(echo "Error: fastlowess.h is stale — run 'cargo build -p $(CPP_PKG) $(CPP_CARGO_PROFILE) $(CPP_CARGO_TARGET)' to regenerate"; exit 1)
	@echo "cbindgen header is up-to-date."
	@echo "=============================================================================="
	@echo "2c. Symbol export verification..."
	@echo "=============================================================================="
	@$(CPP_EXPORT_SCAN) 2>/dev/null | grep -q cpp_lowess_new || \
		(echo "Error: cpp_lowess_new not exported"; exit 1)
	@$(CPP_EXPORT_SCAN) 2>/dev/null | grep -q cpp_streaming_new || \
		(echo "Error: cpp_streaming_new not exported"; exit 1)
	@$(CPP_EXPORT_SCAN) 2>/dev/null | grep -q cpp_online_new || \
		(echo "Error: cpp_online_new not exported"; exit 1)
	@echo "All C++ exports verified."
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@rm -rf tests/cpp/build
	@mkdir -p tests/cpp/build
	@cd tests/cpp/build && cmake $(CPP_CMAKE_GENERATOR) -DFASTLOWESS_LIB="$(CPP_TEST_LIB_ABS)" -DFASTLOWESS_LIB_DIR="$(CPP_LIBRARY_DIR_ABS)" .. && $(CPP_TEST_BUILD) && $(CPP_TEST_RUN)
	@echo "=============================================================================="
	@echo "3b. Valgrind memory check..."
	@echo "=============================================================================="
	@if [ "$(HOST_PLATFORM)" != "linux" ]; then \
		echo "Valgrind: skipped on $(HOST_PLATFORM)."; \
	elif ! command -v valgrind >/dev/null 2>&1; then \
		echo "Valgrind not found. Skipping memory check."; \
	else \
		valgrind --leak-check=full --error-exitcode=1 --quiet \
		tests/cpp/build/test_fastlowess_suite 2>&1 || \
		(echo "Error: Valgrind detected memory errors"; exit 1); \
		echo "Valgrind: no leaks."; \
	fi
	@echo "$(CPP_PKG) checks completed successfully!"

cpp-clean:
	@echo "Cleaning $(CPP_PKG)..."
	@cargo clean -p $(CPP_PKG)
	@rm -rf $(CPP_DIR)/include/fastlowess.h $(CPP_DIR)/bin $(CPP_DIR)/build
	@rm -rf tests/cpp/build
	@echo "$(CPP_PKG) clean complete!"

# ==============================================================================
# Examples
# ==============================================================================
examples: examples-lowess examples-fastLowess examples-python examples-r examples-julia examples-nodejs examples-cpp
	@echo "All examples completed successfully!"

examples-lowess:
	@echo "Running $(LOWESS_PKG) examples..."
	@echo "=============================================================================="
	@echo "Running examples (no-default-features)..."
	@for example in $(LOWESS_EXAMPLES); do \
		cargo run -q -p examples --example $$example --no-default-features || exit 1; \
	done
	@for feature in $(LOWESS_FEATURES); do \
		echo "Running examples ($$feature)..."; \
		for example in $(LOWESS_EXAMPLES); do \
			cargo run -q -p examples --example $$example --features $$feature || exit 1; \
		done; \
	done
	@echo "=============================================================================="

examples-fastLowess:
	@echo "Running $(FASTLOWESS_PKG) examples..."
	@echo "=============================================================================="
	@echo "Running examples (no-default-features)..."
	@for example in $(FASTLOWESS_EXAMPLES); do \
		cargo run -q -p examples --example $$example --no-default-features > /dev/null || exit 1; \
	done
	@for feature in $(FASTLOWESS_FEATURES); do \
		echo "Running examples with feature: $$feature"; \
		for example in $(FASTLOWESS_EXAMPLES); do \
			if [ "$$feature" = "dev" ]; then \
				cargo run -q -p examples --example $$example --features $$feature || exit 1; \
			else \
				cargo run -q -p examples --example $$example --features $$feature > /dev/null || exit 1; \
			fi; \
		done; \
	done
	@echo "=============================================================================="

examples-python:
	@echo "Running $(PY_PKG) examples..."
	@echo "=============================================================================="
	@. $(PY_ACTIVATE) && pip install -q matplotlib
	@. $(PY_ACTIVATE) && python $(EXAMPLES_DIR)/python/batch_smoothing.py
	@. $(PY_ACTIVATE) && python $(EXAMPLES_DIR)/python/streaming_smoothing.py
	@. $(PY_ACTIVATE) && python $(EXAMPLES_DIR)/python/online_smoothing.py
	@echo "=============================================================================="

examples-r:
	@echo "Running $(R_PKG_NAME) examples..."
	@echo "=============================================================================="
	@Rscript $(EXAMPLES_DIR)/r/batch_smoothing.R
	@Rscript $(EXAMPLES_DIR)/r/streaming_smoothing.R
	@Rscript $(EXAMPLES_DIR)/r/online_smoothing.R
	@echo "=============================================================================="

examples-julia:
	@echo "Running $(JL_PKG) examples..."
	@echo "=============================================================================="
	@julia --project=$(JL_DIR)/julia $(EXAMPLES_DIR)/julia/batch_smoothing.jl
	@julia --project=$(JL_DIR)/julia $(EXAMPLES_DIR)/julia/streaming_smoothing.jl
	@julia --project=$(JL_DIR)/julia $(EXAMPLES_DIR)/julia/online_smoothing.jl
	@echo "=============================================================================="

examples-nodejs:
	@echo "Running $(NODE_PKG) examples..."
	@echo "=============================================================================="
	@cd $(NODE_DIR) && node ../../$(EXAMPLES_DIR)/nodejs/batch_smoothing.js
	@cd $(NODE_DIR) && node ../../$(EXAMPLES_DIR)/nodejs/online_smoothing.js
	@cd $(NODE_DIR) && node ../../$(EXAMPLES_DIR)/nodejs/streaming_smoothing.js
	@echo "=============================================================================="

examples-cpp:
	@echo "Running $(CPP_PKG) examples..."
	@echo "=============================================================================="
	@mkdir -p $(CPP_DIR)/bin
	@g++ -O3 $(EXAMPLES_DIR)/cpp/batch_smoothing.cpp -o $(CPP_DIR)/bin/batch_smoothing -I$(CPP_DIR)/include -L$(CPP_LIBRARY_DIR) -lfastlowess_cpp -lpthread -ldl -lm
	@g++ -O3 $(EXAMPLES_DIR)/cpp/streaming_smoothing.cpp -o $(CPP_DIR)/bin/streaming_smoothing -I$(CPP_DIR)/include -L$(CPP_LIBRARY_DIR) -lfastlowess_cpp -lpthread -ldl -lm
	@g++ -O3 $(EXAMPLES_DIR)/cpp/online_smoothing.cpp -o $(CPP_DIR)/bin/online_smoothing -I$(CPP_DIR)/include -L$(CPP_LIBRARY_DIR) -lfastlowess_cpp -lpthread -ldl -lm
	@$(CPP_EXAMPLE_RUN_ENV) $(CPP_DIR)/bin/batch_smoothing
	@$(CPP_EXAMPLE_RUN_ENV) $(CPP_DIR)/bin/streaming_smoothing
	@$(CPP_EXAMPLE_RUN_ENV) $(CPP_DIR)/bin/online_smoothing
	@echo "=============================================================================="

# ==============================================================================
# Development checks
# ==============================================================================
check-msrv:
	@echo "Checking MSRV..."
	@$(PYTHON) dev/check_msrv.py

# ==============================================================================
# Documentation
# ==============================================================================
docs:
	@echo "Building documentation..."
	@if [ ! -d "$(DOCS_VENV)" ]; then $(PYTHON) -m venv $(DOCS_VENV); fi
	@. $(DOCS_VENV)/$(if $(filter $(HOST_PLATFORM),windows),Scripts,bin)/activate && pip install -q -r docs/requirements.txt && mkdocs build

docs-serve:
	@echo "Starting documentation server..."
	@if [ ! -d "$(DOCS_VENV)" ]; then $(PYTHON) -m venv $(DOCS_VENV); fi
	@. $(DOCS_VENV)/$(if $(filter $(HOST_PLATFORM),windows),Scripts,bin)/activate && pip install -q -r docs/requirements.txt && mkdocs serve

docs-clean:
	@echo "Cleaning documentation build..."
	@rm -rf site/ $(DOCS_VENV)/
	@echo "Documentation clean complete!"

docs-test:
	@echo "Running doc snippet tests..."
	$(PYTHON) dev/verify_snippets.py --timeout 120

# ==============================================================================
# All targets
# ==============================================================================
all: lowess fastLowess python r julia nodejs wasm cpp check-msrv docs-test
	@echo "All checks completed successfully!"

all-coverage: lowess-coverage fastLowess-coverage python-coverage r-coverage
	@echo "All coverage completed!"

all-clean: r-clean lowess-clean fastLowess-clean python-clean julia-clean nodejs-clean wasm-clean cpp-clean
	@echo "Cleaning project root..."
	@cargo clean
	@rm -rf target Cargo.lock .venv .ruff_cache .pytest_cache site docs-venv build bindings/python/.venv bindings/python/target crates/fastLowess/target crates/lowess/target .vscode tests/.pytest_cache local_*.tar.gz bindings/r/.r-lib bindings/r/docs
	@rm -f Rplots.pdf .gitignore~ ..gitignore.un~
	@rm -rf r.Rcheck/
	@rm -f tests/r/testthat/Rplots.pdf
	@rm -rf examples/cpp/bin/
	@rm -f bindings/nodejs/fastlowess.node
	@rm -f bindings/python/python/fastlowess/*.pyd bindings/python/python/fastlowess/*.pdb
	@rm -rf bindings/r/tests/
	@echo "All clean completed!"

.PHONY: lowess lowess-coverage lowess-clean fastLowess fastLowess-coverage fastLowess-clean python python-coverage python-clean r r-coverage r-clean julia julia-clean julia-update-commit nodejs nodejs-clean wasm wasm-clean cpp cpp-clean check-msrv docs docs-serve docs-test docs-clean all all-coverage all-clean examples examples-lowess examples-fastLowess examples-python examples-r examples-julia examples-nodejs examples-cpp ensure-llvm-cov
