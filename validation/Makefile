# validation/Makefile — Rust, Python, and R validation tests and visual tools.
#
# Targets:
#   visual   Generate CSVs used by rust_tests/plot.py
#   plot     Generate SVGs from those CSVs
#   r-tests  Run R property/reference and golden-fixture tests

PYTHON ?= python
RUST_TESTS_DIR := rust_tests
R_LIB_DIR ?= ../bindings/r/.r-lib
PYTHON_VENV ?= ../.venv
ifeq ($(OS),Windows_NT)
PYTHON_BINDING ?= $(PYTHON_VENV)/Scripts/python.exe
else
PYTHON_BINDING ?= $(PYTHON_VENV)/bin/python
endif

.PHONY: visual plot python-deps python-tests rust-tests validation-deps r-tests

visual:
	@echo "Generating visual validation data..."
	@cd $(RUST_TESTS_DIR) && cargo run --bin visual --release

plot:
	@echo "Generating visual validation plots..."
	@cd $(RUST_TESTS_DIR) && $(PYTHON) plot.py

python-deps:
	@if [ ! -x "$(PYTHON_BINDING)" ]; then $(PYTHON) -m venv "$(PYTHON_VENV)"; fi
	@if ! $(PYTHON_BINDING) -c "import fastlowess, numpy, pytest, hypothesis" >/dev/null 2>&1; then \
	  $(PYTHON_BINDING) -m pip install -q "numpy>=1.20.0" pytest "hypothesis>=6,<7" ../bindings/python; \
	fi

python-tests: python-deps
	@echo "Running Python-binding boundary validation..."
	@$(PYTHON_BINDING) -m pytest -q property_tests/test_boundary_padding.py

rust-tests:
	@echo "Running Rust core numerical validation..."
	@cargo test --manifest-path ../Cargo.toml -p rust-tests --test numerical_regressions

validation-deps:
	@echo "Installing Locfit and quickcheck into the validation library when missing..."
	@Rscript -e "lib <- normalizePath('$(R_LIB_DIR)', winslash='/', mustWork=FALSE); dir.create(lib, recursive=TRUE, showWarnings=FALSE); missing <- c('locfit', 'quickcheck')[!dir.exists(file.path(lib, c('locfit', 'quickcheck')))]; if (length(missing)) install.packages(missing, lib=lib, repos=getOption('repos')); if (length(missing)) stopifnot(all(vapply(missing, requireNamespace, logical(1), quietly=TRUE, lib.loc=lib)))"

r-tests: validation-deps
	@echo "Running R validation tests..."
	@Rscript \
	  -e ".libPaths(c(normalizePath('$(R_LIB_DIR)', winslash='/', mustWork=TRUE), .libPaths()))" \
	  -e "my_linters <- lintr::linters_with_defaults(indentation_linter = lintr::indentation_linter(indent = 4L), object_name_linter = NULL, commented_code_linter = NULL, object_usage_linter = NULL)" \
	  -e "lints <- c(lintr::lint_dir('property_tests', linters = my_linters), lintr::lint_dir('fixture_tests', linters = my_linters))" \
	  -e "print(lints)" \
	  -e "if (length(lints) > 0L) quit(status = 1)"
	@Rscript -e ".libPaths(c(normalizePath('$(R_LIB_DIR)', winslash='/', mustWork=TRUE), .libPaths()))" -e "Sys.setenv(NOT_CRAN='true'); testthat::test_dir('property_tests', package = 'rfastlowess'); testthat::test_dir('fixture_tests', package = 'rfastlowess')"