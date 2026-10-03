# validation/Makefile — Visual validation and R property/reference tests.
#
# Targets:
#   visual   Generate CSVs used by visual_validation/plot.py
#   plot     Generate SVGs from those CSVs
#   r-tests  Run R property/reference and golden-fixture tests

PYTHON ?= python
VISUAL_DIR := visual_validation
R_LIB_DIR ?= ../bindings/r/.r-lib
PYTHON_VENV ?= ../.venv
ifeq ($(OS),Windows_NT)
PYTHON_BINDING ?= $(PYTHON_VENV)/Scripts/python.exe
else
PYTHON_BINDING ?= $(PYTHON_VENV)/bin/python
endif

.PHONY: visual plot python-deps python-tests validation-deps r-tests

visual:
	@echo "Generating visual validation data..."
	@cd $(VISUAL_DIR) && cargo run --bin visual --release

plot:
	@echo "Generating visual validation plots..."
	@cd $(VISUAL_DIR) && $(PYTHON) plot.py

python-deps:
	@if [ ! -x "$(PYTHON_BINDING)" ]; then $(PYTHON) -m venv "$(PYTHON_VENV)"; fi
	@if ! $(PYTHON_BINDING) -c "import fastlowess, numpy, pytest, hypothesis" >/dev/null 2>&1; then \
	  $(PYTHON_BINDING) -m pip install -q "numpy>=1.20.0" pytest "hypothesis>=6,<7" ../bindings/python; \
	fi

python-tests: python-deps
	@echo "Running Python-binding boundary validation..."
	@$(PYTHON_BINDING) -m pytest -q property_tests/test_boundary_padding.py

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