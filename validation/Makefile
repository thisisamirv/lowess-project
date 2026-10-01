# validation/Makefile — Visual validation and R property/reference tests.
#
# Targets:
#   visual   Generate CSVs used by visual_validation/plot.py
#   plot     Generate SVGs from those CSVs
#   r-tests  Run R property/reference and golden-fixture tests

PYTHON ?= python
VISUAL_DIR := visual_validation
R_LIB_DIR ?= ../bindings/r/.r-lib

.PHONY: visual plot r-tests

visual:
	@echo "Generating visual validation data..."
	@cd $(VISUAL_DIR) && cargo run --bin visual --release

plot:
	@echo "Generating visual validation plots..."
	@cd $(VISUAL_DIR) && $(PYTHON) plot.py

r-tests:
	@echo "Running R validation tests..."
	@R_LIBS_USER=$(abspath $(R_LIB_DIR)) Rscript \
	  -e "my_linters <- lintr::linters_with_defaults(indentation_linter = lintr::indentation_linter(indent = 4L), object_name_linter = NULL, commented_code_linter = NULL, object_usage_linter = NULL)" \
	  -e "lints <- c(lintr::lint_dir('property_tests', linters = my_linters), lintr::lint_dir('fixture_tests', linters = my_linters))" \
	  -e "print(lints)" \
	  -e "if (length(lints) > 0L) quit(status = 1)"
	@R_LIBS_USER=$(abspath $(R_LIB_DIR)) Rscript -e "Sys.setenv(NOT_CRAN='true'); testthat::test_dir('property_tests', package = 'rfastlowess'); testthat::test_dir('fixture_tests', package = 'rfastlowess')"