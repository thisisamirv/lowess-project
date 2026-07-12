# validation/Makefile — Run validation and comparison for fastLowess vs R.
#
# Targets:
#   fastlowess-visual    Build and run the visual Rust binary (writes output/fastLowess/)
#   fastlowess-validate  Build and run the validate Rust binary (writes output/fastLowess/)
#   r-validate           Run R reference script (writes output/r/)
#   compare              Compare R and fastLowess outputs with compare.py
#   plot                 Generate plots with plot.py

PYTHON ?= python
CARGO_DIR := fastLowess

.PHONY: fastlowess-visual fastlowess-validate r-validate compare plot

fastlowess-visual:
	@echo "Building and running fastLowess visual..."
	@cd $(CARGO_DIR) && cargo run --bin visual --release

fastlowess-validate:
	@echo "Building and running fastLowess validate..."
	@cd $(CARGO_DIR) && cargo run --bin validate --release

r-validate:
	@echo "Running R reference validation..."
	@Rscript R/stats_lowess.R

compare:
	@echo "Comparing outputs..."
	@$(PYTHON) compare.py

plot:
	@echo "Generating plots..."
	@$(PYTHON) plot.py
