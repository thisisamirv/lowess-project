# ==============================================================================
# Configuration
# ==============================================================================
FEATURE_SET ?= all

# Make shell commands fail on error
.SHELLFLAGS := -ec

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

# R bindings
R_PKG_NAME := rfastlowess
R_PKG_VERSION = $(shell grep "^Version:" bindings/r/DESCRIPTION | sed 's/Version: //')
R_PKG_TARBALL = $(R_PKG_NAME)_$(R_PKG_VERSION).tar.gz
R_DIR := bindings/r

# ==============================================================================
# lowess crate
# ==============================================================================
lowess:
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
	@echo "4. Examples..."
	@echo "=============================================================================="
	@for example in $(LOWESS_EXAMPLES); do \
		echo "Running example: $$example"; \
		cargo run -q -p $(LOWESS_PKG) --example $$example --features dev || exit 1; \
	done
	@echo "=============================================================================="
	@echo "All $(LOWESS_PKG) crate checks completed successfully!"

lowess-coverage:
	@echo "Running $(LOWESS_PKG) coverage..."
	@cd $(LOWESS_DIR) && LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov --all-targets --all-features

lowess-clean:
	@echo "Cleaning $(LOWESS_PKG) crate..."
	@cargo clean -p $(LOWESS_PKG)
	@rm -rf $(LOWESS_DIR)/coverage_html
	@rm -rf $(LOWESS_DIR)/benchmarks
	@rm -rf $(LOWESS_DIR)/validation
	@echo "$(LOWESS_PKG) clean complete!"

# ==============================================================================
# fastLowess crate
# ==============================================================================
fastLowess:
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
			cargo test -q -p lowess-project-tests --test $(FASTLOWESS_PKG) --features $$feature -- --test-threads=1 || exit 1; \
		else \
			cargo test -q -p lowess-project-tests --test $(FASTLOWESS_PKG) --features $$feature || exit 1; \
		fi; \
	done
	@echo "=============================================================================="
	@echo "4. Examples..."
	@echo "=============================================================================="
	@for feature in $(FASTLOWESS_FEATURES); do \
		echo "Running examples with feature: $$feature"; \
		for example in $(FASTLOWESS_EXAMPLES); do \
			if [ "$$feature" = "dev" ]; then \
				cargo run -q -p $(FASTLOWESS_PKG) --example $$example --features $$feature || exit 1; \
			else \
				cargo run -q -p $(FASTLOWESS_PKG) --example $$example --features $$feature > /dev/null || exit 1; \
			fi; \
		done; \
	done
	@echo "=============================================================================="
	@echo "All $(FASTLOWESS_PKG) crate checks completed successfully!"

fastLowess-coverage:
	@echo "Running $(FASTLOWESS_PKG) coverage..."
	@cd $(FASTLOWESS_DIR) && LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov --all-targets --all-features

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
	@echo "Running $(PY_PKG) checks..."
	@echo "=============================================================================="
	@echo "0. Version Sync..."
	@echo "=============================================================================="
	@dev/sync_version.py Cargo.toml $(R_DIR)/inst/CITATION -p $(PY_DIR)/python/fastlowess/__version__.py -c CITATION.cff -q
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(PY_PKG) -- --check
	@ruff format $(PY_DIR)/python/ $(PY_TEST_DIR)/
	@echo "=============================================================================="
	@echo "2. Linting..."
	@echo "=============================================================================="
	@VIRTUAL_ENV= PYO3_PYTHON=python3 cargo clippy -q -p $(PY_PKG) --all-targets -- -D warnings
	@ruff check $(PY_DIR)/python/ $(PY_TEST_DIR)/
	@echo "=============================================================================="
	@echo "3. Environment Setup..."
	@echo "=============================================================================="
	@if [ ! -d "$(PY_VENV)" ]; then python3 -m venv $(PY_VENV); fi
	@. $(PY_VENV)/bin/activate && pip install pytest numpy maturin
	@echo "=============================================================================="
	@echo "4. Building..."
	@echo "=============================================================================="
	@. $(PY_VENV)/bin/activate && cd $(PY_DIR) && maturin develop -q
	@echo "=============================================================================="
	@echo "5. Testing..."
	@echo "=============================================================================="
	@VIRTUAL_ENV= PYO3_PYTHON=python3 cargo test -q -p $(PY_PKG)
	@. $(PY_VENV)/bin/activate && python -m pytest $(PY_TEST_DIR) -q
	@echo "=============================================================================="
	@echo "6. Examples..."
	@echo "=============================================================================="
	@. $(PY_VENV)/bin/activate && pip install -q matplotlib
	@. $(PY_VENV)/bin/activate && python $(PY_DIR)/examples/batch_smoothing.py
	@. $(PY_VENV)/bin/activate && python $(PY_DIR)/examples/streaming_smoothing.py
	@. $(PY_VENV)/bin/activate && python $(PY_DIR)/examples/online_smoothing.py
	@echo "=============================================================================="
	@echo "$(PY_PKG) checks completed successfully!"

python-coverage:
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
	@rm -rf $(PY_DIR)/fastlowess/__pycache__
	@rm -rf $(PY_TEST_DIR)/__pycache__
	@rm -rf $(PY_DIR)/*.egg-info
	@rm -rf $(PY_DIR)/.ruff_cache
	@rm -rf $(PY_DIR)/*.so
	@echo "$(PY_PKG) clean complete!"

# ==============================================================================
# R bindings
# ==============================================================================
r:
	@echo "Running $(R_PKG_NAME) checks..."
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
	@WS_EDITION=$$(grep 'edition = ' Cargo.toml | head -1 | sed 's/.*edition = "\([^"]*\)".*/\1/'); \
	WS_VERSION=$$(grep 'version = ' Cargo.toml | head -1 | sed 's/.*version = "\([^"]*\)".*/\1/'); \
	WS_AUTHORS=$$(grep 'authors = ' Cargo.toml | head -1 | sed 's/.*authors = \[\(.*\)\]/\1/'); \
	WS_LICENSE=$$(grep 'license = ' Cargo.toml | head -1 | sed 's/.*license = "\([^"]*\)".*/\1/'); \
	WS_RUST_VERSION=$$(grep 'rust-version = ' Cargo.toml | head -1 | sed 's/.*rust-version = "\([^"]*\)".*/\1/'); \
	WS_EXTENDR=$$(grep 'extendr-api = ' Cargo.toml | head -1 | sed 's/.*extendr-api = "\([^"]*\)".*/\1/'); \
	sed -i "s/^version = \".*\"/version = \"$$WS_VERSION\"/" $(R_DIR)/src/Cargo.toml; \
	sed -i "s/^edition = \".*\"/edition = \"$$WS_EDITION\"/" $(R_DIR)/src/Cargo.toml; \
	sed -i "s/^authors = \\[.*\\]/authors = [$$WS_AUTHORS]/" $(R_DIR)/src/Cargo.toml; \
	sed -i "s/^license = \".*\"/license = \"$$WS_LICENSE\"/" $(R_DIR)/src/Cargo.toml; \
	sed -i "s/^rust-version = \".*\"/rust-version = \"$$WS_RUST_VERSION\"/" $(R_DIR)/src/Cargo.toml; \
	sed -i "s/^extendr-api = \".*\"/extendr-api = \"$$WS_EXTENDR\"/" $(R_DIR)/src/Cargo.toml; \
	sed -i '/^\[workspace\]/d' $(R_DIR)/src/Cargo.toml; \
	sed -i '/^\[patch\.crates-io\]/d' $(R_DIR)/src/Cargo.toml; \
	sed -i '/^lowess = { path = "vendor\/lowess" }/d' $(R_DIR)/src/Cargo.toml; \
	rm -rf $(R_DIR)/*.Rcheck $(R_DIR)/*.BiocCheck $(R_DIR)/src/target $(R_DIR)/target $(R_DIR)/src/vendor; \
	echo "" >> $(R_DIR)/src/Cargo.toml
	@dev/sync_version.py Cargo.toml $(R_DIR)/inst/CITATION -d $(R_DIR)/DESCRIPTION -c CITATION.cff -q
	@mkdir -p $(R_DIR)/src/.cargo && cp $(R_DIR)/src/cargo-config.toml $(R_DIR)/src/.cargo/config.toml
	@echo "Patched $(R_DIR)/src/Cargo.toml"
	@echo "=============================================================================="
	@echo "2. Installing R development packages..."
	@echo "=============================================================================="
	@Rscript -e "options(repos = c(CRAN = 'https://cloud.r-project.org')); suppressWarnings(install.packages(c('styler', 'prettycode', 'covr', 'BiocManager', 'urlchecker', 'toml', 'V8'), quiet = TRUE))" || true
	@Rscript -e "suppressWarnings(BiocManager::install('BiocCheck', quiet = TRUE, update = FALSE, ask = FALSE))" || true
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
	@(cd $(R_DIR)/src && tar --sort=name --mtime='1970-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner --xz --create --file=vendor.tar.xz vendor Cargo.lock)
	@rm -rf $(R_DIR)/src/vendor
	@echo "Vendor update complete. Archive: $(R_DIR)/src/vendor.tar.xz"
	@if [ -f $(R_DIR)/src/vendor.tar.xz ] && [ ! -d $(R_DIR)/src/vendor ]; then \
		(cd $(R_DIR)/src && tar --extract --xz -f vendor.tar.xz) && \
		find $(R_DIR)/src/vendor -name "CITATION.cff" -delete && \
		find $(R_DIR)/src/vendor -name "CITATION" -delete; \
	fi
	@echo "=============================================================================="
	@echo "4. Building..."
	@echo "=============================================================================="
	@(cd $(R_DIR)/src && cargo build -q --release || (mv Cargo.toml.orig Cargo.toml && exit 1))
	@rm -rf $(R_DIR)/src/.cargo
	@cd $(R_DIR) && R CMD build .
	@echo "=============================================================================="
	@echo "5. Installing..."
	@echo "=============================================================================="
	@cd $(R_DIR) && R CMD INSTALL $(R_PKG_TARBALL) || true
	@cd $(R_DIR) && Rscript -e "devtools::install(quiet = TRUE)"
	@echo "=============================================================================="
	@echo "6. Formatting..."
	@echo "=============================================================================="
	@cd $(R_DIR)/src && cargo fmt -q
	@cd $(R_DIR) && Rscript $(PWD)/dev/style_pkg.R || true
	@cd $(R_DIR)/src && cargo fmt -- --check || (echo "Run 'cargo fmt' to fix"; exit 1)
	@cd $(R_DIR)/src && cargo clippy -q -- -D warnings
	@Rscript -e "my_linters <- lintr::linters_with_defaults(indentation_linter = lintr::indentation_linter(indent = 4L)); lints <- c(lintr::lint_dir('$(R_DIR)/R', linters = my_linters), lintr::lint_dir('tests/r/testthat', linters = my_linters)); print(lints); if (length(lints) > 0) quit(status = 1)"
	@echo "=============================================================================="
	@echo "7. Documentation..."
	@echo "=============================================================================="
	@rm -rf $(R_DIR)/*.Rcheck
	@cd $(R_DIR)/src && RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps
	@cd $(R_DIR) && Rscript -e "devtools::document(quiet = TRUE)"
	@cd $(R_DIR) && Rscript -e "devtools::build_vignettes(quiet = TRUE)" || true
	@rm -f $(R_DIR)/.gitignore
	@echo "=============================================================================="
	@echo "8. Testing..."
	@echo "=============================================================================="
	@cd $(R_DIR)/src && cargo test -q
	@Rscript -e "Sys.setenv(NOT_CRAN='true'); testthat::test_dir('tests/r/testthat', package = 'rfastlowess')"
	@echo "=============================================================================="
	@echo "9. Submission checks..."
	@echo "=============================================================================="
	@cd $(R_DIR) && R_MAKEVARS_USER=$(PWD)/dev/Makevars.check R CMD check --as-cran $(R_PKG_TARBALL) || true
	@cd $(R_DIR) && Rscript -e "if (requireNamespace('urlchecker', quietly=TRUE)) urlchecker::url_check(skip = c('https://crates.io'))" || true
	@cd $(R_DIR) && Rscript -e "if (requireNamespace('BiocCheck', quietly=TRUE)) BiocCheck::BiocCheck('$(R_PKG_TARBALL)')" || true
	@echo "Package size (Limit: 5MB):"
	@ls -lh $(R_DIR)/$(R_PKG_TARBALL) || true
	@if [ -f $(R_DIR)/src/Cargo.toml.orig ]; then mv $(R_DIR)/src/Cargo.toml.orig $(R_DIR)/src/Cargo.toml; fi
	@echo "=============================================================================="
	@echo "All $(R_PKG_NAME) checks completed successfully!"

r-coverage:
	@echo "Calculating $(R_PKG_NAME) coverage..."
	@cd $(R_DIR) && Rscript -e "if (!requireNamespace('covr', quietly = TRUE)) { message('covr missing'); quit(status=0) } else { Sys.setenv(NOT_CRAN='true'); covr::package_coverage() }"

r-clean:
	@echo "Cleaning $(R_PKG_NAME)..."
	@if [ -d $(R_DIR)/src/target ]; then \
		rm -rf $(R_DIR)/src/target 2>/dev/null || \
		(command -v docker >/dev/null && docker run --rm -v "$(PWD)/$(R_DIR)":/pkg ghcr.io/r-universe-org/build-wasm:latest rm -rf /pkg/src/target) || \
		echo "Warning: Failed to clean src/target"; \
	fi
	@(cd $(R_DIR)/src && cargo clean 2>/dev/null || true)
	@rm -rf $(R_DIR)/src/vendor $(R_DIR)/target
	@rm -rf $(R_DIR)/$(R_PKG_NAME).Rcheck $(R_DIR)/$(R_PKG_NAME).BiocCheck
	@rm -f $(R_DIR)/$(R_PKG_NAME)_*.tar.gz
	@rm -rf $(R_DIR)/src/*.o $(R_DIR)/src/*.so $(R_DIR)/src/*.dll $(R_DIR)/src/Cargo.toml.orig
	@rm -rf $(R_DIR)/doc $(R_DIR)/Meta $(R_DIR)/vignettes/*.html $(R_DIR)/README.html
	@find $(R_DIR) -name "*.Rout" -delete
	@Rscript -e "try(remove.packages('$(R_PKG_NAME)'), silent = TRUE)" || true
	@rm -rf $(R_DIR)/src/Makevars $(R_DIR)/rfastlowess*.tgz
	@rm -rf $(R_DIR)/benchmarks $(R_DIR)/validation $(R_DIR)/docs
	@echo "$(R_PKG_NAME) clean complete!"


# ==============================================================================
# Development checks
# ==============================================================================
check-msrv:
	@echo "Checking MSRV..."
	@python3 dev/check_msrv.py

# ==============================================================================
# Documentation
# ==============================================================================
DOCS_VENV := docs-venv

docs:
	@echo "Building documentation..."
	@if [ ! -d "$(DOCS_VENV)" ]; then python3 -m venv $(DOCS_VENV); fi
	@. $(DOCS_VENV)/bin/activate && pip install -q -r docs/requirements.txt && mkdocs build

docs-serve:
	@echo "Starting documentation server..."
	@if [ ! -d "$(DOCS_VENV)" ]; then python3 -m venv $(DOCS_VENV); fi
	@. $(DOCS_VENV)/bin/activate && pip install -q -r docs/requirements.txt && mkdocs serve

docs-clean:
	@echo "Cleaning documentation build..."
	@rm -rf site/ $(DOCS_VENV)/
	@echo "Documentation clean complete!"

# ==============================================================================
# All targets
# ==============================================================================
all: lowess fastLowess python r check-msrv
	@echo "All checks completed successfully!"

all-coverage: lowess-coverage fastLowess-coverage python-coverage r-coverage
	@echo "All coverage completed!"

all-clean: r-clean lowess-clean fastLowess-clean python-clean
	@echo "Cleaning project root..."
	@cargo clean
	@rm -rf target Cargo.lock .venv .ruff_cache .pytest_cache site docs-venv
	@echo "All clean completed!"

.PHONY: lowess lowess-coverage lowess-clean fastLowess fastLowess-coverage fastLowess-clean python python-coverage python-clean r r-coverage r-clean check-msrv docs docs-serve docs-clean all all-coverage all-clean
