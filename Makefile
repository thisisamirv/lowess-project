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
FASTLOWESS_EXAMPLES := batch_smoothing online_smoothing streaming_smoothing

# Python bindings
PY_PKG := fastLowess-py
PY_DIR := bindings/python
PY_VENV := .venv
PY_TEST_DIR := tests/python

# R bindings
R_PKG_NAME := rfastlowess
R_PKG_VERSION := $(shell grep "^Version:" bindings/r/DESCRIPTION | sed 's/Version: //')
R_PKG_TARBALL := $(R_PKG_NAME)_$(R_PKG_VERSION).tar.gz
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
			cargo run -q -p $(FASTLOWESS_PKG) --example $$example --features $$feature || exit 1; \
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
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(PY_PKG) -- --check
	@ruff format --check $(PY_DIR)/fastlowess/ $(PY_TEST_DIR)/ || true
	@echo "=============================================================================="
	@echo "2. Linting..."
	@echo "=============================================================================="
	@cargo clippy -q -p $(PY_PKG) --all-targets -- -D warnings
	@ruff check $(PY_DIR)/fastlowess/ $(PY_TEST_DIR)/
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
	@cargo test -q -p $(PY_PKG)
	@. $(PY_VENV)/bin/activate && python -m pytest $(PY_TEST_DIR) -q
	@echo "=============================================================================="
	@echo "6. Documentation..."
	@echo "=============================================================================="
	@RUSTDOCFLAGS="-D warnings" cargo doc -q -p $(PY_PKG) --no-deps
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
	sed -i "s/^Version: .*/Version: $$WS_VERSION/" $(R_DIR)/DESCRIPTION; \
	rm -rf $(R_DIR)/*.Rcheck $(R_DIR)/*.BiocCheck $(R_DIR)/src/target $(R_DIR)/target $(R_DIR)/src/vendor; \
	(cd $(R_DIR) && Rscript -e "if (requireNamespace('codemetar', quietly=TRUE)) codemetar::write_codemeta()") || true; \
	echo "" >> $(R_DIR)/src/Cargo.toml
	@mkdir -p $(R_DIR)/src/.cargo && cp $(R_DIR)/src/cargo-config.toml $(R_DIR)/src/.cargo/config.toml
	@echo "Patched $(R_DIR)/src/Cargo.toml"
	@echo "=============================================================================="
	@echo "2. Installing R development packages..."
	@echo "=============================================================================="
	@Rscript -e "options(repos = c(CRAN = 'https://cloud.r-project.org')); suppressWarnings(install.packages(c('styler', 'prettycode', 'covr', 'codemetar', 'BiocManager', 'urlchecker', 'pkgdown'), quiet = TRUE))" || true
	@Rscript -e "suppressWarnings(BiocManager::install('BiocCheck', quiet = TRUE, update = FALSE, ask = FALSE))" || true
	@echo "R development packages installed!"
	@echo "=============================================================================="
	@echo "3. Vendoring..."
	@echo "=============================================================================="
	@echo "Updating and re-vendoring crates.io dependencies..."
	@# First, prepare the R package Cargo.toml for isolated vendoring
	@cd $(R_DIR) && sed -i 's|fastLowess = { path = "vendor/fastLowess",|fastLowess = { path = "vendor/fastLowess",|g' src/Cargo.toml
	@cd $(R_DIR) && sed -i '/\[patch.crates-io\]/,/lowess = { path = "vendor\/lowess" }/d' src/Cargo.toml
	@cd $(R_DIR) && perl -i -0pe 's/\s+$$/\n/' src/Cargo.toml
	@rm -rf $(R_DIR)/src/vendor $(R_DIR)/src/vendor.tar.xz
	@mkdir -p $(R_DIR)/src/vendor
	@# Step 1: Copy local crates FIRST
	@cp -rL crates/fastLowess $(R_DIR)/src/vendor/
	@cp -rL crates/lowess $(R_DIR)/src/vendor/
	@rm -rf $(R_DIR)/src/vendor/fastLowess/target $(R_DIR)/src/vendor/lowess/target
	@rm -f $(R_DIR)/src/vendor/fastLowess/Cargo.lock $(R_DIR)/src/vendor/lowess/Cargo.lock
	@rm -f $(R_DIR)/src/vendor/fastLowess/README.md $(R_DIR)/src/vendor/fastLowess/CHANGELOG.md
	@rm -f $(R_DIR)/src/vendor/lowess/README.md $(R_DIR)/src/vendor/lowess/CHANGELOG.md
	@# Step 2: Patch local crates to remove workspace inheritance BEFORE vendoring
	@EDITION=$$(grep 'edition = ' Cargo.toml | head -1 | sed 's/.*edition = "\([^"]*\)".*/\1/'); \
	VERSION=$$(grep 'version = ' Cargo.toml | head -1 | sed 's/.*version = "\([^"]*\)".*/\1/'); \
	AUTHORS=$$(grep 'authors = ' Cargo.toml | head -1 | sed 's/.*authors = \[\(.*\)\]/\1/'); \
	LICENSE=$$(grep 'license = ' Cargo.toml | head -1 | sed 's/.*license = "\([^"]*\)".*/\1/'); \
	RUST_VERSION=$$(grep 'rust-version = ' Cargo.toml | head -1 | sed 's/.*rust-version = "\([^"]*\)".*/\1/'); \
	V_NUM_TRAITS=$$(grep 'num-traits = ' Cargo.toml | head -1 | sed 's/.*version = "\([^"]*\)".*/\1/'); \
	V_NDARRAY=$$(grep 'ndarray = ' Cargo.toml | head -1 | sed 's/.*version = "\([^"]*\)".*/\1/'); \
	V_RAYON=$$(grep 'rayon = ' Cargo.toml | head -1 | sed 's/.*version = "\([^"]*\)".*/\1/'); \
	V_WIDE=$$(grep 'wide = ' Cargo.toml | head -1 | sed 's/.*version = "\([^"]*\)".*/\1/'); \
	V_APPROX=$$(grep 'approx = ' Cargo.toml | head -1 | sed 's/.*approx = "\([^"]*\)".*/\1/'); \
	V_WGPU=$$(grep 'wgpu = ' Cargo.toml | head -1 | sed 's/.*version = "\([^"]*\)".*/\1/'); \
	V_BYTEMUCK=$$(grep 'bytemuck = ' Cargo.toml | head -1 | sed 's/.*version = "\([^"]*\)".*/\1/'); \
	V_POLLSTER=$$(grep 'pollster = ' Cargo.toml | head -1 | sed 's/.*version = "\([^"]*\)".*/\1/'); \
	V_FUTURES=$$(grep 'futures-intrusive = ' Cargo.toml | head -1 | sed 's/.*version = "\([^"]*\)".*/\1/'); \
	for crate in fastLowess lowess; do \
		toml=$(R_DIR)/src/vendor/$$crate/Cargo.toml; \
		sed -i "s/^version = { workspace = true }/version = \"$$VERSION\"/" $$toml; \
		sed -i "s/^authors = { workspace = true }/authors = [$$AUTHORS]/" $$toml; \
		sed -i "s/^edition = { workspace = true }/edition = \"$$EDITION\"/" $$toml; \
		sed -i "s/^license = { workspace = true }/license = \"$$LICENSE\"/" $$toml; \
		sed -i "s/^rust-version = { workspace = true }/rust-version = \"$$RUST_VERSION\"/" $$toml; \
		sed -i '/^description = { workspace = true }/d' $$toml; \
		sed -i '/^readme = { workspace = true }/d' $$toml; \
		sed -i '/^repository = { workspace = true }/d' $$toml; \
		sed -i '/^homepage = { workspace = true }/d' $$toml; \
		sed -i '/^keywords = { workspace = true }/d' $$toml; \
		sed -i '/^categories = { workspace = true }/d' $$toml; \
		sed -i "s/^num-traits = { workspace = true/num-traits = { version = \"$$V_NUM_TRAITS\"/" $$toml; \
		sed -i "s/^ndarray = { workspace = true/ndarray = { version = \"$$V_NDARRAY\"/" $$toml; \
		sed -i "s/^rayon = { workspace = true/rayon = { version = \"$$V_RAYON\"/" $$toml; \
		sed -i "s/^wide = { workspace = true/wide = { version = \"$$V_WIDE\"/" $$toml; \
		sed -i "s/^approx = { workspace = true }/approx = \"$$V_APPROX\"/" $$toml; \
		sed -i "s/^wgpu = { workspace = true/wgpu = { version = \"$$V_WGPU\"/" $$toml; \
		sed -i "s/^bytemuck = { workspace = true/bytemuck = { version = \"$$V_BYTEMUCK\"/" $$toml; \
		sed -i "s/^pollster = { workspace = true/pollster = { version = \"$$V_POLLSTER\"/" $$toml; \
		sed -i "s/^futures-intrusive = { workspace = true/futures-intrusive = { version = \"$$V_FUTURES\"/" $$toml; \
	done; \
	sed -i 's/^lowess = { workspace = true/lowess = { path = "..\/lowess"/' $(R_DIR)/src/vendor/fastLowess/Cargo.toml
	@# Step 3: Remove GPU dependencies from fastLowess (not needed for R binding)
	@sed -i '/^wgpu = /d' $(R_DIR)/src/vendor/fastLowess/Cargo.toml
	@sed -i '/^bytemuck = /d' $(R_DIR)/src/vendor/fastLowess/Cargo.toml
	@sed -i '/^pollster = /d' $(R_DIR)/src/vendor/fastLowess/Cargo.toml
	@sed -i '/^futures-intrusive = /d' $(R_DIR)/src/vendor/fastLowess/Cargo.toml
	@sed -i 's/^gpu = .*/gpu = []/' $(R_DIR)/src/vendor/fastLowess/Cargo.toml
	@# Step 4: Create dummy checksum files for local crates
	@echo '{"files":{},"package":null}' > $(R_DIR)/src/vendor/lowess/.cargo-checksum.json
	@echo '{"files":{},"package":null}' > $(R_DIR)/src/vendor/fastLowess/.cargo-checksum.json
	@# Step 5: Add [workspace] and [patch.crates-io] to isolate from main workspace
	@sed -i '/^\[workspace\]/d' $(R_DIR)/src/Cargo.toml
	@echo "" >> $(R_DIR)/src/Cargo.toml
	@echo "[workspace]" >> $(R_DIR)/src/Cargo.toml
	@echo "" >> $(R_DIR)/src/Cargo.toml
	@echo "[patch.crates-io]" >> $(R_DIR)/src/Cargo.toml
	@echo "lowess = { path = \"vendor/lowess\" }" >> $(R_DIR)/src/Cargo.toml
	@# Step 6: Temporarily exclude R package from root workspace so cargo vendor is truly isolated
	@cp Cargo.toml Cargo.toml.vendor-backup
	@sed -i 's|"bindings/r/src",|# "bindings/r/src", # temporarily excluded for vendoring|' Cargo.toml
	@# Step 7: Now vendor crates.io dependencies with --no-delete to preserve local crates
	@(cd $(R_DIR)/src && cargo vendor -q --no-delete vendor)
	@# Step 8: Restore root Cargo.toml
	@mv Cargo.toml.vendor-backup Cargo.toml
	@# Step 9: Regenerate checksums after file cleanup
	@$(R_DIR)/scripts/clean_checksums.py -q $(R_DIR)/src/vendor
	@echo "Creating vendor.tar.xz archive..."
	@(cd $(R_DIR)/src && tar --sort=name --mtime='1970-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner --xz --create --file=vendor.tar.xz vendor)
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
	@cd $(R_DIR) && Rscript scripts/style_pkg.R || true
	@cd $(R_DIR)/src && cargo fmt -- --check || (echo "Run 'cargo fmt' to fix"; exit 1)
	@cd $(R_DIR)/src && cargo clippy -q -- -D warnings
	@cd $(R_DIR) && Rscript -e "lints <- lintr::lint_package(); print(lints); if (length(lints) > 0) quit(status = 1)"
	@echo "=============================================================================="
	@echo "7. Documentation..."
	@echo "=============================================================================="
	@rm -rf $(R_DIR)/*.Rcheck
	@cd $(R_DIR)/src && RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps
	@cd $(R_DIR) && Rscript -e "devtools::document(quiet = TRUE)"
	@cd $(R_DIR) && Rscript -e "devtools::build_vignettes(quiet = TRUE)" || true
	@cd $(R_DIR) && Rscript -e "if (file.exists('README.Rmd')) rmarkdown::render('README.Rmd', quiet = TRUE)" || true
	@cd $(R_DIR) && Rscript -e "if (requireNamespace('pkgdown', quietly=TRUE)) pkgdown::build_site(quiet = TRUE)" || true
	@echo "=============================================================================="
	@echo "8. Testing..."
	@echo "=============================================================================="
	@cd $(R_DIR)/src && cargo test -q
	@Rscript -e "Sys.setenv(NOT_CRAN='true'); testthat::test_dir('tests/r/testthat', package = 'rfastlowess')"
	@echo "=============================================================================="
	@echo "9. Submission checks..."
	@echo "=============================================================================="
	@cd $(R_DIR) && R_MAKEVARS_USER=$(PWD)/$(R_DIR)/scripts/Makevars.check R CMD check --as-cran $(R_PKG_TARBALL) || true
	@cd $(R_DIR) && Rscript -e "if (requireNamespace('urlchecker', quietly=TRUE)) urlchecker::url_check()" || true
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
# All targets
# ==============================================================================
all: lowess fastLowess python r
	@echo "All checks completed successfully!"

all-coverage: lowess-coverage fastLowess-coverage python-coverage r-coverage
	@echo "All coverage completed!"

all-clean: lowess-clean fastLowess-clean python-clean r-clean
	@echo "All clean completed!"

.PHONY: lowess lowess-coverage lowess-clean fastLowess fastLowess-coverage fastLowess-clean python python-coverage python-clean r r-coverage r-clean all all-coverage all-clean
