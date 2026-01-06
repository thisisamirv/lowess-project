# Configuration
PKG_NAME := rfastlowess
PKG_VERSION := $(shell grep "^Version:" DESCRIPTION | sed 's/Version: //')
PKG_TARBALL := $(PKG_NAME)_$(PKG_VERSION).tar.gz

# Standard dev check
check: vendor build install fmt doc test submission

.PHONY: check install-dev-packages vendor vendor-update extract-vendor build build-rust build-r install install-r install-dev fmt fmt-fix fmt-check clippy style-r lint-r doc doc-rust doc-r vignettes codemeta readme test test-rust test-r test-cran-emulation submission check-cran bioccheck size wasm-build clean clean-wasm clean-rust clean-r clean-other coverage urlcheck

# Install dev packages
install-dev-packages:
	@echo "Installing development packages..."
	@Rscript -e "options(repos = c(CRAN = 'https://cloud.r-project.org')); install.packages(c('styler', 'prettycode', 'covr', 'codemetar', 'BiocManager', 'urlchecker', 'pkgdown'), quiet = TRUE); BiocManager::install('BiocCheck', quiet = TRUE)"

# ============================================================================== #
# Vendor
# ============================================================================== #
vendor: vendor-update extract-vendor

vendor-update:
	@echo "Updating and re-vendoring crate.io dependencies..."
	@rm -rf src/vendor src/vendor.tar.xz
	@# Temporarily isolate to allow cargo vendor to resolve from crates.io
	@cp src/Cargo.toml src/Cargo.toml.orig
	@echo '[workspace]' >> src/Cargo.toml
	@(cd src && cargo vendor vendor || (mv Cargo.toml.orig Cargo.toml && exit 1))
	@mv src/Cargo.toml.orig src/Cargo.toml
	@./scripts/clean_checksums.py src/vendor
	@echo "Creating vendor.tar.xz archive..."
	@(cd src && tar --sort=name --mtime='1970-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner --xz --create --file=vendor.tar.xz vendor)
	@rm -rf src/vendor
	@echo "Vendor update complete. Archive: src/vendor.tar.xz"

extract-vendor:
	@if [ -f src/vendor.tar.xz ] && [ ! -d src/vendor ]; then \
		echo "Extracting vendored dependencies..."; \
		(cd src && tar --extract --xz -f vendor.tar.xz); \
	fi

# ==============================================================================
# Build
# ==============================================================================
build: build-rust build-r

build-rust:
	@echo "Building Rust crate..."
	@cp src/Cargo.toml src/Cargo.toml.orig
	@echo '[workspace]' >> src/Cargo.toml
	@# Ensure .cargo/config.toml exists if vendor exists (it should from extract-vendor)
	@mkdir -p src/.cargo && cp src/cargo-config.toml src/.cargo/config.toml
	@(cd src && cargo build --release || (mv Cargo.toml.orig Cargo.toml && exit 1))
	@mv src/Cargo.toml.orig src/Cargo.toml
	@rm -rf src/.cargo

build-r: doc-r
	@echo "Building R package tarball..."
	R CMD build .

# ==============================================================================
# Install
# ==============================================================================
install: install-r install-dev

install-r:
	@echo "Installing R package..."
	R CMD INSTALL $(PKG_TARBALL)

install-dev:
	@echo "Installing R package (devtools mode)..."
	Rscript -e "devtools::install()"

# ==============================================================================
# Formatting
# ==============================================================================
fmt: fmt-fix style-r fmt-check clippy lint-r

fmt-fix:
	@(cd src && cargo fmt --all)

fmt-check:
	@echo "Rust formatting..."
	@(cd src && cargo fmt --all -- --check || (echo "Run 'make fmt-fix' to fix"; exit 1))

clippy:
	@echo "Rust clippy..."
	@(cd src && cargo clippy --all-targets -- -D warnings)

style-r:
	@echo "R styling..."
	@Rscript -e "if (!requireNamespace('styler', quietly=TRUE) || !requireNamespace('prettycode', quietly=TRUE)) { message('Packages missing. Run make install-dev-packages'); quit(status=0) }"
	@./scripts/style_pkg.R

lint-r:
	@echo "R linting..."
	@Rscript -e "lintr::lint_package()" || true

# ==============================================================================
# Documentation
# ==============================================================================
doc: doc-rust doc-r vignettes codemeta readme site

doc-rust:
	@echo "Generating Rust docs..."
	@(cd src && RUSTDOCFLAGS="-D warnings" cargo doc --no-deps)

doc-r:
	@echo "Generating R docs (roxygen2)..."
	Rscript -e "devtools::document()"

vignettes:
	@echo "Building vignettes..."
	Rscript -e "devtools::build_vignettes()"

codemeta:
	@echo "Generating codemeta.json..."
	@Rscript -e "if (!requireNamespace('codemetar', quietly = TRUE)) { message('codemetar missing. Run make install-dev-packages'); quit(status=0) } else { codemetar::write_codemeta() }"

readme:
	@echo "Generating README.md..."
	@Rscript -e "rmarkdown::render('README.Rmd')"

site:
	@echo "Building pkgdown site..."
	@Rscript -e "if (!requireNamespace('pkgdown', quietly = TRUE)) { message('pkgdown missing. Run make install-dev-packages'); quit(status=0) } else { pkgdown::build_site() }"

# ==============================================================================
# Testing
# ==============================================================================
test: test-rust test-r test-cran-emulation

test-rust:
	@echo "Running Rust tests..."
	@(cd src && cargo test)

test-r:
	@echo "Running R tests (devtools)..."
	Rscript -e "Sys.setenv(NOT_CRAN='true'); devtools::test()"

test-cran-emulation:
	@echo "Running R tests as CRAN would (skipping protected)."
	Rscript -e "devtools::test()"

# ==============================================================================
# Submission Checks
# ==============================================================================
submission: check-cran bioccheck size wasm-build urlcheck

check-cran: build-r
	@echo "Running R CMD check --as-cran..."
	R_MAKEVARS_USER=$(PWD)/scripts/Makevars.check R CMD check --as-cran $(PKG_TARBALL)

urlcheck:
	@echo "Checking URLs..."
	@Rscript -e "if (!requireNamespace('urlchecker', quietly = TRUE)) { message('urlchecker missing. Run make install-dev-packages'); quit(status=0) } else { urlchecker::url_check() }"

bioccheck: build-r
	@echo "Running BiocCheck..."
	@Rscript -e "if (!requireNamespace('BiocCheck', quietly = TRUE)) { message('BiocCheck missing. Run make install-dev-packages'); quit(status=0) } else { BiocCheck::BiocCheck('$(PKG_TARBALL)') }"

size: build-r
	@echo "Package size (Limit: 5MB):"
	@ls -lh $(PKG_TARBALL)

wasm-build:
	@echo "Building WASM binary using R-universe Docker container..."
	@if ! command -v docker &> /dev/null; then \
		echo "Error: Docker is required. Install with: sudo pacman -S docker"; \
		exit 1; \
	fi
	@if ! docker image inspect ghcr.io/r-universe-org/build-wasm:latest &> /dev/null; then \
		echo "Pulling R-universe Docker image (this may take a while)..."; \
		docker pull ghcr.io/r-universe-org/build-wasm:latest; \
	fi
	@docker run --rm \
		-v "$(PWD)":/pkg \
		-w /pkg \
		ghcr.io/r-universe-org/build-wasm:latest \
		bash -c "R -e \"rwasm::build('./$(PKG_TARBALL)')\" >/dev/null 2>&1 && echo 'WASM build successful!'"

# ==============================================================================
# Other
# ==============================================================================
clean: clean-wasm clean-rust clean-r clean-other

clean-wasm:
	@if [ -d src/target ]; then \
		if rm -rf src/target 2>/dev/null; then \
			true; \
		elif command -v docker >/dev/null; then \
			docker run --rm -v "$(PWD)":/pkg ghcr.io/r-universe-org/build-wasm:latest rm -rf /pkg/src/target; \
		else \
			echo "Warning: Failed to clean src/target (permissions) and Docker not found."; \
		fi \
	fi

clean-rust:
	@(cd src && cargo clean 2>/dev/null || true)
	@rm -rf src/vendor
	@rm -rf target

clean-r:
	@rm -rf $(PKG_NAME).Rcheck
	@rm -rf $(PKG_NAME).BiocCheck
	@rm -f $(PKG_NAME)_*.tar.gz
	@rm -rf src/*.o src/*.so src/*.dll
	@rm -rf doc Meta vignettes/*.html README.html
	@find . -name "*.Rout" -delete
	@Rscript -e "try(remove.packages('$(PKG_NAME)'), silent = TRUE)"

clean-other:
	@rm -rf src/Makevars
	@rm -rf src/target
	@rm -rf rfastlowess*.tgz
	@rm -rf benchmarks
	@rm -rf validation
	@rm -rf docs
	

coverage:
	@echo "Calculating R coverage..."
	@Rscript -e "if (!requireNamespace('covr', quietly = TRUE)) { message('covr missing. Run make install-dev-packages'); quit(status=0) } else { Sys.setenv(NOT_CRAN='true'); covr::package_coverage() }"
