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

# Set WITH_GPU=1 to build rfastlowess with the GPU execution backend enabled
# (requires a Vulkan/Metal/DX12-capable GPU driver at runtime). Off by default
# so CRAN/Bioconductor submission builds never depend on wgpu/system GPU libs.
WITH_GPU ?= 0
R_CARGO_FEATURES :=
ifeq ($(WITH_GPU),1)
	R_CARGO_FEATURES := --features gpu
endif

ifeq ($(HOST_PLATFORM),windows)
AIR_INSTALL_CMD := if command -v pwsh >/dev/null 2>&1; then pwsh -ExecutionPolicy Bypass -c "irm https://github.com/posit-dev/air/releases/latest/download/air-installer.ps1 | iex"; else cmd //c 'powershell -ExecutionPolicy Bypass -c "irm https://github.com/posit-dev/air/releases/latest/download/air-installer.ps1 | iex"'; fi
else
AIR_INSTALL_CMD := curl -LsSf https://github.com/posit-dev/air/releases/latest/download/air-installer.sh | sh
endif

# jq (used to patch vendored .cargo-checksum.json files and generate
# inst/AUTHORS). Not preinstalled on plain Windows/macOS dev machines the
# way it is on GitHub-hosted CI runners, so fall back to downloading a
# static binary if it's not already on PATH.
JQ_DIR := $(R_DIR)/.bin
ifeq ($(HOST_PLATFORM),windows)
	JQ_LOCAL := $(JQ_DIR)/jq.exe
	JQ_ASSET := jq-windows-amd64.exe
else ifeq ($(HOST_PLATFORM),macos)
	JQ_LOCAL := $(JQ_DIR)/jq
	JQ_ASSET := jq-macos-amd64
else
	JQ_LOCAL := $(JQ_DIR)/jq
	JQ_ASSET := jq-linux-amd64
endif
JQ := $(shell command -v jq 2>/dev/null || echo $(JQ_LOCAL))
JQ_ENSURE_CMD := if ! command -v jq >/dev/null 2>&1 && [ ! -x "$(JQ_LOCAL)" ]; then \
	echo "jq not found on PATH — downloading a static binary..."; \
	mkdir -p $(JQ_DIR); \
	curl -fLsS -o "$(JQ_LOCAL)" "https://github.com/jqlang/jq/releases/latest/download/$(JQ_ASSET)"; \
	chmod +x "$(JQ_LOCAL)"; \
	fi

TEMP ?= /tmp
ifeq ($(OS),Windows_NT)
	TEMP := /tmp
endif

R_PKG_NAME := rfastlowess
R_PKG_VERSION = $(shell grep "^Version:" bindings/r/DESCRIPTION | sed 's/Version: //')
R_PKG_TARBALL = $(R_PKG_NAME)_$(R_PKG_VERSION).tar.gz
R_DIR := bindings/r
R_LIB_DIR := $(R_DIR)/.r-lib
R_CARGO_TARGET :=
R_CROSS_ENV :=
ifeq ($(OS),Windows_NT)
	R_CARGO_TARGET := --target x86_64-pc-windows-gnu
	R_CROSS_ENV    := PATH="/c/rtools45/x86_64-w64-mingw32.static.posix/bin:$$PATH"
endif

.PHONY: default dev clean coverage

default:
	@WITH_GPU=$(WITH_GPU) R CMD INSTALL $(R_DIR)

dev:
	@echo "Running $(R_PKG_NAME) checks..."
	@# Sync version from Cargo.toml to DESCRIPTION
	@VERSION=$$(grep "^version =" $(R_DIR)/src/Cargo.toml | head -n1 | sed 's/version = "\(.*\)"/\1/'); \
	sed -i.bak "s/^Version: .*/Version: $$VERSION/" $(R_DIR)/DESCRIPTION; \
	rm -f $(R_DIR)/DESCRIPTION.bak; \
	echo "Synced DESCRIPTION version to $$VERSION"
	@rm -rf $(R_DIR)/*.Rcheck $(R_DIR)/src/target $(R_DIR)/target $(R_DIR)/src/vendor
	@echo "=============================================================================="
	@echo "1. Installing R development packages..."
	@echo "=============================================================================="
	@echo "Installing R required packages (styler, testthat, rmarkdown, knitr, lintr, roxygen2, pkgdown, remotes)..."
	@echo "This may take a while depending on your internet connection and system performance."
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript \
	  -e "lib <- Sys.getenv('R_LIBS_USER'); dir.create(lib, recursive = TRUE, showWarnings = FALSE); .libPaths(c(lib, .libPaths()))" \
	  -e "# Bootstrap pak; it handles RSPM binary vs source selection automatically" \
	  -e "if (!requireNamespace('pak', quietly = TRUE, lib.loc = lib)) { options(install.lock = FALSE); install.packages('pak', lib = lib, repos = 'https://r-lib.github.io/p/pak/stable/') }" \
	  -e "# Required packages (CRAN)" \
	  -e "pak::pkg_install(c('styler', 'testthat', 'rmarkdown', 'knitr', 'lintr', 'roxygen2', 'pkgdown', 'remotes'), lib = lib, ask = FALSE)" \
	  -e "# srr lives on its own R-universe; optional so a dead mirror doesn't break the build" \
	  -e "options(repos = c(ropenscireviewtools = 'https://ropensci-review-tools.r-universe.dev', CRAN = 'https://cloud.r-project.org')); try(pak::pkg_install('srr', lib = lib, ask = FALSE), silent = TRUE)" \
	  -e "# Optional: rOpenSci tooling and local dev helpers" \
	  -e "options(repos = c(ropensci = 'https://ropensci.r-universe.dev', CRAN = 'https://cloud.r-project.org')); try(pak::pkg_install(c('pkgcheck', 'pkgstats', 'covr', 'prettycode', 'toml', 'V8', 'visNetwork'), lib = lib, ask = FALSE), silent = TRUE)" \
	  >/dev/null
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "if (!rmarkdown::pandoc_available()) { message('\nERROR: Pandoc is required to build R Markdown vignettes but is not available.\nPlease install Pandoc (https://pandoc.org/installing.html) and ensure it is in your PATH.\n'); quit(status = 1) }"
	@echo "R development packages installed!"
	@echo "=============================================================================="
	@echo "2. Vendoring..."
	@echo "=============================================================================="
	@(cd $(R_DIR)/src && WITH_GPU=$(WITH_GPU) sh vendor-update.sh)
	@echo "=============================================================================="
	@echo "3. Building..."
	@echo "=============================================================================="
	@mkdir -p $(R_DIR)/src/.cargo && cp $(R_DIR)/src/cargo-config.toml $(R_DIR)/src/.cargo/config.toml
	@(cd $(R_DIR)/src && $(R_CROSS_ENV) cargo build -q --release $(R_CARGO_TARGET) $(R_CARGO_FEATURES)) || \
	    { rm -rf $(R_DIR)/src/.cargo; exit 1; }
	@rm -rf $(R_DIR)/src/.cargo
	@echo "=============================================================================="
	@echo "4. Formatting..."
	@echo "=============================================================================="
	@if ! command -v air >/dev/null 2>&1; then \
	    echo "air not found — installing..."; \
	    $(AIR_INSTALL_CMD); \
	    export PATH="$$HOME/.local/bin:$$PATH"; \
	fi; \
	air format $(R_DIR)/R/ $(R_DIR)/tests/testthat/ $(R_DIR)/vignettes/ || true
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript \
	  -e "my_linters <- lintr::linters_with_defaults(indentation_linter = lintr::indentation_linter(indent = 4L), object_name_linter = NULL, commented_code_linter = NULL, object_usage_linter = NULL)" \
	  -e "lints <- c(lintr::lint_dir('$(R_DIR)/R', linters = my_linters), lintr::lint_dir('$(R_DIR)/tests/testthat', linters = my_linters))" \
	  -e "print(lints)" \
	  -e "if (length(lints) > 0L) quit(status = 1)"
	@echo "=============================================================================="
	@echo "5. Documentation..."
	@echo "=============================================================================="
	@cd $(R_DIR) && R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript \
	  -e "roxygen2::roxygenise(package.dir = '.', roclets = c('namespace', 'rd', 'srr::srr_stats_roclet'))"
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript $(CURDIR)/dev/fix_rd_style.R $(R_DIR)/man
	@$(PYTHON) dev/update_changelogs.py r
	@echo "=============================================================================="
	@echo "6. Packaging..."
	@echo "=============================================================================="
	@echo "Generating $(R_DIR)/inst/AUTHORS..."
	@mkdir -p $(R_DIR)/inst
	@$(JQ_ENSURE_CMD)
	@{ \
		echo "Authors and Copyright Holders for Rust Dependencies:"; \
		echo ""; \
		_f='. as $$root | [.packages[] | select(.id as $$id | ($$root.workspace_members | index($$id)) == null)] | unique_by(.name)[]'; \
		_f="$$_f"' | "Package: \(.name) (\(.version))\nAuthors: \(.authors | join(", "))\nLicense: \(.license // "Unknown")\n" + "-" * 40'; \
		(cargo metadata --format-version 1) | $(JQ) -r "$$_f"; \
		} > $(R_DIR)/inst/AUTHORS
	@cd $(R_DIR) && WITH_GPU=$(WITH_GPU) R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) R CMD build .
	@echo "=============================================================================="
	@echo "7. Installing..."
	@echo "=============================================================================="
	@rm -rf $(R_DIR)/00LOCK-$(R_PKG_NAME)
	@mkdir -p $(R_LIB_DIR)
	@echo "Installing R runtime dependencies (testthat)..."
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "suppressMessages({ lib <- Sys.getenv('R_LIBS_USER'); options(repos = c(CRAN = 'https://cloud.r-project.org')); if (!requireNamespace('testthat', quietly = TRUE, lib.loc = lib)) install.packages('testthat', lib = lib, quiet = TRUE) })" >/dev/null
	@cd $(R_DIR) && WITH_GPU=$(WITH_GPU) R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) R CMD INSTALL -l .r-lib $(R_PKG_TARBALL)
	@echo "=============================================================================="
	@echo "8. Testing..."
	@echo "=============================================================================="
	@if [ ! -d $(R_DIR)/src/vendor ]; then (cd $(R_DIR)/src && tar -xf vendor.tar.xz); fi
	@mkdir -p $(R_DIR)/src/.cargo && cp $(R_DIR)/src/cargo-config.toml $(R_DIR)/src/.cargo/config.toml
	@if [ -n "$(R_CARGO_TARGET)" ]; then \
		GCC_LIBDIR="c:/rtools45/x86_64-w64-mingw32.static.posix/lib/gcc/x86_64-w64-mingw32.static.posix/14.3.0"; \
		test -f "$$GCC_LIBDIR/libgcc_eh.a" || x86_64-w64-mingw32.static.posix-ar rcs "$$GCC_LIBDIR/libgcc_eh.a"; \
	fi
	@cd $(R_DIR)/src && $(R_CROSS_ENV) cargo test -q --no-run $(R_CARGO_TARGET)
	@rm -rf $(R_DIR)/src/.cargo
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "Sys.setenv(NOT_CRAN='true'); testthat::test_dir('$(R_DIR)/tests/testthat', package = 'rfastlowess')"
	@echo "=============================================================================="
	@echo "9. Submission checks..."
	@echo "=============================================================================="
	@cd $(R_DIR) && R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) R_MAKEVARS_USER=$(CURDIR)/dev/Makevars.check R CMD check --as-cran --no-manual $(R_PKG_TARBALL) || true
	@echo "Package size (Limit: 5MB):"
	@ls -lh $(R_DIR)/$(R_PKG_TARBALL) || true
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript \
	  -e "library(pkgstats); library(pkgcheck)" \
	  -e "token <- Sys.getenv('GITHUB_TOKEN', ''); if (nzchar(token) && !nzchar(Sys.getenv('GITHUB_PAT', ''))) Sys.setenv(GITHUB_PAT = token)" \
	  -e "tryCatch(pkgcheck(use_cache = FALSE), error = function(err) { msg <- conditionMessage(err); if (grepl('GitHub API error|rate limit exceeded|timeout|could not resolve host', msg, ignore.case = TRUE) || grepl('could not find function', msg, fixed = TRUE)) { message('Skipping pkgcheck due to known incompatibility or network failure: ', msg); return(invisible(NULL)) }; stop(err) })"
	@echo "=============================================================================="
	@echo "Doc snippet tests (r)..."
	@$(PYTHON) dev/verify_snippets.py --lang r --timeout 120
	@echo "All $(R_PKG_NAME) checks completed successfully!"

coverage:
	@echo "Calculating $(R_PKG_NAME) coverage..."
	@cd $(R_DIR) && NOT_CRAN=true Rscript -e "\
	  if (!requireNamespace('covr', quietly = TRUE)) { message('covr missing'); quit(status=0) }; \
	  cov <- covr::package_coverage(); \
	  covr::zero_coverage(cov); \
	  print(cov)"

clean:
	@echo "Cleaning $(R_PKG_NAME)..."
	@if [ -d $(R_DIR)/src/target ]; then \
		rm -rf $(R_DIR)/src/target 2>/dev/null || \
		(command -v docker >/dev/null && docker run --rm -v "$(PWD)/$(R_DIR)":/pkg ghcr.io/r-universe-org/build-wasm:latest rm -rf /pkg/src/target) || \
		echo "Warning: Failed to clean src/target"; \
	fi
	@(cd $(R_DIR)/src && cargo clean 2>/dev/null || true)
	@Rscript -e "try(remove.packages('$(R_PKG_NAME)'), silent = TRUE)" || true
	@git clean -fdX -- $(R_DIR)
	@echo "$(R_PKG_NAME) clean complete!"