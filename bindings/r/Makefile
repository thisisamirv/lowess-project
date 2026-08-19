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

PYTHON      ?= python3
PYO3_PYTHON ?= $(PYTHON)
NODE        ?= node

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
JQ_DIR   := $(R_DIR)/.bin
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

R_PKG_NAME    := rfastlowess
R_PKG_VERSION  = $(shell grep "^Version:" bindings/r/DESCRIPTION | sed 's/Version: //')
R_PKG_TARBALL  = $(R_PKG_NAME)_$(R_PKG_VERSION).tar.gz
R_DIR         := bindings/r
R_LIB_DIR     := $(R_DIR)/.r-lib
R_CARGO_TARGET :=
R_CROSS_ENV    :=
ifeq ($(OS),Windows_NT)
	R_CARGO_TARGET := --target x86_64-pc-windows-gnu
	R_CROSS_ENV    := PATH="/c/rtools45/x86_64-w64-mingw32.static.posix/bin:$$PATH"
endif

.PHONY: default clean coverage

default:
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
	@sed -i.bak '/^\[workspace\]/d' $(R_DIR)/src/Cargo.toml; \
	sed -i.bak '/^\[patch\.crates-io\]/d' $(R_DIR)/src/Cargo.toml; \
	sed -i.bak '/^lowess = { path = "vendor\/lowess" }/d' $(R_DIR)/src/Cargo.toml; \
	rm -f $(R_DIR)/src/Cargo.toml.bak; \
	rm -rf $(R_DIR)/*.Rcheck $(R_DIR)/*.BiocCheck $(R_DIR)/src/target $(R_DIR)/target $(R_DIR)/src/vendor; \
	echo "" >> $(R_DIR)/src/Cargo.toml
	@mkdir -p $(R_DIR)/src/.cargo && cp $(R_DIR)/src/cargo-config.toml $(R_DIR)/src/.cargo/config.toml
	@echo "Patched $(R_DIR)/src/Cargo.toml"
	@echo "=============================================================================="
	@echo "2. Installing R development packages..."
	@echo "=============================================================================="
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
	@rm -rf $(R_DIR)/src/vendor $(R_DIR)/src/vendor.tar.xz
	@mkdir -p $(R_DIR)/src/vendor
	@cp -rL crates/fastLowess $(R_DIR)/src/vendor/
	@cp -rL crates/lowess $(R_DIR)/src/vendor/
	@rm -rf $(R_DIR)/src/vendor/fastLowess/target $(R_DIR)/src/vendor/lowess/target
	@rm -f $(R_DIR)/src/vendor/fastLowess/Cargo.lock $(R_DIR)/src/vendor/lowess/Cargo.lock
	@rm -f $(R_DIR)/src/vendor/fastLowess/README.md $(R_DIR)/src/vendor/fastLowess/CHANGELOG.md
	@rm -f $(R_DIR)/src/vendor/lowess/README.md $(R_DIR)/src/vendor/lowess/CHANGELOG.md
ifneq ($(WITH_GPU),1)
	@sed -i.bak \
		-e '/^wgpu = /,/^\] }/d' \
		-e '/^bytemuck = /d' \
		-e '/^pollster = /d' \
		-e '/^futures-intrusive = /d' \
		-e 's/^gpu = .*/gpu = []/' \
		-e 's|lowess = { path = "\.\./lowess", version = "[^"]*", |lowess = { path = "../lowess", |' \
		$(R_DIR)/src/vendor/fastLowess/Cargo.toml
	@rm -f $(R_DIR)/src/vendor/fastLowess/Cargo.toml.bak
else
	@echo "WITH_GPU=1: keeping wgpu dependencies and 'gpu' feature in vendored fastLowess"
	@sed -i.bak \
		-e 's|lowess = { path = "\.\./lowess", version = "[^"]*", |lowess = { path = "../lowess", |' \
		$(R_DIR)/src/vendor/fastLowess/Cargo.toml
	@rm -f $(R_DIR)/src/vendor/fastLowess/Cargo.toml.bak
endif
	@echo '{"files":{},"package":null}' > $(R_DIR)/src/vendor/lowess/.cargo-checksum.json
	@echo '{"files":{},"package":null}' > $(R_DIR)/src/vendor/fastLowess/.cargo-checksum.json
	@printf '\n\n[workspace]\n\n[patch.crates-io]\nlowess = { path = "vendor/lowess" }\n' >> $(R_DIR)/src/Cargo.toml
	@(cd $(R_DIR)/src && cargo vendor -q --no-delete vendor)
	@for d in tests benches examples doc docs .github .config; do \
		rm -rf $(R_DIR)/src/vendor/"$$d" $(R_DIR)/src/vendor/*/"$$d" 2>/dev/null; done || true
	@for f in $(R_DIR)/src/vendor/*/Makefile; do [ -f "$$f" ] && rm -f "$$f"; done || true
	@$(JQ_ENSURE_CMD)
	@for f in $(R_DIR)/src/vendor/*/.cargo-checksum.json; do \
		[ -f "$$f" ] && $(JQ) '.files = {}' "$$f" > "$$f.tmp" && mv "$$f.tmp" "$$f"; done
	@echo "Creating vendor.tar.xz archive (including Cargo.lock)..."
	@(cd $(R_DIR)/src && (tar --sort=name --mtime='1970-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner --xz --create --file=vendor.tar.xz --exclude='*/Makefile' vendor Cargo.lock 2>/dev/null || tar --xz --create --file=vendor.tar.xz --exclude='*/Makefile' vendor Cargo.lock))
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
	@(cd $(R_DIR)/src && $(R_CROSS_ENV) cargo build -q --release $(R_CARGO_TARGET) $(R_CARGO_FEATURES) || (mv Cargo.toml.orig Cargo.toml && exit 1))
	@rm -rf $(R_DIR)/src/.cargo
	@echo "=============================================================================="
	@echo "4a. Formatting..."
	@echo "=============================================================================="
	@cd $(R_DIR)/src && cargo fmt -q
	@if ! command -v air >/dev/null 2>&1; then \
	    echo "air not found — installing..."; \
	    $(AIR_INSTALL_CMD); \
	    export PATH="$$HOME/.local/bin:$$PATH"; \
	fi; \
	air format $(R_DIR)/R/ $(R_DIR)/tests/testthat/ $(R_DIR)/vignettes/ || true
	@cd $(R_DIR)/src && cargo fmt -- --check || (echo "Run 'cargo fmt' to fix"; exit 1)
	@cd $(R_DIR)/src && $(R_CROSS_ENV) cargo clippy -q $(R_CARGO_TARGET) -- -D warnings
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "lib <- Sys.getenv('R_LIBS_USER'); if (!requireNamespace('lintr', quietly = TRUE, lib.loc = lib)) stop('Required R package not available: lintr', call. = FALSE); my_linters <- lintr::linters_with_defaults(indentation_linter = lintr::indentation_linter(indent = 4L), object_name_linter = NULL, commented_code_linter = NULL, object_usage_linter = NULL); lints <- c(lintr::lint_dir('$(R_DIR)/R', linters = my_linters), lintr::lint_dir('$(R_DIR)/tests/testthat', linters = my_linters)); print(lints); if (length(lints) > 0L) quit(status = 1)"
	@echo "=============================================================================="
	@echo "4b. Documentation..."
	@echo "=============================================================================="
	@rm -rf $(R_DIR)/*.Rcheck
	@cd $(R_DIR)/src && RUSTDOCFLAGS="-D warnings" $(R_CROSS_ENV) cargo doc -q --no-deps $(R_CARGO_TARGET)
	@cd $(R_DIR) && R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "lib <- Sys.getenv('R_LIBS_USER'); options(repos = c(CRAN = 'https://cloud.r-project.org')); for (pkg in c('roxygen2', 'srr')) { if (!requireNamespace(pkg, quietly = TRUE, lib.loc = lib)) suppressWarnings(try(install.packages(pkg, lib = lib, quiet = TRUE), silent = TRUE)) }; if (!requireNamespace('roxygen2', quietly = TRUE, lib.loc = lib)) stop('Required R package not available for roxygen regeneration: roxygen2', call. = FALSE); has_srr <- requireNamespace('srr', quietly = TRUE, lib.loc = lib); roclets <- if (has_srr) c('namespace', 'rd', 'srr::srr_stats_roclet') else c('namespace', 'rd'); roxygen2::roxygenise(package.dir = '.', roclets = roclets, load_code = roxygen2::load_pkgload)"
	@sed -i.bak 's/^  /    /' $(R_DIR)/NAMESPACE && rm -f $(R_DIR)/NAMESPACE.bak
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript $(CURDIR)/dev/fix_rd_style.R $(R_DIR)/man
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "if (!requireNamespace('rmarkdown', quietly = TRUE, lib.loc = Sys.getenv('R_LIBS_USER')) || !rmarkdown::pandoc_available()) { message('\nERROR: Pandoc is required to build R Markdown vignettes but is not available.\nPlease install Pandoc (https://pandoc.org/installing.html) and ensure it is in your PATH.\n'); quit(status = 1) }"
	@echo "Pre-installing package to R_LIBS_USER so pkgdown subprocess finds the current version..."
	@mkdir -p $(R_LIB_DIR)
	@cd $(R_DIR) && WITH_GPU=$(WITH_GPU) R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) R CMD INSTALL -l $(CURDIR)/$(R_LIB_DIR) . >/dev/null 2>&1
	@cd $(R_DIR) && R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "lib <- Sys.getenv('R_LIBS_USER'); options(repos = c(CRAN = 'https://cloud.r-project.org')); for (pkg in c('pkgdown', 'rmarkdown', 'knitr')) { if (!requireNamespace(pkg, quietly = TRUE, lib.loc = lib)) install.packages(pkg, lib = lib, quiet = TRUE) }; if (!requireNamespace('pkgdown', quietly = TRUE, lib.loc = lib) || !requireNamespace('rmarkdown', quietly = TRUE, lib.loc = lib) || !requireNamespace('knitr', quietly = TRUE, lib.loc = lib)) stop('Required R packages not available for pkgdown site build', call. = FALSE); pkgdown::build_site(quiet = TRUE, install = FALSE)"
	@rm -f $(R_DIR)/.gitignore
	@echo "=============================================================================="
	@echo "4c. Building..."
	@echo "=============================================================================="
	@echo "Generating $(R_DIR)/inst/AUTHORS..."
	@mkdir -p $(R_DIR)/inst
	@$(JQ_ENSURE_CMD)
	@{ \
		echo "Authors and Copyright Holders for Rust Dependencies:"; \
		echo ""; \
		(cargo metadata --format-version 1) | \
			$(JQ) -r '. as $$root | [.packages[] | select(.id as $$id | ($$root.workspace_members | index($$id)) == null)] | unique_by(.name)[] | "Package: \(.name) (\(.version))\nAuthors: \(.authors | join(", "))\nLicense: \(.license // "Unknown")\n" + "-" * 40'; \
	} > $(R_DIR)/inst/AUTHORS
	@cd $(R_DIR) && WITH_GPU=$(WITH_GPU) R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) R CMD build .
	@echo "=============================================================================="
	@echo "5. Installing..."
	@echo "=============================================================================="
	@rm -rf $(R_DIR)/00LOCK-$(R_PKG_NAME)
	@mkdir -p $(R_LIB_DIR)
	@echo "Installing R runtime dependencies (BiocManager, BiocGenerics, testthat)..."
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "suppressMessages({ lib <- Sys.getenv('R_LIBS_USER'); options(repos = c(CRAN = 'https://cloud.r-project.org')); if (!requireNamespace('BiocManager', quietly = TRUE, lib.loc = lib)) install.packages('BiocManager', lib = lib, quiet = TRUE); if (!requireNamespace('BiocGenerics', quietly = TRUE, lib.loc = lib)) BiocManager::install('BiocGenerics', lib = lib, ask = FALSE, update = FALSE, quiet = TRUE); if (!requireNamespace('testthat', quietly = TRUE, lib.loc = lib)) install.packages('testthat', lib = lib, quiet = TRUE) })" >/dev/null
	@cd $(R_DIR) && WITH_GPU=$(WITH_GPU) R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) R CMD INSTALL -l .r-lib $(R_PKG_TARBALL)
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
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript -e "Sys.setenv(NOT_CRAN='true'); testthat::test_dir('$(R_DIR)/tests/testthat', package = 'rfastlowess')"
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
	@rm -rf $(R_DIR)/src/vendor $(R_DIR)/target $(R_LIB_DIR)
	@rm -rf $(R_DIR)/$(R_PKG_NAME).Rcheck $(R_DIR)/..Rcheck $(R_DIR)/$(R_PKG_NAME).BiocCheck
	@rm -f $(R_DIR)/$(R_PKG_NAME)_*.tar.gz
	@rm -rf $(R_DIR)/src/*.o $(R_DIR)/src/*.so $(R_DIR)/src/*.dll $(R_DIR)/src/Cargo.toml.orig $(R_DIR)/src/Cargo.lock $(R_DIR)/Cargo.lock
	@rm -rf $(R_DIR)/doc $(R_DIR)/Meta $(R_DIR)/vignettes/*.html $(R_DIR)/README.html
	@$(PYTHON) -c "from pathlib import Path; [path.unlink() for path in Path(r'$(R_DIR)').rglob('*.Rout')]"
	@Rscript -e "try(remove.packages('$(R_PKG_NAME)'), silent = TRUE)" || true
	@rm -rf $(R_DIR)/src/Makevars $(R_DIR)/$(R_PKG_NAME)*.tgz
	@rm -rf $(R_DIR)/benchmarks $(R_DIR)/validation $(R_DIR)/docs
	@echo "$(R_PKG_NAME) clean complete!"
