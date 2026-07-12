# benchmarks/Makefile — Build, install rfastlowess on system R, and run benchmarks.
#
# Targets:
#   install        (default) Build rfastlowess and install to system R
#   build          Build the rfastlowess tarball only
#   bench          Run both --serial and --parallel benchmarks
#   bench-serial   Run --serial benchmark only
#   bench-parallel Run --parallel benchmark only
#   clean          Remove build artefacts

PYTHON ?= python

REPO_ROOT := $(abspath ..)
R_DIR     := $(REPO_ROOT)/bindings/r
DEV_DIR   := $(REPO_ROOT)/dev

R_PKG_NAME    := rfastlowess
R_PKG_VERSION  = $(shell grep "^Version:" $(R_DIR)/DESCRIPTION | sed 's/Version: //')
R_PKG_TARBALL  = $(R_PKG_NAME)_$(R_PKG_VERSION).tar.gz

R_CARGO_TARGET :=
R_CROSS_ENV    :=
ifeq ($(OS),Windows_NT)
	R_CARGO_TARGET := --target x86_64-pc-windows-gnu
	R_CROSS_ENV    := PATH="/c/rtools45/x86_64-w64-mingw32.static.posix/bin:$$PATH"
endif

.PHONY: all install build bench-r bench-rfastlowess-serial bench-rfastlowess-parallel compare clean

all: install

# ==============================================================================
# Install rfastlowess to system R
# ==============================================================================
install: build
	@echo "Installing $(R_PKG_NAME) to system R..."
	@R CMD INSTALL $(R_DIR)/$(R_PKG_TARBALL)
	@echo "$(R_PKG_NAME) installed!"

# ==============================================================================
# Build the rfastlowess tarball
# ==============================================================================
build:
	@echo "============================================================"
	@echo "1. Syncing version..."
	@echo "============================================================"
	@VERSION=$$(grep "^version =" $(R_DIR)/src/Cargo.toml | head -n1 | sed 's/version = "\(.*\)"/\1/'); \
	 sed -i.bak "s/^Version: .*/Version: $$VERSION/" $(R_DIR)/DESCRIPTION && rm -f $(R_DIR)/DESCRIPTION.bak; \
	 echo "  Version: $$VERSION"

	@echo "============================================================"
	@echo "2. Patching Cargo.toml for isolated build..."
	@echo "============================================================"
	@if [ -f $(R_DIR)/src/Cargo.toml.orig ]; then mv $(R_DIR)/src/Cargo.toml.orig $(R_DIR)/src/Cargo.toml; fi
	@cp $(R_DIR)/src/Cargo.toml $(R_DIR)/src/Cargo.toml.orig
	@sed -i.bak '/^\[workspace\]/d'                              $(R_DIR)/src/Cargo.toml; \
	 sed -i.bak '/^\[patch\.crates-io\]/d'                       $(R_DIR)/src/Cargo.toml; \
	 sed -i.bak '/^lowess = { path = "vendor\/lowess" }/d'        $(R_DIR)/src/Cargo.toml; \
	 rm -f $(R_DIR)/src/Cargo.toml.bak
	@mkdir -p $(R_DIR)/src/.cargo && cp $(R_DIR)/src/cargo-config.toml $(R_DIR)/src/.cargo/config.toml

	@echo "============================================================"
	@echo "3. Installing Python build tools..."
	@echo "============================================================"
	@$(PYTHON) -m pip install -q tomli tomli_w >/dev/null 2>&1 || true

	@echo "============================================================"
	@echo "4. Vendoring Cargo dependencies..."
	@echo "============================================================"
	@$(PYTHON) $(DEV_DIR)/prepare_cargo.py clean $(R_DIR)/src/Cargo.toml -q
	@rm -rf $(R_DIR)/src/vendor $(R_DIR)/src/vendor.tar.xz
	@mkdir -p $(R_DIR)/src/vendor
	@cp -rL $(REPO_ROOT)/crates/fastLowess $(R_DIR)/src/vendor/
	@cp -rL $(REPO_ROOT)/crates/lowess  $(R_DIR)/src/vendor/
	@rm -rf $(R_DIR)/src/vendor/fastLowess/target $(R_DIR)/src/vendor/lowess/target
	@rm -f  $(R_DIR)/src/vendor/fastLowess/Cargo.lock $(R_DIR)/src/vendor/lowess/Cargo.lock
	@rm -f  $(R_DIR)/src/vendor/fastLowess/README.md  $(R_DIR)/src/vendor/fastLowess/CHANGELOG.md
	@rm -f  $(R_DIR)/src/vendor/lowess/README.md   $(R_DIR)/src/vendor/lowess/CHANGELOG.md
	@$(PYTHON) $(DEV_DIR)/patch_vendor_crates.py $(REPO_ROOT)/Cargo.toml $(R_DIR)/src/vendor -q
	@echo '{"files":{},"package":null}' > $(R_DIR)/src/vendor/lowess/.cargo-checksum.json
	@echo '{"files":{},"package":null}' > $(R_DIR)/src/vendor/fastLowess/.cargo-checksum.json
	@$(PYTHON) $(DEV_DIR)/prepare_cargo.py isolate $(R_DIR)/src/Cargo.toml -q
	@(cd $(R_DIR)/src && cargo vendor -q --no-delete vendor)
	@$(PYTHON) $(DEV_DIR)/clean_checksums.py -q $(R_DIR)/src/vendor
	@echo "  Creating vendor.tar.xz..."
	@(cd $(R_DIR)/src && \
	  (tar --sort=name --mtime='1970-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner \
	       --xz -cf vendor.tar.xz vendor Cargo.lock 2>/dev/null \
	   || tar --xz -cf vendor.tar.xz vendor Cargo.lock))
	@rm -rf $(R_DIR)/src/vendor
	@(cd $(R_DIR)/src && tar -xf vendor.tar.xz) \
	 && rm -f $(R_DIR)/src/vendor/*/CITATION.cff $(R_DIR)/src/vendor/*/CITATION

	@echo "============================================================"
	@echo "5. Building Rust library..."
	@echo "============================================================"
	@(cd $(R_DIR)/src && $(R_CROSS_ENV) cargo build -q --release $(R_CARGO_TARGET) \
	  || (mv Cargo.toml.orig Cargo.toml && exit 1))
	@rm -rf $(R_DIR)/src/.cargo

	@echo "============================================================"
	@echo "6. Building R package tarball..."
	@echo "============================================================"
	@cd $(R_DIR) && R CMD build --no-build-vignettes --no-manual .
	@if [ -f $(R_DIR)/src/Cargo.toml.orig ]; then mv $(R_DIR)/src/Cargo.toml.orig $(R_DIR)/src/Cargo.toml; fi
	@echo "  Tarball: $(R_DIR)/$(R_PKG_TARBALL)"

# ==============================================================================
# Run benchmarks (rfastlowess must be installed first)
# ==============================================================================
bench-rfastlowess-serial:
	@echo "Running rfastlowess benchmarks (serial)..."
	@Rscript rfastlowess.R --serial

bench-rfastlowess-parallel:
	@echo "Running rfastlowess benchmarks (parallel)..."
	@Rscript rfastlowess.R --parallel

bench-r:
	@echo "Running stats::lowess benchmarks..."
	@Rscript stats_lowess.R

# ==============================================================================
# Compare / plot
# ==============================================================================
compare:
	@echo "Generating benchmark comparison plot..."
	@$(PYTHON) compare.py

# ==============================================================================
# Clean
# ==============================================================================
clean:
	@echo "Cleaning benchmark build artefacts..."
	@if [ -f $(R_DIR)/src/Cargo.toml.orig ]; then mv $(R_DIR)/src/Cargo.toml.orig $(R_DIR)/src/Cargo.toml; fi
	@rm -rf $(R_DIR)/src/vendor $(R_DIR)/src/vendor.tar.xz $(R_DIR)/src/.cargo
	@rm -f  $(R_DIR)/$(R_PKG_NAME)_*.tar.gz
	@(cd $(R_DIR)/src && cargo clean -q 2>/dev/null || true)
	@echo "Clean complete!"
