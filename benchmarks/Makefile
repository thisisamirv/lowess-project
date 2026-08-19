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

# Set WITH_GPU=1 to build rfastlowess with the GPU execution backend enabled
# (requires a Vulkan/Metal/DX12-capable GPU driver at runtime).
WITH_GPU ?= 0
R_CARGO_FEATURES :=
ifeq ($(WITH_GPU),1)
	R_CARGO_FEATURES := --features gpu
endif

R_PKG_NAME    := rfastlowess
R_PKG_VERSION  = $(shell grep "^Version:" $(R_DIR)/DESCRIPTION | sed 's/Version: //')
R_PKG_TARBALL  = $(R_PKG_NAME)_$(R_PKG_VERSION).tar.gz

R_CARGO_TARGET :=
R_CROSS_ENV    :=
ifeq ($(OS),Windows_NT)
	R_CARGO_TARGET := --target x86_64-pc-windows-gnu
	R_CROSS_ENV    := PATH="/c/rtools45/x86_64-w64-mingw32.static.posix/bin:$$PATH"
endif

# jq (used to patch vendored .cargo-checksum.json files). Not preinstalled
# on plain Windows/macOS dev machines the way it is on GitHub-hosted CI
# runners, so fall back to downloading a static binary if it's not already
# on PATH.
JQ_DIR := $(R_DIR)/.bin
ifeq ($(OS),Windows_NT)
	JQ_LOCAL := $(JQ_DIR)/jq.exe
	JQ_ASSET := jq-windows-amd64.exe
else ifeq ($(shell uname -s),Darwin)
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

.PHONY: all install build bench-r bench-rfastlowess-serial bench-rfastlowess-parallel bench-gpu-transfer bench-gpu-rust bench-cpu-vs-gpu compare clean

all: install

# ==============================================================================
# Install rfastlowess to system R
# ==============================================================================
install: build
	@echo "Installing $(R_PKG_NAME) to system R..."
	@WITH_GPU=$(WITH_GPU) R CMD INSTALL $(R_DIR)/$(R_PKG_TARBALL)
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
	@echo "3. Vendoring Cargo dependencies..."
	@echo "============================================================"
	@rm -rf $(R_DIR)/src/vendor $(R_DIR)/src/vendor.tar.xz
	@mkdir -p $(R_DIR)/src/vendor
	@cp -rL $(REPO_ROOT)/crates/fastLowess $(R_DIR)/src/vendor/
	@cp -rL $(REPO_ROOT)/crates/lowess  $(R_DIR)/src/vendor/
	@rm -rf $(R_DIR)/src/vendor/fastLowess/target $(R_DIR)/src/vendor/lowess/target
	@rm -f  $(R_DIR)/src/vendor/fastLowess/Cargo.lock $(R_DIR)/src/vendor/lowess/Cargo.lock
	@rm -f  $(R_DIR)/src/vendor/fastLowess/README.md  $(R_DIR)/src/vendor/fastLowess/CHANGELOG.md
	@rm -f  $(R_DIR)/src/vendor/lowess/README.md   $(R_DIR)/src/vendor/lowess/CHANGELOG.md
	@echo '{"files":{},"package":null}' > $(R_DIR)/src/vendor/lowess/.cargo-checksum.json
	@echo '{"files":{},"package":null}' > $(R_DIR)/src/vendor/fastLowess/.cargo-checksum.json
ifneq ($(WITH_GPU),1)
	@# Strip GPU deps and version from the lowess path dep in fastLowess
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
	@printf '\n\n[workspace]\n\n[patch.crates-io]\nlowess = { path = "vendor/lowess" }\n' >> $(R_DIR)/src/Cargo.toml
	@(cd $(R_DIR)/src && cargo vendor -q --no-delete vendor)
	@# Strip test/bench/example/doc dirs to reduce archive size
	@find $(R_DIR)/src/vendor -mindepth 2 -maxdepth 2 -type d \
		\( -name tests -o -name benches -o -name examples \
		   -o -name doc -o -name docs -o -name .github -o -name .config \) \
		-exec rm -rf {} \; 2>/dev/null || true
	@# Reset checksums — cargo accepts {"files":{}} for vendored crates.
	@# Must preserve each crate's "package" checksum (only "files" is reset),
	@# otherwise cargo can't verify crates.io-sourced deps against Cargo.lock.
	@$(JQ_ENSURE_CMD)
	@for f in $(R_DIR)/src/vendor/*/.cargo-checksum.json; do \
		[ -f "$$f" ] && $(JQ) '.files = {}' "$$f" > "$$f.tmp" && mv "$$f.tmp" "$$f"; done
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
	@(cd $(R_DIR)/src && $(R_CROSS_ENV) cargo build -q --release $(R_CARGO_TARGET) $(R_CARGO_FEATURES) \
	  || (mv Cargo.toml.orig Cargo.toml && exit 1))
	@rm -rf $(R_DIR)/src/.cargo

	@echo "============================================================"
	@echo "6. Building R package tarball..."
	@echo "============================================================"
	@cd $(R_DIR) && WITH_GPU=$(WITH_GPU) R CMD build --no-build-vignettes --no-manual .
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
# CPU vs GPU crossover benchmark
# ==============================================================================
bench-cpu-vs-gpu:
	@echo "Building CPU-vs-GPU benchmark binary..."
	@(cd $(REPO_ROOT) && cargo build -q --release --bin bench_cpu_vs_gpu)
	@echo "Running CPU vs GPU sweep..."
	@$(REPO_ROOT)/target/release/bench_cpu_vs_gpu output/rust_benchmark_cpu_vs_gpu.json
	@echo "Analysing crossover points..."
	@$(PYTHON) compare_cpu_gpu.py output/rust_benchmark_cpu_vs_gpu.json

# ==============================================================================
# GPU Rust end-to-end benchmark
# ==============================================================================
bench-gpu-rust:
	@echo "Building GPU benchmark binary..."
	@(cd $(REPO_ROOT) && cargo build -q --release --bin bench_gpu)
	@echo "Running GPU end-to-end benchmark..."
	@$(REPO_ROOT)/target/release/bench_gpu output/rust_benchmark_gpu.json

# ==============================================================================
# GPU transfer benchmark
# ==============================================================================
bench-gpu-transfer:
	@echo "Installing required Python packages..."
	@$(PYTHON) -m pip install -q wgpu numpy
	@echo "Running GPU data-transfer overhead benchmark..."
	@$(PYTHON) gpu_transfer.py

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
