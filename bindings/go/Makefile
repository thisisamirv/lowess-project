THIS_MK := $(lastword $(MAKEFILE_LIST))

.SHELLFLAGS := -ec

UNAME_S := $(shell uname -s)

ifeq ($(OS),Windows_NT)
	HOST_PLATFORM  := windows
else ifeq ($(UNAME_S),Darwin)
	HOST_PLATFORM  := macos
else
	HOST_PLATFORM  := linux
endif

PYTHON ?= python3

GO_PKG := fastlowess-go
GO_DIR := bindings/go
GO_CARGO_PROFILE := --profile release-c
GO_CARGO_TARGET :=

# Go's cgo drives a GCC-compatible C compiler (MinGW-w64 on Windows), so the
# native library must be built for the GNU ABI on Windows, unlike the C++
# binding which targets MSVC.
ifeq ($(HOST_PLATFORM),windows)
	GO_CARGO_TARGET := --target x86_64-pc-windows-gnu
	GO_LIBRARY_DIR  := target/x86_64-pc-windows-gnu/release-c
else
	GO_LIBRARY_DIR  := target/release-c
endif

GO_LIBRARY_DIR_ABS := $(abspath $(GO_LIBRARY_DIR))

.PHONY: default dev clean docs

default:
	@if [ -n "$(GO_CARGO_TARGET)" ]; then \
		_rust_target=$$(echo "$(GO_CARGO_TARGET)" | sed 's/--target //'); \
		rustup target add "$$_rust_target" 2>/dev/null || true; \
	fi
	@cargo build -q -p $(GO_PKG) $(GO_CARGO_PROFILE) $(GO_CARGO_TARGET)
	@echo "C header generated at $(GO_DIR)/include/fastlowess_go.h"
	@cd $(GO_DIR)/fastlowess && go build ./...
	@echo "$(GO_PKG) built successfully!"

dev:
	@echo "Running $(GO_PKG) checks..."
	@echo "=============================================================================="
	@echo "1. Rust formatting & linting..."
	@echo "=============================================================================="
	@cargo fmt -p $(GO_PKG) -- --check
	@cargo clippy -q -p $(GO_PKG) --all-targets $(GO_CARGO_TARGET) -- -D warnings
	@if [ -n "$(GO_CARGO_TARGET)" ]; then \
		_rust_target=$$(echo "$(GO_CARGO_TARGET)" | sed 's/--target //'); \
		rustup target add "$$_rust_target" 2>/dev/null || true; \
	fi
	@cargo build -q -p $(GO_PKG) $(GO_CARGO_PROFILE) $(GO_CARGO_TARGET)
	@echo "C header generated at $(GO_DIR)/include/fastlowess_go.h"
	@echo "=============================================================================="
	@echo "1b. cbindgen idempotency check..."
	@echo "=============================================================================="
	@if ! command -v cbindgen >/dev/null 2>&1; then \
		echo "cbindgen not found. Installing latest version..."; \
		cargo install cbindgen --force; \
	fi
	@cbindgen --config $(GO_DIR)/cbindgen.toml --crate $(GO_PKG) --output /tmp/fastlowess_go_new.h 2>/dev/null && \
		diff -q $(GO_DIR)/include/fastlowess_go.h /tmp/fastlowess_go_new.h > /dev/null || \
		(echo "Error: fastlowess_go.h is stale — run 'cargo build -p $(GO_PKG) $(GO_CARGO_PROFILE) $(GO_CARGO_TARGET)' to regenerate"; exit 1)
	@echo "cbindgen header is up-to-date."
	@echo "=============================================================================="
	@echo "1c. Symbol export verification..."
	@echo "=============================================================================="
	@nm $(GO_LIBRARY_DIR)/libfastlowess_go.a 2>/dev/null | grep -q go_lowess_new || \
		(echo "Error: go_lowess_new not exported"; exit 1)
	@nm $(GO_LIBRARY_DIR)/libfastlowess_go.a 2>/dev/null | grep -q go_streaming_new || \
		(echo "Error: go_streaming_new not exported"; exit 1)
	@nm $(GO_LIBRARY_DIR)/libfastlowess_go.a 2>/dev/null | grep -q go_online_new || \
		(echo "Error: go_online_new not exported"; exit 1)
	@echo "All Go FFI exports verified."
	@echo "=============================================================================="
	@echo "2. Go formatting, vetting & testing..."
	@echo "=============================================================================="
	@cd $(GO_DIR)/fastlowess && test -z "$$(gofmt -l .)" || (echo "Error: gofmt found unformatted files"; gofmt -l .; exit 1)
	@cd $(GO_DIR)/fastlowess && go vet ./...
	@cd $(GO_DIR)/fastlowess && go test ./... -v
	@echo "=============================================================================="
	@echo "Doc snippet tests (go)..."
	@$(PYTHON) dev/add-go-outputs.py $(GO_DIR)
	@$(PYTHON) dev/verify_snippets.py --lang go --timeout 120
	@$(PYTHON) dev/update_changelogs.py
	@echo "$(GO_PKG) checks completed successfully!"

clean:
	@echo "Cleaning $(GO_PKG)..."
	@cargo clean -p $(GO_PKG)
	@git clean -fdX -- $(GO_DIR)
	@echo "$(GO_PKG) clean complete!"

docs:
	@echo "Building $(GO_PKG) docs (Hugo)..."
	@if ! command -v hugo >/dev/null 2>&1; then \
		echo "hugo not found. Installing latest version..."; \
		go install github.com/gohugoio/hugo@latest; \
	fi
	@cd $(GO_DIR)/docs-site && hugo --gc --minify
	@echo "Docs built at $(GO_DIR)/docs-site/public/index.html"