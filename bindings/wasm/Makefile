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

TEMP ?= /tmp
ifeq ($(OS),Windows_NT)
	TEMP := /tmp
endif

WASM_PKG := fastlowess-wasm
WASM_DIR := bindings/wasm
WASM_TEST_DIR := bindings/wasm/tests

.PHONY: default dev clean

default:
	@if ! command -v wasm-pack >/dev/null 2>&1; then \
		echo "wasm-pack not found. Installing..."; \
		cargo install wasm-pack; \
	fi
	@cd $(WASM_DIR) && wasm-pack build --target nodejs --out-dir pkg
	@cd $(WASM_DIR) && wasm-pack build --target web --out-dir pkg-web
	@cd $(WASM_DIR)/pkg && $(NPM) link

dev:
	@echo "Running $(WASM_PKG) checks..."
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(WASM_PKG) -- --check
	@echo "=============================================================================="
	@echo "2. Linting & Building..."
	@echo "=============================================================================="
	@cargo clippy -q -p $(WASM_PKG) --all-targets -- -D warnings
	@echo "Linting WASM JS files..."
	@cd $(WASM_DIR) && $(NPM) install -q
	@cd $(WASM_DIR) && $(NPM) update
	@cd $(WASM_DIR) && $(NPM) audit || true
	@cd $(WASM_DIR) && $(NPX) -y depcheck --ignores="oxlint,typedoc-plugin-pages,typedoc-plugin-markdown,astro:content,katex"
	@cd $(WASM_DIR) && ($(NPM) outdated | grep -v "fastlowess-" || true)
	@cd $(WASM_DIR) && $(NPM) ci --dry-run
	@$(NPX) -y oxlint $(WASM_DIR)/src/*.js $(WASM_TEST_DIR)/*.js
	@if ! command -v wasm-pack >/dev/null 2>&1; then \
		echo "wasm-pack not found. Installing..."; \
		cargo install wasm-pack; \
	fi
	@cd $(WASM_DIR) && wasm-pack build --target nodejs --out-dir pkg
	@echo "Checking WASM size (Limit: 2MB)..."
	@[ $$($(STAT_SIZE_CMD) $(WASM_DIR)/pkg/fastlowess_wasm_bg.wasm) -le 2097152 ] || (echo "Error: WASM size exceeded 2MB"; exit 1)
	@echo "Building for Web..."
	@cd $(WASM_DIR) && wasm-pack build --target web --out-dir pkg-web
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@cd $(WASM_DIR) && wasm-pack test --node
	@echo "Running JS tests..."
	@node --test $(WASM_TEST_DIR)/test_fastlowess_wasm.js
	@echo "=============================================================================="
	@echo "Doc snippet tests (wasm)..."
	@echo "=============================================================================="
	@$(PYTHON) dev/add-readme-to-docs.py $(WASM_DIR)
	@$(PYTHON) dev/verify_snippets.py --lang wasm --timeout 120 --update-outputs
	@$(PYTHON) dev/update_changelogs.py wasm
	@echo "=============================================================================="
clean:
	@echo "Cleaning $(WASM_PKG)..."
	@cargo clean -p $(WASM_PKG)
	@git clean -fdX -- $(WASM_DIR)
	@echo "$(WASM_PKG) clean complete!"