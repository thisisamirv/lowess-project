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

TEMP ?= /tmp
ifeq ($(OS),Windows_NT)
	TEMP := /tmp
endif

NODE_PKG := fastlowess-node
NODE_DIR := bindings/nodejs
NODE_TEST_DIR := bindings/nodejs/tests

.PHONY: default dev clean

default:
	@cd $(NODE_DIR) && $(NPM) install -q
	@cd $(NODE_DIR) && $(NPM) run build
	@cd $(NODE_DIR) && $(NPM) link

dev:
	@echo "Running $(NODE_PKG) checks..."
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(NODE_PKG) -- --check
	@echo "=============================================================================="
	@echo "2. Linting & Building..."
	@echo "=============================================================================="
	@cargo clippy -q -p $(NODE_PKG) --all-targets -- -D warnings
	@echo "Linting Node.js files..."
	@cd $(NODE_DIR) && $(NPM) install
	@cd $(NODE_DIR) && $(NPM) update
	@cd $(NODE_DIR) && $(NPM) audit || true
	@cd $(NODE_DIR) && $(NPX) -y depcheck --ignores="fastlowess-*,oxlint,typedoc-plugin-pages,typedoc-plugin-markdown,astro:content,katex"
	@cd $(NODE_DIR) && ($(NPM) outdated | grep -v "fastlowess-" || true)
	@cd $(NODE_DIR) && $(NPM) ci --dry-run
	@cd $(NODE_DIR) && $(NPX) -y -p typescript tsc -p tsconfig.typedoc.json
	@$(NPX) -y oxlint $(NODE_DIR)/index.js $(NODE_TEST_DIR)/test_fastlowess.js
	@cd $(NODE_DIR) && $(NPM) run build
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@cd $(NODE_DIR) && $(NPM) test
	@echo "=============================================================================="
	@echo "Doc snippet tests (nodejs)..."
	@$(PYTHON) dev/add-readme-to-docs.py $(NODE_DIR)
	@$(PYTHON) dev/check_links.py --lang nodejs
	@$(PYTHON) dev/verify_snippets.py --lang nodejs --timeout 120 --update-outputs
	@$(PYTHON) dev/update_changelogs.py nodejs
	@echo "$(NODE_PKG) checks completed successfully!"

clean:
	@echo "Cleaning $(NODE_PKG)..."
	@cargo clean -p $(NODE_PKG)
	@git clean -fdX -- $(NODE_DIR)
	@echo "$(NODE_PKG) clean complete!"