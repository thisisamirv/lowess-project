.SHELLFLAGS := -ec

include mk/config.mk

NODE_PKG      := fastlowess-node
NODE_DIR      := bindings/nodejs
NODE_TEST_DIR := tests/nodejs

.PHONY: default clean

default:
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
	@cd $(NODE_DIR) && $(NPM) audit || true
	@cd $(NODE_DIR) && $(NPX) -y depcheck --ignores="fastlowess-*,oxlint"
	@cd $(NODE_DIR) && ($(NPM) outdated | grep -v "fastlowess-" || true)
	@cd $(NODE_DIR) && $(NPM) ci --dry-run
	@cd $(NODE_DIR) && $(NPX) -y -p typescript tsc index.d.ts --noEmit --allowJs
	@$(NPX) -y oxlint $(NODE_DIR)/index.js tests/nodejs/test_fastlowess.js examples/nodejs/*.js
	@cd $(NODE_DIR) && $(NPM) run build
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@cd $(NODE_DIR) && $(NPM) test
	@echo "$(NODE_PKG) checks completed successfully!"

clean:
	@echo "Cleaning $(NODE_PKG)..."
	@cargo clean -p $(NODE_PKG)
	@rm -rf $(NODE_DIR)/node_modules $(NODE_DIR)/fastlowess.*.node
	@echo "$(NODE_PKG) clean complete!"
