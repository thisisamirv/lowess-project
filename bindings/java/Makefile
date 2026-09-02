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

CHECKSTYLE_VERSION := 14.1.0
CHECKSTYLE_JAR := checkstyle-$(CHECKSTYLE_VERSION)-all.jar
CHECKSTYLE_URL := https://github.com/checkstyle/checkstyle/releases/download/checkstyle-$(CHECKSTYLE_VERSION)/$(CHECKSTYLE_JAR)

JAVA_PKG := fastlowess-java
JAVA_DIR := bindings/java

ifeq ($(HOST_PLATFORM),windows)
	JAVA_SHARED_LIB := target/debug/fastlowess_java.dll
	JAVA_EXPORT_SCAN := objdump -p $(JAVA_SHARED_LIB)
else ifeq ($(HOST_PLATFORM),macos)
	JAVA_SHARED_LIB := target/debug/libfastlowess_java.dylib
	JAVA_EXPORT_SCAN := nm -gU $(JAVA_SHARED_LIB)
else
	JAVA_SHARED_LIB := target/debug/libfastlowess_java.so
	JAVA_EXPORT_SCAN := nm -D $(JAVA_SHARED_LIB)
endif

.PHONY: default dev clean docs

default:
	@cargo build -q -p $(JAVA_PKG)
	@cd $(JAVA_DIR) && mvn -q package
	@echo "$(JAVA_PKG) built successfully!"

dev:
	@echo "Running $(JAVA_PKG) checks..."
	@echo "=============================================================================="
	@echo "1. Rust formatting & linting..."
	@echo "=============================================================================="
	@cargo fmt -p $(JAVA_PKG) -- --check
	@cargo clippy -q -p $(JAVA_PKG) --all-targets -- -D warnings
	@echo "=============================================================================="
	@echo "2. Java lint (Checkstyle)..."
	@echo "=============================================================================="
	@CS_JAR="$$HOME/.cache/checkstyle/$(CHECKSTYLE_JAR)"; \
		if [ ! -f "$$CS_JAR" ]; then \
			echo "Checkstyle not found. Downloading v$(CHECKSTYLE_VERSION)..."; \
			mkdir -p "$$(dirname "$$CS_JAR")"; \
			curl -fsSL -o "$$CS_JAR" "$(CHECKSTYLE_URL)"; \
		fi; \
		java -jar "$$CS_JAR" -c $(JAVA_DIR)/checkstyle.xml $(JAVA_DIR)/src/main/java $(JAVA_DIR)/src/test/java
	@echo "=============================================================================="
	@echo "3. Building..."
	@echo "=============================================================================="
	@cargo build -q -p $(JAVA_PKG)
	@echo "=============================================================================="
	@echo "4. Verifying library exports..."
	@echo "=============================================================================="
	@$(JAVA_EXPORT_SCAN) 2>/dev/null | grep -q Java_fastlowess_NativeBridge_lowessNew || \
		(echo "Error: Java_fastlowess_NativeBridge_lowessNew not exported"; exit 1)
	@$(JAVA_EXPORT_SCAN) 2>/dev/null | grep -q Java_fastlowess_NativeBridge_streamingNew || \
		(echo "Error: Java_fastlowess_NativeBridge_streamingNew not exported"; exit 1)
	@$(JAVA_EXPORT_SCAN) 2>/dev/null | grep -q Java_fastlowess_NativeBridge_onlineNew || \
		(echo "Error: Java_fastlowess_NativeBridge_onlineNew not exported"; exit 1)
	@echo "All exports verified!"
	@echo "=============================================================================="
	@echo "5. Maven build & tests..."
	@echo "=============================================================================="
	@cd $(JAVA_DIR) && mvn -q clean test
	@echo "=============================================================================="
	@echo "6. Checking for outdated Maven plugins & dependencies..."
	@echo "=============================================================================="
	@cd $(JAVA_DIR) && mvn -q versions:display-plugin-updates
	@cd $(JAVA_DIR) && mvn -q versions:display-dependency-updates
	@echo "=============================================================================="
	@echo "Doc snippet tests (java)..."
	@echo "=============================================================================="
	@$(PYTHON) dev/check_links.py --lang java
	@$(PYTHON) dev/verify_snippets.py --lang java --timeout 120 --update-outputs
	@$(PYTHON) dev/update_changelogs.py java
	@$(PYTHON) dev/add-readme-to-docs.py $(JAVA_DIR)
	@echo "$(JAVA_PKG) checks completed successfully!"

clean:
	@echo "Cleaning $(JAVA_PKG)..."
	@cargo clean -p $(JAVA_PKG)
	@cd $(JAVA_DIR) && mvn -q clean
	@git clean -fdX -- $(JAVA_DIR)
	@echo "$(JAVA_PKG) clean complete!"

docs:
	@echo "Building $(JAVA_PKG) docs (Antora)..."
	@$(PYTHON) dev/add-readme-to-docs.py $(JAVA_DIR)
	@cd $(JAVA_DIR)/docs-site && npx --yes -p @antora/cli@3 -p @antora/site-generator@3 antora antora-playbook.yml
	@echo "Docs built at $(JAVA_DIR)/docs-site/public/index.html"