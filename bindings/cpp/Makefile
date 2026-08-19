THIS_MK := $(lastword $(MAKEFILE_LIST))

.SHELLFLAGS := -ec

UNAME_S := $(shell uname -s)

ifeq ($(OS),Windows_NT)
	HOST_PLATFORM  := windows
	PATH_SEPARATOR := :
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

TEMP ?= /tmp
ifeq ($(OS),Windows_NT)
	TEMP := /tmp
endif

CPP_PKG           := fastlowess-cpp
CPP_DIR           := bindings/cpp
CPP_CARGO_PROFILE := --profile release-c
CPP_LIBRARY_DIR   := target/release-c
CPP_CARGO_TARGET  :=

ifeq ($(OS),Windows_NT)
	_CPP_GCC_MACHINE := $(shell gcc -dumpmachine 2>/dev/null)
	ifneq ($(findstring mingw,$(_CPP_GCC_MACHINE)),)
		_CPP_WIN_TOOLCHAIN := mingw
		CPP_CARGO_TARGET   := --target x86_64-pc-windows-gnu
		CPP_LIBRARY_DIR    := target/x86_64-pc-windows-gnu/release-c
	else
		_CPP_WIN_TOOLCHAIN := msvc
		CPP_CARGO_TARGET   := --target x86_64-pc-windows-msvc
		CPP_LIBRARY_DIR    := target/x86_64-pc-windows-msvc/release-c
	endif
endif

ifeq ($(HOST_PLATFORM),windows)
	CPP_SHARED_LIB  := $(CPP_LIBRARY_DIR)/fastlowess_cpp.dll
	CPP_EXPORT_SCAN := objdump -p $(CPP_SHARED_LIB)
	ifeq ($(_CPP_WIN_TOOLCHAIN),mingw)
		CPP_TEST_LIB := $(CPP_LIBRARY_DIR)/libfastlowess_cpp.dll.a
		_HAVE_MINGW32_MAKE := $(shell mingw32-make --version 2>/dev/null | head -c 3)
		ifneq ($(_HAVE_MINGW32_MAKE),)
			CPP_CMAKE_GENERATOR := -G "MinGW Makefiles"
			CPP_TEST_BUILD      := cmake --build .
			CPP_TEST_RUN        := cp -f "$(abspath $(CPP_SHARED_LIB))" . && ./test_fastlowess_suite.exe
		else
			CPP_CMAKE_GENERATOR :=
			CPP_TEST_BUILD      := cmake --build . --config Release
			CPP_TEST_RUN        := cp -f "$(abspath $(CPP_SHARED_LIB))" Release/ && ./Release/test_fastlowess_suite.exe
		endif
	else
		CPP_TEST_LIB        := $(CPP_LIBRARY_DIR)/fastlowess_cpp.lib
		CPP_CMAKE_GENERATOR :=
		CPP_TEST_BUILD      := cmake --build . --config Release
		CPP_TEST_RUN        := cp -f "$(abspath $(CPP_SHARED_LIB))" Release/ && ./Release/test_fastlowess_suite.exe
	endif
else ifeq ($(HOST_PLATFORM),macos)
	CPP_SHARED_LIB      := $(CPP_LIBRARY_DIR)/libfastlowess_cpp.dylib
	CPP_TEST_LIB        := $(CPP_SHARED_LIB)
	CPP_EXPORT_SCAN     := nm -gU $(CPP_SHARED_LIB)
	CPP_CMAKE_GENERATOR :=
	CPP_TEST_BUILD      := make
	CPP_TEST_RUN        := ./test_fastlowess_suite
else
	CPP_SHARED_LIB      := $(CPP_LIBRARY_DIR)/libfastlowess_cpp.so
	CPP_TEST_LIB        := $(CPP_SHARED_LIB)
	CPP_EXPORT_SCAN     := nm -D $(CPP_SHARED_LIB)
	CPP_CMAKE_GENERATOR :=
	CPP_TEST_BUILD      := make
	CPP_TEST_RUN        := ./test_fastlowess_suite
endif

CPP_LIBRARY_DIR_ABS := $(abspath $(CPP_LIBRARY_DIR))
CPP_TEST_LIB_ABS    := $(abspath $(CPP_TEST_LIB))

TEMP ?= /tmp

.PHONY: default clean

default:
	@echo "Running $(CPP_PKG) checks..."
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(CPP_PKG) -- --check
	@echo "=============================================================================="
	@echo "2. Linting & Building..."
	@echo "=============================================================================="
	@cargo clippy -q -p $(CPP_PKG) --all-targets $(CPP_CARGO_TARGET) -- -D warnings
	@echo "Linting C++ files..."
	@if command -v clang-tidy >/dev/null 2>&1; then \
		clang_tidy_log="$(TEMP)/clang-tidy-cpp.log"; \
		clang-tidy --config-file=bindings/cpp/.clang-tidy bindings/cpp/include/fastlowess.hpp $(CPP_DIR)/tests/test_fastlowess.cpp -- -I bindings/cpp/include -std=c++17 > "$$clang_tidy_log" 2>&1; \
		clang_tidy_status=$$?; \
		grep -Ev '^(\[[0-9]+/[0-9]+\] Processing file |[0-9]+ warnings generated\.|Suppressed [0-9]+ warnings \([0-9]+ in non-user code\)\.|Use -header-filter=\.\* or leave it as default to display errors from all non-system headers\.|Use -system-headers to display errors from system headers as well\.)' "$$clang_tidy_log" || true; \
		rm -f "$$clang_tidy_log"; \
		if [ $$clang_tidy_status -ne 0 ]; then \
			echo "C++ linting failed"; \
			exit $$clang_tidy_status; \
		fi; \
	else \
		echo "clang-tidy not found. Skipping C++ lint pass."; \
	fi
	@echo "Running cppcheck..."
	@if command -v cppcheck >/dev/null 2>&1; then \
		cppcheck_log="$(TEMP)/cppcheck-cpp.log"; \
		if cppcheck --error-exitcode=1 --enable=warning,performance,portability \
			--suppress=missingInclude --suppress=missingIncludeSystem \
			-I $(CPP_DIR)/include \
			$(CPP_DIR)/include/fastlowess.hpp $(CPP_DIR)/tests/test_fastlowess.cpp > "$$cppcheck_log" 2>&1; then \
			cat "$$cppcheck_log"; \
		else \
			cppcheck_status=$$?; \
			cat "$$cppcheck_log"; \
			if grep -q "Failed to load library configuration file 'std.cfg'" "$$cppcheck_log"; then \
				echo "cppcheck installation is broken. Skipping static analysis pass."; \
			else \
				rm -f "$$cppcheck_log"; \
				exit $$cppcheck_status; \
			fi; \
		fi; \
		rm -f "$$cppcheck_log"; \
	else \
		echo "cppcheck not found. Skipping static analysis pass."; \
	fi
	@if [ -n "$(CPP_CARGO_TARGET)" ]; then \
		_rust_target=$$(echo "$(CPP_CARGO_TARGET)" | sed 's/--target //'); \
		rustup target add "$$_rust_target" 2>/dev/null || true; \
	fi
	@cargo build -q -p $(CPP_PKG) $(CPP_CARGO_PROFILE) $(CPP_CARGO_TARGET)
	@echo "C header generated at $(CPP_DIR)/include/fastlowess.h"
	@echo "=============================================================================="
	@echo "2b. cbindgen idempotency check..."
	@echo "=============================================================================="
	@if ! command -v cbindgen >/dev/null 2>&1; then \
		echo "cbindgen not found. Installing latest version..."; \
		cargo install cbindgen --force; \
	fi
	@cbindgen --config $(CPP_DIR)/cbindgen.toml --crate $(CPP_PKG) --output $(TEMP)/fastlowess_new.h 2>/dev/null && \
		diff -q $(CPP_DIR)/include/fastlowess.h $(TEMP)/fastlowess_new.h > /dev/null || \
		(echo "Error: fastlowess.h is stale — run 'cargo build -p $(CPP_PKG) $(CPP_CARGO_PROFILE) $(CPP_CARGO_TARGET)' to regenerate"; exit 1)
	@echo "cbindgen header is up-to-date."
	@echo "=============================================================================="
	@echo "2c. Symbol export verification..."
	@echo "=============================================================================="
	@$(CPP_EXPORT_SCAN) 2>/dev/null | grep -q cpp_lowess_new || \
		(echo "Error: cpp_lowess_new not exported"; exit 1)
	@$(CPP_EXPORT_SCAN) 2>/dev/null | grep -q cpp_streaming_new || \
		(echo "Error: cpp_streaming_new not exported"; exit 1)
	@$(CPP_EXPORT_SCAN) 2>/dev/null | grep -q cpp_online_new || \
		(echo "Error: cpp_online_new not exported"; exit 1)
	@echo "All C++ exports verified."
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@rm -rf $(CPP_DIR)/tests/build
	@mkdir -p $(CPP_DIR)/tests/build
	@cd $(CPP_DIR)/tests/build && cmake $(CPP_CMAKE_GENERATOR) -DFASTLOWESS_LIB="$(CPP_TEST_LIB_ABS)" -DFASTLOWESS_LIB_DIR="$(CPP_LIBRARY_DIR_ABS)" .. && $(CPP_TEST_BUILD) && $(CPP_TEST_RUN)
	@echo "=============================================================================="
	@echo "3b. Valgrind memory check..."
	@echo "=============================================================================="
	@if [ "$(HOST_PLATFORM)" != "linux" ]; then \
		echo "Valgrind: skipped on $(HOST_PLATFORM)."; \
	elif ! command -v valgrind >/dev/null 2>&1; then \
		echo "Valgrind not found. Skipping memory check."; \
	else \
		valgrind --leak-check=full --error-exitcode=1 --quiet \
		$(CPP_DIR)/tests/build/test_fastlowess_suite 2>&1 || \
		(echo "Error: Valgrind detected memory errors"; exit 1); \
		echo "Valgrind: no leaks."; \
	fi
	@echo "$(CPP_PKG) checks completed successfully!"

clean:
	@echo "Cleaning $(CPP_PKG)..."
	@cargo clean -p $(CPP_PKG)
	@git clean -fdX -- $(CPP_DIR)
	@echo "$(CPP_PKG) clean complete!"
