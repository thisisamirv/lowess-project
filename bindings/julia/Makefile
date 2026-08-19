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

TEMP ?= /tmp
ifeq ($(OS),Windows_NT)
	TEMP := /tmp
endif

JL_PKG      := fastlowess-jl
JL_DIR      := bindings/julia
JL_TEST_DIR := bindings/julia/tests

ifeq ($(HOST_PLATFORM),windows)
	JL_SHARED_LIB  := target/release/fastlowess_jl.dll
	JL_EXPORT_SCAN := objdump -p $(JL_SHARED_LIB)
else ifeq ($(HOST_PLATFORM),macos)
	JL_SHARED_LIB  := target/release/libfastlowess_jl.dylib
	JL_EXPORT_SCAN := nm -gU $(JL_SHARED_LIB)
else
	JL_SHARED_LIB  := target/release/libfastlowess_jl.so
	JL_EXPORT_SCAN := nm -D $(JL_SHARED_LIB)
endif

JL_SHARED_LIB_ABS := $(abspath $(JL_SHARED_LIB))

.PHONY: default clean _julia_checks_internal

default:
	@echo "Running $(JL_PKG) checks..."
	@PROJECT_TOML="$(JL_DIR)/julia/Project.toml"; \
	BACKUP="$$PROJECT_TOML.bak"; \
	cp "$$PROJECT_TOML" "$$BACKUP"; \
	trap 'if [ -f "$$BACKUP" ]; then mv "$$BACKUP" "$$PROJECT_TOML"; echo "=============================================================================="; echo "Restored $$PROJECT_TOML"; fi' EXIT; \
	echo "=============================================================================="; \
	echo "0. Local environment setup (relaxing JLL constraint)..."; \
	echo "=============================================================================="; \
	LATEST=$$(julia -e 'using Pkg; Pkg.activate(temp=true); try Pkg.add("fastlowess_jll"); pkgs = Pkg.dependencies(); v = [p.version for (_,p) in pkgs if p.name == "fastlowess_jll"]; isempty(v) ? print("1.0.0") : print(first(v)) catch; print("1.0.0") end' | cut -d'+' -f1); \
	CURRENT=$$(grep "^version =" "$$PROJECT_TOML" | cut -d"\"" -f2); \
	julia -e "using TOML; path = \"$$PROJECT_TOML\"; p = TOML.parsefile(path); if haskey(get(p, \"compat\", Dict()), \"fastlowess_jll\"); p[\"compat\"][\"fastlowess_jll\"] = \"$$LATEST, $$CURRENT\"; open(path, \"w\") do io; TOML.print(io, p); end; end"; \
	echo "Modified $$PROJECT_TOML (fastlowess_jll = \"$$LATEST, $$CURRENT\") using TOML parser."; \
	"$(MAKE)" -f $(THIS_MK) _julia_checks_internal

_julia_checks_internal:
	@echo "=============================================================================="
	@echo "0. Commit hash update..."
	@echo "=============================================================================="
	@git fetch origin main 2>/dev/null || true
	@COMMIT=$$(git rev-parse origin/main 2>/dev/null) && \
		COMMIT="$$COMMIT" $(PYTHON) -c 'from pathlib import Path; import os, re; path = Path("dev/build_tarballs_julia.jl"); text = path.read_text(encoding="utf-8"); commit = os.environ["COMMIT"]; new_text, _ = re.subn(r"GitSource\\(\"[^\"]*\",\\s*\"[a-f0-9]+\"\\)", f"GitSource(\"https://github.com/thisisamirv/lowess-project.git\", \"{commit}\")", text); path.write_text(new_text, encoding="utf-8")' && \
		echo "Commit: $$COMMIT"
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cd $(JL_DIR) && cargo fmt -- --check
	@echo "Formatting complete!"
	@echo "=============================================================================="
	@echo "2. Linting..."
	@echo "=============================================================================="
	@cd $(JL_DIR) && cargo clippy -q --all-targets -- -D warnings
	@echo "Linting Julia files..."
	@julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format(["bindings/julia/julia", "$(JL_TEST_DIR)"], verbose=true, overwrite=true)' || true
	@julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format(["bindings/julia/julia", "$(JL_TEST_DIR)"], verbose=true, overwrite=false) ? exit(0) : exit(1)'
	@echo "=============================================================================="
	@echo "3. Building..."
	@echo "=============================================================================="
	@cd $(JL_DIR) && cargo build -q --release
	@cd $(JL_DIR) && RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps
	@echo "=============================================================================="
	@echo "4. Testing Rust library..."
	@echo "=============================================================================="
	@cd $(JL_DIR) && cargo test -q
	@echo "=============================================================================="
	@echo "5. Verifying library exports..."
	@echo "=============================================================================="
	@$(JL_EXPORT_SCAN) 2>/dev/null | grep -q jl_lowess_new || \
		(echo "Error: jl_lowess_new not exported"; exit 1)
	@$(JL_EXPORT_SCAN) 2>/dev/null | grep -q jl_streaming_lowess_new || \
		(echo "Error: jl_streaming_lowess_new not exported"; exit 1)
	@$(JL_EXPORT_SCAN) 2>/dev/null | grep -q jl_online_lowess_new || \
		(echo "Error: jl_online_lowess_new not exported"; exit 1)
	@$(JL_EXPORT_SCAN) 2>/dev/null | grep -q jl_lowess_free_result || \
		(echo "Error: jl_lowess_free_result not exported"; exit 1)
	@echo "All exports verified!"
	@echo "=============================================================================="
	@echo "5b. ABI size check (limit: 5 MB)..."
	@echo "=============================================================================="
	@[ $$($(STAT_SIZE_CMD) $(JL_SHARED_LIB)) -le 5242880 ] || \
		(echo "Error: $(JL_SHARED_LIB) exceeds 5 MB ABI size limit"; exit 1)
	@echo "ABI size OK."
	@echo "=============================================================================="
	@echo "6. Testing Julia bindings..."
	@echo "=============================================================================="
	@export FASTLOWESS_LIB=$(JL_SHARED_LIB_ABS) && \
	julia --project=$(JL_DIR)/julia -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()' && \
	julia --project=$(JL_DIR)/julia $(JL_TEST_DIR)/test_FastLOWESS.jl
	@echo "=============================================================================="
	@echo "7. Aqua.jl package quality..."
	@echo "=============================================================================="
	@export FASTLOWESS_LIB=$(JL_SHARED_LIB_ABS) && \
	julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path="$(JL_DIR)/julia"); \
		Pkg.add("Aqua"); using Aqua, FastLOWESS; \
		Aqua.test_all(FastLOWESS; ambiguities=false, stale_deps=(ignore=[:Aqua],))'
	@echo "=============================================================================="
	@echo "8. JET.jl type-inference check..."
	@echo "=============================================================================="
	@export FASTLOWESS_LIB=$(JL_SHARED_LIB_ABS) && \
	julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path="$(JL_DIR)/julia"); \
		Pkg.add("JET"); using JET, FastLOWESS; \
		report = JET.report_package(FastLOWESS; target_modules=(FastLOWESS,)); \
		show(stderr, MIME("text/plain"), report); println(stderr); \
		if length(JET.get_reports(report)) > 0; @error "JET found type errors"; exit(1); end'
	@echo "$(JL_PKG) checks completed successfully!"
	@echo ""
	@echo "To use in Julia:"
	@echo "  julia> using Pkg"
	@echo "  julia> Pkg.develop(path=\"$(JL_DIR)/julia\")"
	@echo "  julia> using FastLOWESS"

clean:
	@echo "Cleaning $(JL_PKG)..."
	@cargo clean -p $(JL_PKG)
	@git clean -fdX -- $(JL_DIR)
	@echo "$(JL_PKG) clean complete!"
