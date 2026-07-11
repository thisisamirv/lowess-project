"""Extract fenced code blocks from docs/ and run them to verify correctness.

Each snippet is prefixed with language-specific boilerplate that provides common
sample data and imports, so fragment-style doc examples can execute end-to-end.

Usage
-----
    python dev/verify_snippets.py                    # run all supported languages
    python dev/verify_snippets.py --lang python      # Python only
    python dev/verify_snippets.py --lang nodejs      # Node.js only
    python dev/verify_snippets.py --lang julia       # Julia only
    python dev/verify_snippets.py --lang r           # R only
    python dev/verify_snippets.py --lang wasm        # WebAssembly only
    python dev/verify_snippets.py --lang rust        # Rust only
    python dev/verify_snippets.py --lang cpp         # C++ only
    python dev/verify_snippets.py --file docs/api/python.md
    python dev/verify_snippets.py --dry-run          # list snippets, don't run
    python dev/verify_snippets.py --verbose          # show snippet source on failure
    python dev/verify_snippets.py --output out.json  # also write JSON report
    python dev/verify_snippets.py --timeout 60       # per-snippet timeout (seconds)
    python dev/verify_snippets.py --stop-on-fail     # exit after first failure
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

# ---------------------------------------------------------------------------
# Terminal colours (disabled on non-TTY or Windows without colour support)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.name != "nt" or os.environ.get("FORCE_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def green(t: str) -> str:
    return _c("32", t)


def red(t: str) -> str:
    return _c("31", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def bold(t: str) -> str:
    return _c("1", t)


# ---------------------------------------------------------------------------
# Python executable detection (prefer venv where fastlowess is installed)
# ---------------------------------------------------------------------------

_PYTHON_BIN: str = sys.executable  # may be replaced in main()


def _find_python_with_fastlowess() -> str:
    """Return the best Python executable that has fastlowess installed."""
    candidates = [
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",  # Windows root venv
        REPO_ROOT / ".venv" / "bin" / "python",  # Unix root venv
        REPO_ROOT / "bindings" / "python" / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / "bindings" / "python" / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            try:
                r = subprocess.run(
                    [str(c), "-c", "import fastlowess"],
                    capture_output=True,
                    timeout=10,
                )
                if r.returncode == 0:
                    return str(c)
            except Exception:
                pass
    return sys.executable


# ---------------------------------------------------------------------------
# Language-specific boilerplate injected before every snippet
# ---------------------------------------------------------------------------

# Tab labels in the docs that map to each runner
_TAB_ALIASES: dict[str, set[str]] = {
    "python": {"Python"},
    "julia": {"Julia"},
    "nodejs": {"Node.js"},
    # These are not run by default (need external toolchain)
    "wasm": {"WebAssembly"},
    "r": {"R"},
    "cpp": {"C++"},
    "rust": {
        "Rust",
        "Rust (fastLowess)",
        "lowess (no_std compatible)",
        "fastLowess (parallel)",
    },
}

# Code-block language tags for each runner
_LANG_TAGS: dict[str, set[str]] = {
    "python": {"python"},
    "julia": {"julia"},
    "nodejs": {"javascript", "js"},
    "wasm": {"javascript", "js"},
    "r": {"r"},
    "cpp": {"cpp", "c++"},
    "rust": {"rust"},
}

_PYTHON_PREAMBLE = ""

_JULIA_PREAMBLE = ""

_NODEJS_PREAMBLE = ""

_R_PREAMBLE = textwrap.dedent("""\
    suppressMessages({{
        .libPaths(c(
            normalizePath(file.path(
                Sys.getenv("LOWESS_REPO_ROOT", "{repo_root}"),
                "bindings", "r", ".r-lib"
            ), mustWork = FALSE),
            .libPaths()
        ))
        library(rfastlowess)
    }})
    suppressWarnings(pdf(NULL))
    plot.new()
""").format(repo_root=str(REPO_ROOT).replace("\\", "/"))


_WASM_PKG_DIR = REPO_ROOT / "bindings" / "wasm" / "pkg"

_WASM_PREAMBLE = ""

# Fixed temp project for Rust snippets (reuses Cargo artifacts)
_RUST_SNIPPET_DIR = REPO_ROOT / "target" / "doc-snippet-runner"

PREAMBLES: dict[str, str] = {
    "python": _PYTHON_PREAMBLE,
    "julia": _JULIA_PREAMBLE,
    "nodejs": _NODEJS_PREAMBLE,
    "r": _R_PREAMBLE,
    "wasm": _WASM_PREAMBLE,
}

# ---------------------------------------------------------------------------
# Snippet data class
# ---------------------------------------------------------------------------


@dataclass
class Snippet:
    file: Path
    line: int  # 1-based line number of the opening fence
    lang_tag: str  # code-block language tag (e.g. "python")
    tab: Optional[str]  # nearest === "Tab" label, or None
    code: str

    @property
    def runner(self) -> Optional[str]:
        """Return which runner handles this snippet, or None to skip."""
        for runner, tags in _LANG_TAGS.items():
            if self.lang_tag.lower() in tags:
                # For JS: distinguish Node.js from WASM by tab label
                if runner in ("nodejs", "wasm"):
                    if self.tab in _TAB_ALIASES["wasm"]:
                        return "wasm"
                    if self.tab in _TAB_ALIASES["nodejs"]:
                        return "nodejs"
                    # No tab: fall back to nodejs if no WASM markers
                    if (
                        "fastlowess-wasm" not in self.code
                        and "fastlowess_wasm"
                        not in self.code  # catches require('./fastlowess_wasm.js')
                        and "import {" not in self.code[:80]
                    ):
                        return "nodejs"
                    return "wasm"
                return runner
        return None

    @property
    def label(self) -> str:
        tab = f" [{self.tab}]" if self.tab else ""
        return f"{self.file.relative_to(REPO_ROOT)}:{self.line}{tab}"


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------

_TAB_RE = re.compile(r'^[ \t]*===\s+"([^"]+)"', re.MULTILINE)
_FENCE_RE = re.compile(r"^([ \t]*)```(\w+)", re.MULTILINE)


def extract_snippets(md_file: Path) -> List[Snippet]:
    """Extract all fenced code blocks from a markdown file."""
    text = md_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    result: List[Snippet] = []

    current_tab: Optional[str] = None
    i = 0
    while i < len(lines):
        line = lines[i]

        # Track tab context
        m = _TAB_RE.match(line)
        if m:
            current_tab = m.group(1)
            i += 1
            continue

        # Detect fence opening: optional leading whitespace then ```lang
        m = re.match(r"^([ \t]*)```(\w+)\s*$", line)
        if m:
            fence_indent = m.group(1)
            lang_tag = m.group(2)
            start_line = i + 1  # 1-based
            code_lines: List[str] = []
            i += 1
            while i < len(lines):
                close = lines[i]
                # Closing fence: same indent + ```
                if re.match(r"^" + re.escape(fence_indent) + r"```\s*$", close):
                    i += 1
                    break
                # Strip the common fence indent
                stripped = (
                    close[len(fence_indent) :]
                    if close.startswith(fence_indent)
                    else close
                )
                code_lines.append(stripped)
                i += 1
            code = "\n".join(code_lines)
            result.append(
                Snippet(
                    file=md_file,
                    line=start_line,
                    lang_tag=lang_tag,
                    tab=current_tab,
                    code=code,
                )
            )
            # A tab label covers only the next block (reset after capture)
            current_tab = None
            continue

        # Reset tab on section headers or dividers
        if line.startswith("#") or line.strip() == "---":
            current_tab = None

        i += 1

    return result


def should_skip(snippet: Snippet, runner: str) -> Optional[str]:
    """Return a skip reason string, or None if the snippet should be run."""
    code = snippet.code

    # MkDocs file-include directives are not runnable
    if "--8<--" in code:
        return "file-include (--8<--)"

    # Empty or whitespace-only
    if not code.strip():
        return "empty"

    # Skip snippets that reference variables or packages we can't supply
    if runner == "python":
        # Genomic-specific heavy I/O (would need real files)
        if any(s in code for s in ["read_csv", "open(", "glob(", "argparse"]):
            return "file I/O"
        # Lines that are obviously just output examples (no executable Python)
        if not any(c in code for c in ["=", "(", "import", "print"]):
            return "no executable statements"
        # Large synthetic datasets that exceed the per-snippet timeout
        if re.search(r"total_points\s*=\s*[1-9][0-9]{4,}", code):
            return "large synthetic dataset (too slow for CI)"
        # Snippets that use fastlowess without importing it first
        if re.search(
            r"\bfl\b|\bfastlowess\b|\bLowess\b|\bStreamingLowess\b|\bOnlineLowess\b",
            code,
        ):
            if not re.search(r"\bimport\b.*fastlowess|\bfrom\b.*fastlowess", code):
                return "fastlowess not imported (snippet is not self-contained)"

    if runner == "julia":
        # Skip package-management / installation snippets
        if re.search(r"\bPkg\.(add|develop|clone|rm|pin)\s*\(", code):
            return "Pkg management snippet"
        # Skip API method-signature snippets: type-annotated keyword arguments
        # (e.g. `; custom_weights::Union{...} = nothing`) are valid only in
        # function *definitions*, not call sites — Julia rejects them as calls.
        if re.search(r";\s*\w+::", code, re.DOTALL):
            return "Julia method signature (keyword arg with type annotation — not callable)"
    if runner == "nodejs":
        # TypeScript-only syntax (type annotations)
        if ": SmoothOptions" in code or ": LowessResult" in code:
            return "TypeScript (not Node.js)"
        # Snippets must load fastlowess themselves (no preamble)
        if not re.search(r"require\s*\(", code):
            return "no require() — snippet must load fastlowess itself"

    if runner == "r":
        # Skip install/devtools snippets
        if re.search(r"\binstall\.packages\b|\bdevtools::", code):
            return "package installation snippet"
        # Skip if no actual R statements (e.g., pure output blocks)
        if not any(c in code for c in ["<-", "=", "(", "library"]):
            return "no executable R statements"
        # Multi-dim input (x2d/x3d) requires a package rebuild to work correctly
        if re.search(r"\bx2d\b|\bx3d\b|\bdimensions\s*=\s*[23]\b", code):
            return "multi-dim R (needs package rebuild)"
        # API signature snippets that use R's '...' outside a function definition
        # (e.g. 'model <- Lowess(...)') — not valid in a script context.
        if re.search(r"\(\s*\.\.\.\s*\)", code):
            return "R API signature with ... (not runnable outside function)"

    if runner == "wasm":
        # Skip any ES-module import syntax — wasm runner uses CJS require().
        # Catches `import { X }`, `import init, { X }`, `import X from ...`, etc.
        if re.search(r"^import\b", code, re.MULTILINE) or "await init(" in code:
            return "ES module import (not supported in CJS runner)"
        # Snippets must load the WASM package themselves (no preamble)
        if not re.search(r"require\s*\(", code):
            return "no require() — snippet must load the WASM package itself"

    if runner == "rust":
        # Without a structural wrapper, only complete programs (with fn main) compile
        if not re.search(r"\bfn\s+main\s*\(", code):
            return "fragment — no fn main (not a standalone Rust program)"
        # Skip snippets that look like TOML config (Cargo.toml examples)
        if code.strip().startswith("[") and "=" in code and "fn " not in code:
            return "TOML/config snippet"
        # Backend::GPU requires the optional gpu feature flag
        if re.search(r"\bBackend\s*::\s*GPU\b", code):
            return "requires gpu feature flag (not enabled in snippet workspace)"
        # cross_validate / KFold are not in the stable public API
        if re.search(r"\bcross_validate\b|\bKFold\b", code):
            return "cross_validate/KFold not in stable public API"

    if runner == "cpp":
        # Without a structural wrapper, only complete programs (with int main) compile
        if not re.search(r"\bint\s+main\s*\(", code):
            return "fragment — no int main (not a standalone C++ program)"

    return None


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    snippet: Snippet
    runner: str
    skipped: bool = False
    skip_reason: str = ""
    passed: bool = False
    duration: float = 0.0
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1


def run_python(snippet: Snippet, timeout: int) -> RunResult:
    code = PREAMBLES["python"] + snippet.code
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name
    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [_PYTHON_BIN, tmp],
            capture_output=True,
            timeout=timeout,
            text=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="python",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="python",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )
    finally:
        os.unlink(tmp)


def run_julia(snippet: Snippet, timeout: int) -> RunResult:
    code = PREAMBLES["julia"] + snippet.code
    with tempfile.NamedTemporaryFile(
        suffix=".jl", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name

    julia_bin = _find_exe("julia")
    if julia_bin is None:
        os.unlink(tmp)
        return RunResult(
            snippet=snippet,
            runner="julia",
            skipped=True,
            skip_reason="julia not found in PATH",
        )

    # Find the Julia project for the bindings
    julia_project = REPO_ROOT / "bindings" / "julia" / "julia"
    env = {**os.environ}
    if julia_project.exists():
        env["JULIA_PROJECT"] = str(julia_project)

    # Prefer the locally-built library over a potentially-stale JLL artifact
    if "FASTLOWESS_LIB" not in env:
        _jl_lib_name = (
            "fastlowess_jl.dll"
            if sys.platform == "win32"
            else (
                "libfastlowess_jl.dylib"
                if sys.platform == "darwin"
                else "libfastlowess_jl.so"
            )
        )
        _local_lib = REPO_ROOT / "target" / "release" / _jl_lib_name
        if _local_lib.exists():
            env["FASTLOWESS_LIB"] = str(_local_lib)

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [julia_bin, "--startup-file=no", tmp],
            capture_output=True,
            timeout=timeout,
            text=True,
            env=env,
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="julia",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="julia",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )
    finally:
        os.unlink(tmp)


def _ensure_nodejs_selflink(nodejs_dir: Path) -> None:
    """Create node_modules/fastlowess shim so require('fastlowess') resolves locally.

    npm does not self-install the package, so we create a minimal shim that
    re-exports the local binding.  The shim is idempotent and safe to leave in
    node_modules alongside the real platform packages.
    """
    nm_fastlowess = nodejs_dir / "node_modules" / "fastlowess"
    if nm_fastlowess.exists():
        return
    nm_fastlowess.mkdir(parents=True, exist_ok=True)
    (nm_fastlowess / "index.js").write_text(
        "module.exports = require('../../');\n", encoding="utf-8"
    )
    (nm_fastlowess / "package.json").write_text(
        '{"name":"fastlowess","main":"index.js","version":"0.0.0"}\n',
        encoding="utf-8",
    )


def _ensure_wasm_selflink(wasm_pkg_dir: Path) -> None:
    """Create node_modules/fastlowess-wasm shim so require('fastlowess-wasm') resolves locally.

    The wasm pkg/ directory IS the fastlowess-wasm package, but npm doesn't
    self-install it, so we create a shim that re-exports from the pkg root.
    """
    nm_wasm = wasm_pkg_dir / "node_modules" / "fastlowess-wasm"
    if nm_wasm.exists():
        return
    nm_wasm.mkdir(parents=True, exist_ok=True)
    (nm_wasm / "index.js").write_text(
        "module.exports = require('../../');\n", encoding="utf-8"
    )
    (nm_wasm / "package.json").write_text(
        '{"name":"fastlowess-wasm","main":"index.js","version":"0.0.0"}\n',
        encoding="utf-8",
    )


def run_nodejs(snippet: Snippet, timeout: int) -> RunResult:
    code = snippet.code

    node_bin = _find_exe("node")
    if node_bin is None:
        return RunResult(
            snippet=snippet,
            runner="nodejs",
            skipped=True,
            skip_reason="node not found in PATH",
        )

    # Write the temp file INSIDE the nodejs binding directory so that Node.js
    # module resolution (file-relative) can find node_modules/fastlowess there.
    nodejs_dir = REPO_ROOT / "bindings" / "nodejs"
    cwd = str(nodejs_dir) if nodejs_dir.exists() else str(REPO_ROOT)

    if nodejs_dir.exists():
        _ensure_nodejs_selflink(nodejs_dir)

    import uuid

    tmp_name = f"_snippet_{uuid.uuid4().hex}.js"
    tmp = str(Path(cwd) / tmp_name)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [node_bin, tmp],
            capture_output=True,
            timeout=timeout,
            text=True,
            cwd=cwd,
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="nodejs",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="nodejs",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )
    finally:
        os.unlink(tmp)


def run_r(snippet: Snippet, timeout: int) -> RunResult:
    code = PREAMBLES["r"] + snippet.code
    with tempfile.NamedTemporaryFile(
        suffix=".R", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name

    rscript = _find_exe("Rscript")
    if rscript is None:
        os.unlink(tmp)
        return RunResult(
            snippet=snippet,
            runner="r",
            skipped=True,
            skip_reason="Rscript not found in PATH",
        )

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [rscript, "--vanilla", tmp],
            capture_output=True,
            timeout=timeout,
            text=True,
            env={**os.environ, "LOWESS_REPO_ROOT": str(REPO_ROOT)},
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="r",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="r",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )
    finally:
        os.unlink(tmp)


def run_wasm(snippet: Snippet, timeout: int) -> RunResult:
    # WASM snippets run in Node.js with the pre-built wasm pkg
    if not _WASM_PKG_DIR.exists():
        return RunResult(
            snippet=snippet,
            runner="wasm",
            skipped=True,
            skip_reason="bindings/wasm/pkg/ not built (run 'make wasm' first)",
        )

    node_bin = _find_exe("node")
    if node_bin is None:
        return RunResult(
            snippet=snippet,
            runner="wasm",
            skipped=True,
            skip_reason="node not found in PATH",
        )

    # Write the temp file INSIDE _WASM_PKG_DIR so require('./fastlowess_wasm.js') resolves
    _ensure_wasm_selflink(_WASM_PKG_DIR)

    import uuid

    tmp_name = f"_snippet_{uuid.uuid4().hex}.js"
    tmp = str(_WASM_PKG_DIR / tmp_name)
    code = snippet.code
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [node_bin, tmp],
            capture_output=True,
            timeout=timeout,
            text=True,
            cwd=str(_WASM_PKG_DIR),
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="wasm",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="wasm",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )
    finally:
        os.unlink(tmp)


def _ensure_rust_snippet_project() -> bool:
    """Create the persistent temp Cargo project for Rust snippet execution.

    Returns True if the project is ready, False on error.
    """
    _RUST_SNIPPET_DIR.mkdir(parents=True, exist_ok=True)
    cargo_toml = _RUST_SNIPPET_DIR / "Cargo.toml"
    if not cargo_toml.exists():
        # [workspace] prevents Cargo from merging this into the parent workspace
        lowess_path = str(REPO_ROOT / "crates" / "lowess").replace("\\", "/")
        fastlowess_path = str(REPO_ROOT / "crates" / "fastLowess").replace("\\", "/")
        cargo_toml.write_text(
            textwrap.dedent(f"""\
                [workspace]

                [package]
                name = "doc-snippet"
                version = "0.1.0"
                edition = "2021"

                [[bin]]
                name = "doc-snippet"
                path = "src/main.rs"

                [dependencies]
                lowess    = {{ path = "{lowess_path}" }}
                fastLowess = {{ path = "{fastlowess_path}" }}
            """),
            encoding="utf-8",
        )
    src_dir = _RUST_SNIPPET_DIR / "src"
    src_dir.mkdir(exist_ok=True)
    return True


def run_rust(snippet: Snippet, timeout: int) -> RunResult:
    cargo_bin = _find_exe("cargo")
    if cargo_bin is None:
        return RunResult(
            snippet=snippet,
            runner="rust",
            skipped=True,
            skip_reason="cargo not found in PATH",
        )

    _ensure_rust_snippet_project()

    main_path = _RUST_SNIPPET_DIR / "src" / "main.rs"
    main_path.write_text(snippet.code, encoding="utf-8")

    # Share the parent workspace's target dir to reuse compiled deps
    target_dir = str(REPO_ROOT / "target" / "doc-snippet-target")

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [
                cargo_bin,
                "run",
                "--manifest-path",
                str(_RUST_SNIPPET_DIR / "Cargo.toml"),
                "--target-dir",
                target_dir,
                "--quiet",
            ],
            capture_output=True,
            timeout=timeout,
            text=True,
            cwd=str(REPO_ROOT),
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="rust",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="rust",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )


def _find_cpp_compiler() -> Optional[str]:
    for name in ("g++", "clang++", "c++"):
        exe = _find_exe(name)
        if exe:
            return exe
    return None


def _find_cpp_library() -> Optional[Path]:
    """Return the path to the built fastlowess_cpp shared/static library, or None."""
    candidates = [
        # MinGW targets on Windows (Makefile uses these when GCC is available)
        REPO_ROOT / "target" / "x86_64-pc-windows-gnu" / "release-c",
        REPO_ROOT / "target" / "aarch64-pc-windows-gnu" / "release-c",
        # Default release-c (MSVC on Windows, native on Unix)
        REPO_ROOT / "target" / "release-c",
        REPO_ROOT / "target" / "debug",
    ]
    lib_names = [
        "fastlowess_cpp.dll",
        "fastlowess_cpp.lib",
        "libfastlowess_cpp.so",
        "libfastlowess_cpp.dylib",
        "libfastlowess_cpp.a",
    ]
    seen: set[Path] = set()
    for d in candidates:
        if d in seen:
            continue
        seen.add(d)
        if not d.exists():
            continue
        for name in lib_names:
            if (d / name).exists():
                return d
    return None


def run_cpp(snippet: Snippet, timeout: int) -> RunResult:
    compiler = _find_cpp_compiler()
    if compiler is None:
        return RunResult(
            snippet=snippet,
            runner="cpp",
            skipped=True,
            skip_reason="no C++ compiler (g++/clang++) found in PATH",
        )

    lib_dir = _find_cpp_library()
    if lib_dir is None:
        return RunResult(
            snippet=snippet,
            runner="cpp",
            skipped=True,
            skip_reason="fastlowess_cpp library not built (run 'make cpp' first)",
        )

    include_dir = str(REPO_ROOT / "bindings" / "cpp" / "include")
    lib_dir_str = str(lib_dir)

    code = snippet.code

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        src = os.path.join(tmpdir, "snippet.cpp")
        exe = os.path.join(tmpdir, "snippet.exe" if os.name == "nt" else "snippet")
        with open(src, "w", encoding="utf-8") as f:
            f.write(code)

        compile_cmd = [
            compiler,
            "-std=c++17",
            "-D_USE_MATH_DEFINES",
            "-O0",
            f"-I{include_dir}",
            f"-L{lib_dir_str}",
            src,
            "-o",
            exe,
            "-lfastlowess_cpp",
        ]

        try:
            t0 = time.monotonic()
            cproc = subprocess.run(
                compile_cmd,
                capture_output=True,
                timeout=60,
                text=True,
            )
            if cproc.returncode != 0:
                dur = time.monotonic() - t0
                # Detect MSVC-ABI / MinGW linker mismatch so we skip rather than fail.
                _abi_markers = ("__chkstk", "??_7type_info", "_Unwind_Resume")
                if os.name == "nt" and any(m in cproc.stderr for m in _abi_markers):
                    return RunResult(
                        snippet=snippet,
                        runner="cpp",
                        skipped=True,
                        skip_reason=(
                            "C++ library ABI mismatch (MSVC vs MinGW) — "
                            "rebuild with: make cpp (using the x86_64-pc-windows-gnu target)"
                        ),
                    )
                return RunResult(
                    snippet=snippet,
                    runner="cpp",
                    passed=False,
                    duration=dur,
                    stdout=cproc.stdout,
                    stderr=cproc.stderr,
                    returncode=cproc.returncode,
                )

            # Set library search path and run
            env = dict(os.environ)
            if os.name == "nt":
                env["PATH"] = lib_dir_str + os.pathsep + env.get("PATH", "")
            elif sys.platform == "darwin":
                env["DYLD_LIBRARY_PATH"] = (
                    lib_dir_str + os.pathsep + env.get("DYLD_LIBRARY_PATH", "")
                )
            else:
                env["LD_LIBRARY_PATH"] = (
                    lib_dir_str + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                )

            rproc = subprocess.run(
                [exe],
                capture_output=True,
                timeout=timeout,
                text=True,
                env=env,
            )
            dur = time.monotonic() - t0
            return RunResult(
                snippet=snippet,
                runner="cpp",
                passed=(rproc.returncode == 0),
                duration=dur,
                stdout=rproc.stdout,
                stderr=rproc.stderr,
                returncode=rproc.returncode,
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                snippet=snippet,
                runner="cpp",
                passed=False,
                duration=timeout,
                stderr=f"Timed out after {timeout}s",
            )


_RUNNERS = {
    "python": run_python,
    "julia": run_julia,
    "nodejs": run_nodejs,
    "r": run_r,
    "wasm": run_wasm,
    "rust": run_rust,
    "cpp": run_cpp,
}


def _find_exe(name: str) -> Optional[str]:
    import shutil

    return shutil.which(name)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def iter_md_files(root: Path, file_filter: Optional[str]) -> Iterator[Path]:
    if file_filter:
        p = Path(file_filter)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.is_file():
            yield p
            return
        # Treat as glob
        yield from sorted(REPO_ROOT.glob(file_filter))
        return
    yield from sorted(root.rglob("*.md"))


def main(argv: Optional[List[str]] = None) -> int:
    # Ensure stdout/stderr use UTF-8 on Windows (avoids UnicodeEncodeError for
    # characters like π that can't be encoded by the default cp1252 codec).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lang",
        choices=["python", "julia", "nodejs", "r", "wasm", "rust", "cpp", "all"],
        default="all",
        help="Which language runner to use (default: all)",
    )
    parser.add_argument(
        "--file",
        metavar="PATH_OR_GLOB",
        help="Restrict to a specific file or glob (relative to repo root)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List snippets without running them"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print snippet source and full output on failure",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop after the first failing snippet",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-snippet timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--output", metavar="FILE", help="Write JSON report to this file"
    )
    args = parser.parse_args(argv)

    active_runners: set[str] = (
        {"python", "julia", "nodejs", "r", "wasm", "rust", "cpp"}
        if args.lang == "all"
        else {args.lang}
    )

    # Detect best Python executable
    global _PYTHON_BIN
    _PYTHON_BIN = _find_python_with_fastlowess()

    # ---- Collect snippets ---------------------------------------------------
    snippets: List[Snippet] = []
    for md in iter_md_files(DOCS_DIR, args.file):
        snippets.extend(extract_snippets(md))

    total_found = len(snippets)

    # Filter to only snippets we can handle
    runnable: List[Tuple[Snippet, str]] = []  # (snippet, runner)
    for s in snippets:
        r = s.runner
        if r is None or r not in active_runners:
            continue
        reason = should_skip(s, r)
        if reason:
            continue
        runnable.append((s, r))

    print(bold("\nfastLowess doc snippet verifier"))
    print(f"Docs dir : {DOCS_DIR}")
    print(f"Runners  : {', '.join(sorted(active_runners))}")
    print(f"Snippets : {len(runnable)} runnable / {total_found} total")
    if args.dry_run:
        print()
        for s, r in runnable:
            print(f"  {cyan(r):20s}  {s.label}")
        print()
        return 0
    print()

    # ---- Run snippets (parallel per language) --------------------------------
    results: List[RunResult] = []
    n_pass = n_fail = n_skip = 0
    _print_lock = threading.Lock()

    # Group runnable snippets by runner so each language executes as a unit.
    # Rust snippets share a single temp project (main.rs), so they must remain
    # sequential within their language — parallelism is across languages only.
    _by_runner: dict[str, List[Tuple[Snippet, str]]] = defaultdict(list)
    for _s, _r in runnable:
        _by_runner[_r].append((_s, _r))

    def _run_language(lang_items: List[Tuple[Snippet, str]]) -> List[RunResult]:
        lang_results: List[RunResult] = []
        for s, runner in lang_items:
            label = s.label
            run_fn = _RUNNERS.get(runner)
            if run_fn is None:
                res = RunResult(
                    snippet=s,
                    runner=runner,
                    skipped=True,
                    skip_reason="no runner implementation",
                )
                with _print_lock:
                    print(
                        f"  {cyan(runner):20s}  {label} … {yellow('SKIP (no runner)')}"
                    )
            else:
                res = run_fn(s, args.timeout)
                with _print_lock:
                    sys.stdout.write(f"  {cyan(runner):20s}  {label} … ")
                    if res.skipped:
                        print(yellow(f"SKIP ({res.skip_reason})"))
                    elif res.passed:
                        print(green(f"PASS ({res.duration:.2f}s)"))
                    else:
                        print(red(f"FAIL ({res.duration:.2f}s, exit {res.returncode})"))
                        if args.verbose:
                            _print_failure(s, res)

            lang_results.append(res)
            if args.stop_on_fail and not res.skipped and not res.passed:
                with _print_lock:
                    print(
                        red(
                            f"\n[{runner}] Stopped after first failure (--stop-on-fail)."
                        )
                    )
                break
        return lang_results

    n_workers = len(_by_runner) or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_run_language, items) for items in _by_runner.values()
        ]
        for future in concurrent.futures.as_completed(futures):
            for res in future.result():
                results.append(res)
                if res.skipped:
                    n_skip += 1
                elif res.passed:
                    n_pass += 1
                else:
                    n_fail += 1

    # ---- Summary ------------------------------------------------------------
    print()
    print("-" * 60)
    print(bold("Summary"))
    print(
        f"  {green(f'PASS: {n_pass}'):30s}  {yellow(f'SKIP: {n_skip}'):30s}  {red(f'FAIL: {n_fail}')}"
    )
    print()

    # Print failures in verbose mode (already shown inline) or always in brief
    failures = [r for r in results if not r.passed and not r.skipped]
    if failures and not args.verbose:
        print(bold("Failed snippets:"))
        for r in failures:
            print(f"  {red('FAIL')} {r.snippet.label}")
            # Show first 5 lines of stderr
            if r.stderr.strip():
                for line in r.stderr.strip().splitlines()[:5]:
                    print(f"      {line}")
        print()

    # ---- JSON output --------------------------------------------------------
    if args.output:
        report = {
            "summary": {"pass": n_pass, "fail": n_fail, "skip": n_skip},
            "snippets": [
                {
                    "file": str(r.snippet.file.relative_to(REPO_ROOT)),
                    "line": r.snippet.line,
                    "lang": r.snippet.lang_tag,
                    "tab": r.snippet.tab,
                    "runner": r.runner,
                    "status": "skip" if r.skipped else ("pass" if r.passed else "fail"),
                    "skip_reason": r.skip_reason if r.skipped else None,
                    "returncode": r.returncode if not r.skipped else None,
                    "duration": round(r.duration, 3),
                    "stderr": r.stderr[:2000] if r.stderr else "",
                }
                for r in results
            ],
        }
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.output}")

    return 1 if n_fail > 0 else 0


def _print_failure(snippet: Snippet, res: RunResult) -> None:
    """Print detailed failure information."""
    sep = "-" * 56
    print()
    print(f"  {sep}")
    print(f"  {bold('File:')} {snippet.label}")
    if snippet.tab:
        print(f"  {bold('Tab:')}  {snippet.tab}")
    print(f"  {bold('Code:')}")
    for line in snippet.code.splitlines()[:20]:
        print(f"    {line}")
    if len(snippet.code.splitlines()) > 20:
        print(f"    ... ({len(snippet.code.splitlines())} lines total)")
    if res.stderr.strip():
        print(f"  {bold('Stderr:')}")
        for line in res.stderr.strip().splitlines()[-20:]:
            print(f"    {line}")
    if res.stdout.strip():
        print(f"  {bold('Stdout:')}")
        for line in res.stdout.strip().splitlines()[-10:]:
            print(f"    {line}")
    print(f"  {sep}")
    print()


if __name__ == "__main__":
    sys.exit(main())
