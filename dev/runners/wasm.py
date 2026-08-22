"""WebAssembly snippet runner (Node.js + pre-built wasm pkg)."""

from __future__ import annotations

import os
import re
import subprocess
import time
import uuid
from pathlib import Path

from .base import REPO_ROOT, RunResult, Snippet, _find_exe


def skip_reason(snippet: Snippet) -> str | None:
    code = snippet.code
    if re.search(r"^import\b", code, re.MULTILINE) or "await init(" in code:
        return "ES module import (not supported in CJS runner)"
    if not re.search(r"require\s*\(", code):
        return "no require() — snippet must load the WASM package itself"
    return None


_WASM_PKG_DIR = REPO_ROOT / "bindings" / "wasm" / "pkg"


def _ensure_wasm_selflink(wasm_pkg_dir: Path) -> None:
    """Create node_modules/fastlowess-wasm shim so require('fastlowess-wasm') resolves locally."""
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


def run_wasm(snippet: Snippet, timeout: int) -> RunResult:
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

    _ensure_wasm_selflink(_WASM_PKG_DIR)

    tmp_name = f"_snippet_{uuid.uuid4().hex}.js"
    tmp = str(_WASM_PKG_DIR / tmp_name)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(snippet.code)

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [node_bin, tmp],
            capture_output=True,
            check=False,
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
