"""WebAssembly snippet runner (Node.js + pre-built wasm pkg)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time

from .base import REPO_ROOT, RunResult, Snippet, _find_exe


def skip_reason(snippet: Snippet) -> str | None:
    code = snippet.code
    if re.search(r"^import\b", code, re.MULTILINE) or "await init(" in code:
        return "ES module import (not supported in CJS runner)"
    if not re.search(r"require\s*\(", code):
        return "no require() — snippet must load the WASM package itself"
    return None


_WASM_PKG_DIR = REPO_ROOT / "bindings" / "wasm" / "pkg"


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

    pkg_meta_path = _WASM_PKG_DIR / "package.json"
    try:
        pkg_meta = json.loads(pkg_meta_path.read_text(encoding="utf-8"))
        index_path = str(_WASM_PKG_DIR / pkg_meta["main"]).replace("\\", "/")
        pkg_name = pkg_meta.get("name", "")
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        return RunResult(
            snippet=snippet,
            runner="wasm",
            skipped=True,
            skip_reason=f"pkg/package.json missing or invalid: {exc}",
        )

    # Patch require('<pkg-name>') → require('/abs/path') so the temp file
    # can live outside the workspace (avoids VS Code TS language-server churn).
    patched = re.sub(
        r"require\(['\"]" + re.escape(pkg_name) + r"['\"]\)",
        f"require('{index_path}')",
        snippet.code,
    )

    tmp_fd, tmp = tempfile.mkstemp(suffix=".js", prefix="_wasm_snippet_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(patched)
        t0 = time.monotonic()
        proc = subprocess.run(
            [node_bin, tmp],
            capture_output=True,
            check=False,
            timeout=timeout,
            text=True,
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
        try:
            os.unlink(tmp)
        except OSError:
            pass
