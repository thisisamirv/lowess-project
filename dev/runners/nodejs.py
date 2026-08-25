"""Node.js snippet runner."""

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
    if ": SmoothOptions" in code or ": LowessResult" in code:
        return "TypeScript (not Node.js)"
    if not re.search(r"require\s*\(", code):
        return "no require() — snippet must load fastlowess itself"
    if re.search(r"\binstallGpu\s*\(|backend:\s*[\"']gpu[\"']", code):
        return "requires gpu feature (not enabled in CI build)"
    return None


_NODEJS_DIR = REPO_ROOT / "bindings" / "nodejs"

# Rayon's global thread pool (spun up by `parallel: true`, the default for batch
# `Lowess`) can race with libuv's handle teardown when the Node process exits on
# Windows, aborting with "Assertion failed: !(handle->flags & UV_HANDLE_CLOSING)"
# in src\\win\\async.c. This is a transient runtime race, not a snippet bug, so
# retry a few times before reporting failure.
_LIBUV_CRASH_RE = re.compile(r"UV_HANDLE_CLOSING|src\\win\\async\.c", re.IGNORECASE)
_MAX_ATTEMPTS = 3


def _ensure_nodejs_selflink(nodejs_dir: Path) -> None:
    """Create node_modules/fastlowess shim so require('fastlowess') resolves locally."""
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


def run_nodejs(snippet: Snippet, timeout: int) -> RunResult:
    node_bin = _find_exe("node")
    if node_bin is None:
        return RunResult(
            snippet=snippet,
            runner="nodejs",
            skipped=True,
            skip_reason="node not found in PATH",
        )

    cwd = str(_NODEJS_DIR) if _NODEJS_DIR.exists() else str(REPO_ROOT)
    if _NODEJS_DIR.exists():
        _ensure_nodejs_selflink(_NODEJS_DIR)

    tmp_name = f"_snippet_{uuid.uuid4().hex}.js"
    tmp = str(Path(cwd) / tmp_name)
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
            cwd=cwd,
        )
        for _ in range(_MAX_ATTEMPTS - 1):
            if proc.returncode == 0 or not _LIBUV_CRASH_RE.search(proc.stderr or ""):
                break
            proc = subprocess.run(
                [node_bin, tmp],
                capture_output=True,
                check=False,
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
