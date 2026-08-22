"""Python snippet runner."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time

from .base import RunResult, Snippet

# Set to a venv python that has fastlowess installed (overridden by main()).
PYTHON_BIN: str = sys.executable


def skip_reason(snippet: Snippet) -> str | None:
    code = snippet.code
    if any(s in code for s in ["read_csv", "open(", "glob(", "argparse"]):
        return "file I/O"
    if not any(c in code for c in ["=", "(", "import", "print"]):
        return "no executable statements"
    if re.search(r"total_points\s*=\s*[1-9][0-9]{4,}", code):
        return "large synthetic dataset (too slow for CI)"
    if re.search(
        r"\bfl\b|\bfastlowess\b|\bLowess\b|\bStreamingLowess\b|\bOnlineLowess\b",
        code,
    ) and not re.search(r"\bimport\b.*fastlowess|\bfrom\b.*fastlowess", code):
        return "fastlowess not imported (snippet is not self-contained)"
    if re.search(r"\binstall_gpu\s*\(|backend\s*=\s*[\"']gpu[\"']", code):
        return "requires gpu feature (not enabled in CI build)"
    return None


def run_python(snippet: Snippet, timeout: int) -> RunResult:
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(snippet.code)
        tmp = f.name
    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [PYTHON_BIN, tmp],
            capture_output=True,
            check=False,
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
