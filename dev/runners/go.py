"""Go snippet runner."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .base import REPO_ROOT, RunResult, Snippet, _find_exe

GO_BINDING_DIR = REPO_ROOT / "bindings" / "go"
GO_MODULE_DIR = GO_BINDING_DIR / "fastlowess"
MODULE_PATH = "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"


def skip_reason(snippet: Snippet) -> str | None:
    if not re.search(r"\bfunc\s+main\s*\(", snippet.code):
        return "fragment — no func main (not a standalone Go program)"
    if not (GO_MODULE_DIR / "go.mod").exists():
        return "bindings/go/fastlowess/go.mod not found"
    return None


def run_go(snippet: Snippet, timeout: int) -> RunResult:
    go_exe = _find_exe("go")
    if go_exe is None:
        return RunResult(
            snippet=snippet,
            runner="go",
            skipped=True,
            skip_reason="no 'go' executable found in PATH",
        )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        mod_dir = Path(tmpdir)
        # A throwaway module that requires the real binding via a local
        # `replace` directive, so the snippet exercises the actual package
        # without needing it published or network access.
        (mod_dir / "go.mod").write_text(
            "module snippet\n\n"
            "go 1.23\n\n"
            f"require {MODULE_PATH} v0.0.0\n\n"
            f"replace {MODULE_PATH} => {GO_MODULE_DIR.as_posix()}\n",
            encoding="utf-8",
        )
        (mod_dir / "main.go").write_text(snippet.code, encoding="utf-8")

        env = dict(os.environ)
        env["CGO_ENABLED"] = "1"

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [go_exe, "run", "."],
                cwd=str(mod_dir),
                capture_output=True,
                check=False,
                timeout=timeout,
                text=True,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                snippet=snippet,
                runner="go",
                passed=False,
                duration=time.monotonic() - t0,
                stderr=f"Timed out after {timeout}s",
            )

        return RunResult(
            snippet=snippet,
            runner="go",
            passed=proc.returncode == 0,
            duration=time.monotonic() - t0,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
