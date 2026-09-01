"""Julia snippet runner."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time

from .base import REPO_ROOT, RunResult, Snippet, _find_exe


def skip_reason(snippet: Snippet) -> str | None:
    code = snippet.code
    if re.search(r"\bPkg\.(add|develop|clone|rm|pin)\s*\(", code):
        return "Pkg management snippet"
    if re.search(r";\s*\w+::", code, re.DOTALL):
        return (
            "Julia method signature (keyword arg with type annotation — not callable)"
        )
    if re.search(r"\binstall_gpu\s*\(|backend\s*=\s*[\"']gpu[\"']", code):
        return "requires gpu feature (not enabled in CI build)"
    return None


_JL_LIB_NAME = (
    "fastlowess_jl.dll"
    if sys.platform == "win32"
    else (
        "libfastlowess_jl.dylib" if sys.platform == "darwin" else "libfastlowess_jl.so"
    )
)


def run_julia(snippet: Snippet, timeout: int) -> RunResult:
    julia_bin = _find_exe("julia")
    if julia_bin is None:
        return RunResult(
            snippet=snippet,
            runner="julia",
            skipped=True,
            skip_reason="julia not found in PATH",
        )

    with tempfile.NamedTemporaryFile(
        suffix=".jl", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(snippet.code)
        tmp = f.name

    julia_project = REPO_ROOT / "bindings" / "julia" / "julia"
    env = {**os.environ}
    if julia_project.exists():
        env["JULIA_PROJECT"] = str(julia_project)

    if "FASTLOWESS_LIB" not in env:
        local_lib = REPO_ROOT / "target" / "release" / _JL_LIB_NAME
        if local_lib.exists():
            env["FASTLOWESS_LIB"] = str(local_lib)

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [julia_bin, "--startup-file=no", tmp],
            capture_output=True,
            check=False,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
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
