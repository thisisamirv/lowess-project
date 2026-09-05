"""Go snippet runner."""

from __future__ import annotations

import concurrent.futures
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict

from .base import REPO_ROOT, RunResult, Snippet, _find_exe

GO_BINDING_DIR = REPO_ROOT / "bindings" / "go"
GO_MODULE_DIR = GO_BINDING_DIR / "fastlowess"
MODULE_PATH = "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"

_GO_SNIPPET_DIR = REPO_ROOT / "target" / "doc-snippet-runner" / "go"
_GO_BIN_DIR = REPO_ROOT / "target" / "doc-snippet-target" / "go"
_GO_BIN_PREFIX = "snippet_"


def skip_reason(snippet: Snippet) -> str | None:
    if not re.search(r"\bfunc\s+main\s*\(", snippet.code):
        return "fragment — no func main (not a standalone Go program)"
    if not (GO_MODULE_DIR / "go.mod").exists():
        return "bindings/go/fastlowess/go.mod not found"
    return None


def _ensure_go_snippet_module(names: list[str], snippets: list[Snippet]) -> None:
    """(Re)write the persistent module used for batched Go snippet execution.

    Each snippet becomes its own `cmd/snippet_NNNN/main.go` package so a single
    `go build ./...` compiles the whole batch at once: module resolution and
    cgo compilation/linking of the shared fastlowess binding are paid once
    (with the Go toolchain building independent packages concurrently) instead
    of once per snippet via repeated `go run` invocations, each in its own
    throwaway module.
    """
    cmd_dir = _GO_SNIPPET_DIR / "cmd"
    if cmd_dir.exists():
        shutil.rmtree(cmd_dir)
    cmd_dir.mkdir(parents=True, exist_ok=True)
    (_GO_SNIPPET_DIR / "go.mod").write_text(
        "module snippet\n\n"
        "go 1.23\n\n"
        f"require {MODULE_PATH} v0.0.0\n\n"
        f"replace {MODULE_PATH} => {GO_MODULE_DIR.as_posix()}\n",
        encoding="utf-8",
    )
    for name, snippet in zip(names, snippets):
        pkg_dir = cmd_dir / name
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "main.go").write_text(snippet.code, encoding="utf-8")


def run_go(snippet: Snippet, timeout: int) -> RunResult:
    return run_go_batch([snippet], timeout)[0]


def run_go_batch(snippets: list[Snippet], timeout: int) -> list[RunResult]:
    """Compile every Go snippet in one `go build ./...`, then run each binary.

    Much faster than one `go run` per snippet: each snippet used to pay its
    own `go run` module-resolution + cgo compile/link overhead serially in a
    fresh temp dir; here that cost is paid once for the whole batch, and the
    already-built binaries are then executed concurrently too.
    """
    if not snippets:
        return []

    go_exe = _find_exe("go")
    if go_exe is None:
        return [
            RunResult(
                snippet=s,
                runner="go",
                skipped=True,
                skip_reason="no 'go' executable found in PATH",
            )
            for s in snippets
        ]

    names = [f"{_GO_BIN_PREFIX}{i:04d}" for i in range(len(snippets))]
    _ensure_go_snippet_module(names, snippets)

    _GO_BIN_DIR.mkdir(parents=True, exist_ok=True)
    exe_suffix = ".exe" if os.name == "nt" else ""

    env = dict(os.environ)
    env["CGO_ENABLED"] = "1"

    compile_timeout = timeout * max(6, len(snippets))
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [go_exe, "build", "-o", str(_GO_BIN_DIR) + os.sep, "./..."],
            cwd=str(_GO_SNIPPET_DIR),
            capture_output=True,
            check=False,
            timeout=compile_timeout,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except subprocess.TimeoutExpired:
        return [
            RunResult(
                snippet=s,
                runner="go",
                passed=False,
                duration=compile_timeout,
                stderr=f"Batch compile timed out after {compile_timeout}s",
            )
            for s in snippets
        ]
    compile_dur = time.monotonic() - t0

    # Go groups multi-package build errors under a "# <import path>" header;
    # attribute the lines following each header back to that snippet.
    diagnostics: dict[str, list[str]] = defaultdict(list)
    current: str | None = None
    header_re = re.compile(
        r"^# .*[/\\]cmd[/\\](" + re.escape(_GO_BIN_PREFIX) + r"\d+)\b"
    )
    for line in proc.stderr.splitlines():
        m = header_re.match(line)
        if m:
            current = m.group(1)
            continue
        if current is not None:
            diagnostics[current].append(line)

    def _run_one(name: str, snippet: Snippet) -> RunResult:
        exe_path = _GO_BIN_DIR / f"{name}{exe_suffix}"
        if not exe_path.exists():
            return RunResult(
                snippet=snippet,
                runner="go",
                passed=False,
                duration=compile_dur / len(snippets),
                stderr="\n".join(diagnostics.get(name, [])) or proc.stderr,
                returncode=proc.returncode,
            )
        t1 = time.monotonic()
        try:
            run_proc = subprocess.run(
                [str(exe_path)],
                capture_output=True,
                check=False,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                cwd=str(REPO_ROOT),
            )
            return RunResult(
                snippet=snippet,
                runner="go",
                passed=(run_proc.returncode == 0),
                duration=time.monotonic() - t1,
                stdout=run_proc.stdout,
                stderr=run_proc.stderr,
                returncode=run_proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                snippet=snippet,
                runner="go",
                passed=False,
                duration=timeout,
                stderr=f"Timed out after {timeout}s",
            )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, len(snippets))
    ) as executor:
        return list(executor.map(_run_one, names, snippets))
