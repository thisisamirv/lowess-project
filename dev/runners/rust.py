"""Rust snippet runner."""

from __future__ import annotations

import concurrent.futures
import json
import re
import shutil
import subprocess
import textwrap
import time
from collections import defaultdict

from .base import REPO_ROOT, RunResult, Snippet, _find_exe


def _strip_hidden_lines(code: str) -> str:
    """Remove Rustdoc `# ` hidden-line prefixes so the code is valid Rust."""
    out = []
    for line in code.splitlines():
        if line.startswith("# "):
            out.append(line[2:])
        elif line == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def skip_reason(snippet: Snippet) -> str | None:
    code = _strip_hidden_lines(snippet.code)
    if not re.search(r"\bfn\s+main\s*\(", code):
        return "fragment — no fn main (not a standalone Rust program)"
    if code.strip().startswith("[") and "=" in code and "fn " not in code:
        return "TOML/config snippet"
    if re.search(r"\bBackend\s*::\s*GPU\b", code):
        return "requires gpu feature flag (not enabled in snippet workspace)"
    if re.search(r"\bcross_validate\b|\bKFold\b", code):
        return "cross_validate/KFold not in stable public API"
    return None


_RUST_SNIPPET_DIR = REPO_ROOT / "target" / "doc-snippet-runner"
_RUST_BIN_PREFIX = "snippet_"


def _ensure_rust_snippet_project() -> None:
    """(Re)write the persistent Cargo project used for batched Rust snippet execution.

    Each snippet becomes its own `src/bin/snippet_NNNN.rs` binary target so that a
    single `cargo build` compiles (and thus caches shared dependencies for) every
    snippet at once, instead of paying Cargo's own startup/dependency-resolution
    overhead once per snippet via repeated `cargo run` invocations.
    """
    _RUST_SNIPPET_DIR.mkdir(parents=True, exist_ok=True)
    lowess_path = str(REPO_ROOT / "crates" / "lowess").replace("\\", "/")
    fastlowess_path = str(REPO_ROOT / "crates" / "fastLowess").replace("\\", "/")
    (_RUST_SNIPPET_DIR / "Cargo.toml").write_text(
        textwrap.dedent(f"""\
            [workspace]

            [package]
            name = "doc-snippet"
            version = "0.1.0"
            edition = "2021"

            [dependencies]
            lowess    = {{ path = "{lowess_path}" }}
            fastLowess = {{ path = "{fastlowess_path}" }}
        """),
        encoding="utf-8",
    )
    main_rs = _RUST_SNIPPET_DIR / "src" / "main.rs"
    if main_rs.exists():
        main_rs.unlink()
    bin_dir = _RUST_SNIPPET_DIR / "src" / "bin"
    if bin_dir.exists():
        shutil.rmtree(bin_dir)
    bin_dir.mkdir(parents=True)


def run_rust(snippet: Snippet, timeout: int) -> RunResult:
    return run_rust_batch([snippet], timeout)[0]


def run_rust_batch(snippets: list[Snippet], timeout: int) -> list[RunResult]:
    """Compile every Rust snippet in one `cargo build`, then run each binary.

    This is much faster than one `cargo run` per snippet: shared dependencies and
    Cargo's own overhead are paid once for the whole batch, `--keep-going` keeps
    snippet failures independent (a broken snippet doesn't block the others from
    building), and the already-compiled binaries are executed concurrently.
    """
    if not snippets:
        return []

    cargo_bin = _find_exe("cargo")
    if cargo_bin is None:
        return [
            RunResult(
                snippet=s,
                runner="rust",
                skipped=True,
                skip_reason="cargo not found in PATH",
            )
            for s in snippets
        ]

    _ensure_rust_snippet_project()
    bin_dir = _RUST_SNIPPET_DIR / "src" / "bin"
    names = [f"{_RUST_BIN_PREFIX}{i:04d}" for i in range(len(snippets))]
    for name, snippet in zip(names, snippets):
        (bin_dir / f"{name}.rs").write_text(
            _strip_hidden_lines(snippet.code), encoding="utf-8"
        )

    target_dir = str(REPO_ROOT / "target" / "doc-snippet-target")
    compile_timeout = timeout * max(6, len(snippets))
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [
                cargo_bin,
                "build",
                "--manifest-path",
                str(_RUST_SNIPPET_DIR / "Cargo.toml"),
                "--target-dir",
                target_dir,
                "--keep-going",
                "--message-format=json",
            ],
            capture_output=True,
            check=False,
            timeout=compile_timeout,
            encoding="utf-8",
            errors="replace",
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        return [
            RunResult(
                snippet=s,
                runner="rust",
                passed=False,
                duration=compile_timeout,
                stderr=f"Batch compile timed out after {compile_timeout}s",
            )
            for s in snippets
        ]
    compile_dur = time.monotonic() - t0

    executables: dict[str, str] = {}
    diagnostics: dict[str, list[str]] = defaultdict(list)
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        target_name = (msg.get("target") or {}).get("name", "")
        if not target_name.startswith(_RUST_BIN_PREFIX):
            continue
        if msg.get("reason") == "compiler-artifact" and msg.get("executable"):
            executables[target_name] = msg["executable"]
        elif msg.get("reason") == "compiler-message":
            rendered = (msg.get("message") or {}).get("rendered")
            if rendered:
                diagnostics[target_name].append(rendered)

    def _run_one(name: str, snippet: Snippet) -> RunResult:
        executable = executables.get(name)
        if executable is None:
            return RunResult(
                snippet=snippet,
                runner="rust",
                passed=False,
                duration=compile_dur / len(snippets),
                stderr="\n".join(diagnostics.get(name, [])) or proc.stderr,
                returncode=proc.returncode,
            )
        t1 = time.monotonic()
        try:
            run_proc = subprocess.run(
                [executable],
                capture_output=True,
                check=False,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                cwd=str(REPO_ROOT),
            )
            return RunResult(
                snippet=snippet,
                runner="rust",
                passed=(run_proc.returncode == 0),
                duration=time.monotonic() - t1,
                stdout=run_proc.stdout,
                stderr=run_proc.stderr,
                returncode=run_proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                snippet=snippet,
                runner="rust",
                passed=False,
                duration=timeout,
                stderr=f"Timed out after {timeout}s",
            )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, len(snippets))
    ) as executor:
        return list(executor.map(_run_one, names, snippets))
