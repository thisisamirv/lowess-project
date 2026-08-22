"""Rust snippet runner."""

from __future__ import annotations

import re
import subprocess
import textwrap
import time

from .base import REPO_ROOT, RunResult, Snippet, _find_exe


def skip_reason(snippet: Snippet) -> str | None:
    code = snippet.code
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


def _ensure_rust_snippet_project() -> None:
    """Create the persistent temp Cargo project for Rust snippet execution."""
    _RUST_SNIPPET_DIR.mkdir(parents=True, exist_ok=True)
    cargo_toml = _RUST_SNIPPET_DIR / "Cargo.toml"
    if not cargo_toml.exists():
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
    (_RUST_SNIPPET_DIR / "src").mkdir(exist_ok=True)


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
    (_RUST_SNIPPET_DIR / "src" / "main.rs").write_text(snippet.code, encoding="utf-8")

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
            check=False,
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
