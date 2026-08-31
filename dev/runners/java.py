"""Java snippet runner."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .base import REPO_ROOT, RunResult, Snippet, _find_exe

JAVA_BINDING_DIR = REPO_ROOT / "bindings" / "java"
JAVA_CLASSES_DIR = JAVA_BINDING_DIR / "target" / "classes"
JAVA_NATIVE_DIR = REPO_ROOT / "target" / "debug"

_MAIN_RE = re.compile(r"public\s+static\s+void\s+main\s*\(")
_CLASS_RE = re.compile(r"\bclass\s+(\w+)")


def _find_java_exe(name: str) -> str | None:
    """Prefer JAVA_HOME's bin/ over a bare PATH search.

    A PATH search alone is unreliable here: other installed JDKs (e.g. an
    Eclipse Adoptium installer adding itself to the *Machine*-scope PATH,
    which always precedes the User-scope PATH regardless of ordering) can
    shadow the JDK actually used to compile bindings/java/target/classes,
    causing a "class file has wrong version" mismatch.
    """
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidate = (
            Path(java_home) / "bin" / (f"{name}.exe" if os.name == "nt" else name)
        )
        if candidate.is_file():
            return str(candidate)
    return _find_exe(name)


def skip_reason(snippet: Snippet) -> str | None:
    if not _MAIN_RE.search(snippet.code):
        return "fragment — no public static void main (not a standalone Java program)"
    if not _CLASS_RE.search(snippet.code):
        return "no top-level class declaration found"
    if not JAVA_CLASSES_DIR.exists():
        return "bindings/java/target/classes not found — run `mvn compile` first"
    return None


def run_java(snippet: Snippet, timeout: int) -> RunResult:
    javac_exe = _find_java_exe("javac")
    java_exe = _find_java_exe("java")
    if javac_exe is None or java_exe is None:
        return RunResult(
            snippet=snippet,
            runner="java",
            skipped=True,
            skip_reason="no 'javac'/'java' executable found in PATH",
        )

    class_name = _CLASS_RE.search(snippet.code).group(1)  # type: ignore[union-attr]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        src_dir = Path(tmpdir) / "src"
        out_dir = Path(tmpdir) / "out"
        src_dir.mkdir()
        out_dir.mkdir()
        (src_dir / f"{class_name}.java").write_text(snippet.code, encoding="utf-8")

        t0 = time.monotonic()
        try:
            compile_proc = subprocess.run(
                [
                    javac_exe,
                    "-cp",
                    str(JAVA_CLASSES_DIR),
                    "-d",
                    str(out_dir),
                    str(src_dir / f"{class_name}.java"),
                ],
                capture_output=True,
                check=False,
                timeout=timeout,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                snippet=snippet,
                runner="java",
                passed=False,
                duration=time.monotonic() - t0,
                stderr=f"Compilation timed out after {timeout}s",
            )

        if compile_proc.returncode != 0:
            return RunResult(
                snippet=snippet,
                runner="java",
                passed=False,
                duration=time.monotonic() - t0,
                stdout=compile_proc.stdout,
                stderr=compile_proc.stderr,
            )

        try:
            run_proc = subprocess.run(
                [
                    java_exe,
                    "--enable-native-access=ALL-UNNAMED",
                    f"-Dfastlowess.native.dir={JAVA_NATIVE_DIR}",
                    "-cp",
                    os.pathsep.join([str(out_dir), str(JAVA_CLASSES_DIR)]),
                    class_name,
                ],
                capture_output=True,
                check=False,
                timeout=timeout,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                snippet=snippet,
                runner="java",
                passed=False,
                duration=time.monotonic() - t0,
                stderr=f"Timed out after {timeout}s",
            )

        return RunResult(
            snippet=snippet,
            runner="java",
            passed=run_proc.returncode == 0,
            duration=time.monotonic() - t0,
            stdout=run_proc.stdout,
            stderr=run_proc.stderr,
        )
