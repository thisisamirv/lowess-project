"""Java snippet runner."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from .base import REPO_ROOT, RunResult, Snippet, _find_exe

JAVA_BINDING_DIR = REPO_ROOT / "bindings" / "java"
JAVA_CLASSES_DIR = JAVA_BINDING_DIR / "target" / "classes"
JAVA_NATIVE_DIR = REPO_ROOT / "target" / "debug"

# Comparison-only dependency for the alternative-software guide page (not a
# hard requirement of the binding itself), lazily downloaded and cached so
# `make java-dev` works offline once fetched once.
_COMMONS_MATH_VERSION = "3.6.1"
_COMMONS_MATH_JAR = (
    REPO_ROOT / "target" / "tmp" / f"commons-math3-{_COMMONS_MATH_VERSION}.jar"
)
_COMMONS_MATH_URL = (
    "https://repo1.maven.org/maven2/org/apache/commons/commons-math3/"
    f"{_COMMONS_MATH_VERSION}/commons-math3-{_COMMONS_MATH_VERSION}.jar"
)


def _commons_math_jar() -> Path | None:
    """Path to the cached commons-math3 jar, downloading it on first use.

    Returns `None` (rather than raising) if it can't be obtained, so the
    caller can skip the snippet instead of failing the whole check.
    """
    if _COMMONS_MATH_JAR.is_file():
        return _COMMONS_MATH_JAR
    try:
        _COMMONS_MATH_JAR.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_COMMONS_MATH_URL, _COMMONS_MATH_JAR)
        return _COMMONS_MATH_JAR
    except OSError:
        return None


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
    if "org.apache.commons.math3" in snippet.code and _commons_math_jar() is None:
        return "commons-math3 jar not available (optional comparison-only dependency)"
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
    extra_classpath = (
        [str(_COMMONS_MATH_JAR)]
        if "org.apache.commons.math3" in snippet.code and _COMMONS_MATH_JAR.is_file()
        else []
    )

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
                    os.pathsep.join([str(JAVA_CLASSES_DIR), *extra_classpath]),
                    "-d",
                    str(out_dir),
                    str(src_dir / f"{class_name}.java"),
                ],
                capture_output=True,
                check=False,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
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
                    os.pathsep.join(
                        [str(out_dir), str(JAVA_CLASSES_DIR), *extra_classpath]
                    ),
                    class_name,
                ],
                capture_output=True,
                check=False,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
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
