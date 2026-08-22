"""C++ snippet runner."""

from __future__ import annotations

import glob as _glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .base import REPO_ROOT, RunResult, Snippet, _find_exe


def skip_reason(snippet: Snippet) -> str | None:
    if not re.search(r"\bint\s+main\s*\(", snippet.code):
        return "fragment — no int main (not a standalone C++ program)"
    return None


_msvc_env_cache: dict[str, str] | None = None


def _find_cpp_compiler() -> str | None:
    for name in ("g++", "clang++", "c++"):
        exe = _find_exe(name)
        if exe:
            return exe
    return None


def _find_msvc_compiler() -> str | None:
    if (cl := _find_exe("cl")) is not None:
        return cl
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if os.path.exists(vswhere):
        try:
            result = subprocess.run(
                [vswhere, "-all", "-find", r"VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                path = line.strip()
                if path and os.path.exists(path):
                    return path
        except (OSError, subprocess.TimeoutExpired):
            pass
    for pattern in [
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
    ]:
        matches = sorted(_glob.glob(pattern))
        if matches:
            return matches[-1]
    return None


def _is_msvc_library(lib_dir: Path) -> bool:
    return "windows-msvc" in str(lib_dir)


def _find_vcvarsall(compiler_path: str) -> str | None:
    """Walk up from cl.exe to find vcvarsall.bat (lives at VC/Auxiliary/Build/)."""
    path = Path(compiler_path).parent
    for _ in range(10):
        candidate = path / "Auxiliary" / "Build" / "vcvarsall.bat"
        if candidate.exists():
            return str(candidate)
        path = path.parent
    return None


def _get_msvc_env(vcvarsall: str) -> dict[str, str]:
    """Return the environment after sourcing vcvarsall.bat x64."""
    global _msvc_env_cache
    if _msvc_env_cache is not None:
        return _msvc_env_cache
    try:
        # `call` is required: without it, vcvarsall.bat's GOTO :EOF causes cmd.exe
        # to exit entirely, so `&& set` never runs and the env is never captured.
        result = subprocess.run(
            f'call "{vcvarsall}" x64 > nul 2>&1 && set',
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        env: dict[str, str] = {}
        for line in result.stdout.splitlines():
            key, sep, value = line.partition("=")
            if sep:
                env[key] = value
        _msvc_env_cache = env if env else dict(os.environ)
    except (OSError, subprocess.TimeoutExpired):
        _msvc_env_cache = dict(os.environ)
    return _msvc_env_cache


def _find_cpp_library() -> Path | None:
    candidates = [
        REPO_ROOT / "target" / "x86_64-pc-windows-msvc" / "release-c",
        REPO_ROOT / "target" / "x86_64-pc-windows-gnu" / "release-c",
        REPO_ROOT / "target" / "aarch64-pc-windows-gnu" / "release-c",
        REPO_ROOT / "target" / "release-c",
        REPO_ROOT / "target" / "debug",
    ]
    lib_names = [
        "fastlowess_cpp.dll",
        "fastlowess_cpp.lib",
        "libfastlowess_cpp.so",
        "libfastlowess_cpp.dylib",
        "libfastlowess_cpp.a",
    ]
    seen: set[Path] = set()
    for d in candidates:
        if d in seen:
            continue
        seen.add(d)
        if not d.exists():
            continue
        for name in lib_names:
            if (d / name).exists():
                return d
    return None


def run_cpp(snippet: Snippet, timeout: int) -> RunResult:
    lib_dir = _find_cpp_library()
    if lib_dir is None:
        return RunResult(
            snippet=snippet,
            runner="cpp",
            skipped=True,
            skip_reason="fastlowess_cpp library not built (run 'make cpp' first)",
        )

    include_dir = str(REPO_ROOT / "bindings" / "cpp" / "include")
    lib_dir_str = str(lib_dir)
    use_msvc = os.name == "nt" and _is_msvc_library(lib_dir)

    if use_msvc:
        compiler = _find_msvc_compiler()
        if compiler is None:
            return RunResult(
                snippet=snippet,
                runner="cpp",
                skipped=True,
                skip_reason="no MSVC cl.exe found in PATH (required for MSVC-built library)",
            )
        vcvarsall = _find_vcvarsall(compiler)
        msvc_env = _get_msvc_env(vcvarsall) if vcvarsall else dict(os.environ)
        _env_path = msvc_env.get("Path") or msvc_env.get("PATH", "")
        _cl = shutil.which("cl", path=_env_path) if _env_path else None
        if _cl:
            compiler = _cl
    else:
        msvc_env = None
        compiler = _find_cpp_compiler()
        if compiler is None:
            return RunResult(
                snippet=snippet,
                runner="cpp",
                skipped=True,
                skip_reason="no C++ compiler (g++/clang++) found in PATH",
            )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        src = os.path.join(tmpdir, "snippet.cpp")
        exe = os.path.join(tmpdir, "snippet.exe" if os.name == "nt" else "snippet")
        with open(src, "w", encoding="utf-8") as f:
            f.write(snippet.code)

        if use_msvc:
            import_lib = "fastlowess_cpp.dll.lib"
            if not (lib_dir / import_lib).exists():
                import_lib = "fastlowess_cpp.lib"
            compile_cmd = [
                compiler,
                "/nologo",
                "/EHsc",
                "/std:c++20",
                "/D_USE_MATH_DEFINES",
                "/Od",
                f"/I{include_dir}",
                f"/Fe:{exe}",
                src,
                "/link",
                f"/LIBPATH:{lib_dir_str}",
                import_lib,
            ]
        else:
            compile_cmd = [
                compiler,
                "-std=c++17",
                "-D_USE_MATH_DEFINES",
                "-O0",
                f"-I{include_dir}",
                f"-L{lib_dir_str}",
                src,
                "-o",
                exe,
                "-lfastlowess_cpp",
            ]

        try:
            t0 = time.monotonic()
            cproc = subprocess.run(
                compile_cmd,
                capture_output=True,
                check=False,
                timeout=60,
                text=True,
                env=msvc_env if use_msvc else None,
            )
            if cproc.returncode != 0:
                dur = time.monotonic() - t0
                return RunResult(
                    snippet=snippet,
                    runner="cpp",
                    passed=False,
                    duration=dur,
                    stdout=cproc.stdout,
                    stderr=cproc.stderr,
                    returncode=cproc.returncode,
                )

            env = dict(os.environ)
            if os.name == "nt":
                env["PATH"] = lib_dir_str + os.pathsep + env.get("PATH", "")
            elif sys.platform == "darwin":
                env["DYLD_LIBRARY_PATH"] = (
                    lib_dir_str + os.pathsep + env.get("DYLD_LIBRARY_PATH", "")
                )
            else:
                env["LD_LIBRARY_PATH"] = (
                    lib_dir_str + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                )

            rproc = subprocess.run(
                [exe],
                capture_output=True,
                check=False,
                timeout=timeout,
                text=True,
                env=env,
            )
            dur = time.monotonic() - t0
            return RunResult(
                snippet=snippet,
                runner="cpp",
                passed=(rproc.returncode == 0),
                duration=dur,
                stdout=rproc.stdout,
                stderr=rproc.stderr,
                returncode=rproc.returncode,
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                snippet=snippet,
                runner="cpp",
                passed=False,
                duration=timeout,
                stderr=f"Timed out after {timeout}s",
            )
