"""C++ snippet runner."""

from __future__ import annotations

import concurrent.futures
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


# Cached vcvarsall environments, keyed by (vcvarsall path, target arch).
_msvc_env_cache: dict[tuple[str, str], dict[str, str]] = {}

# MSVC host\target toolchain directories, most-preferred first, per target arch.
# Windows on ARM ships a native Hostarm64\arm64 toolchain; an x64 host can also
# cross-compile to arm64 via Hostx64\arm64. Building against the wrong arch
# fails to link the library, so the arch must follow the built library.
_MSVC_HOST_DIRS: dict[str, list[str]] = {
    "arm64": [r"Hostarm64\arm64", r"Hostx64\arm64", r"Hostx64\x64"],
    "x64": [r"Hostx64\x64", r"Hostarm64\x64"],
}


def _find_cpp_compiler() -> str | None:
    for name in ("g++", "clang++", "c++"):
        exe = _find_exe(name)
        if exe:
            return exe
    return None


def _find_msvc_compiler(arch: str = "x64") -> str | None:
    """Locate an ``cl.exe`` whose host/target toolchain matches ``arch``."""
    host_dirs = _MSVC_HOST_DIRS.get(arch, _MSVC_HOST_DIRS["x64"])
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if os.path.exists(vswhere):
        for host_dir in host_dirs:
            try:
                result = subprocess.run(
                    [
                        vswhere,
                        "-all",
                        "-find",
                        rf"VC\Tools\MSVC\**\bin\{host_dir}\cl.exe",
                    ],
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=10,
                )
                for line in result.stdout.splitlines():
                    path = line.strip()
                    if path and os.path.exists(path):
                        return path
            except (OSError, subprocess.TimeoutExpired):
                pass
    for host_dir in host_dirs:
        for pattern in [
            rf"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\{host_dir}\cl.exe",
            rf"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\{host_dir}\cl.exe",
        ]:
            matches = sorted(_glob.glob(pattern))
            if matches:
                return matches[-1]
    # Last resort: whatever `cl` happens to be on PATH (may be the wrong arch).
    return _find_exe("cl")


def _is_msvc_library(lib_dir: Path) -> bool:
    return "windows-msvc" in str(lib_dir)


def _cpp_target_arch(lib_dir: Path) -> str | None:
    """Infer the target architecture ("arm64" or "x64") from a library path."""
    parts = [part.lower() for part in lib_dir.parts]
    if any("aarch64" in part or "arm64" in part for part in parts):
        return "arm64"
    if any("x86_64" in part or "amd64" in part or "i686" in part for part in parts):
        return "x64"
    return None


def _find_vcvarsall(compiler_path: str) -> str | None:
    """Walk up from cl.exe to find vcvarsall.bat (lives at VC/Auxiliary/Build/)."""
    path = Path(compiler_path).parent
    for _ in range(10):
        candidate = path / "Auxiliary" / "Build" / "vcvarsall.bat"
        if candidate.exists():
            return str(candidate)
        path = path.parent
    return None


def _get_msvc_env(vcvarsall: str, arch: str = "x64") -> dict[str, str]:
    """Return the environment after sourcing vcvarsall.bat for ``arch``."""
    cache_key = (vcvarsall, arch)
    if cache_key in _msvc_env_cache:
        return _msvc_env_cache[cache_key]
    try:
        # `call` is required: without it, vcvarsall.bat's GOTO :EOF causes cmd.exe
        # to exit entirely, so `&& set` never runs and the env is never captured.
        result = subprocess.run(
            f'call "{vcvarsall}" {arch} > nul 2>&1 && set',
            shell=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=60,
        )
        env: dict[str, str] = {}
        for line in result.stdout.splitlines():
            key, sep, value = line.partition("=")
            if sep:
                env[key] = value
        _msvc_env_cache[cache_key] = env if env else dict(os.environ)
    except (OSError, subprocess.TimeoutExpired):
        _msvc_env_cache[cache_key] = dict(os.environ)
    return _msvc_env_cache[cache_key]


def _find_cpp_library() -> Path | None:
    candidates = [
        REPO_ROOT / "target" / "x86_64-pc-windows-msvc" / "release-c",
        REPO_ROOT / "target" / "aarch64-pc-windows-msvc" / "release-c",
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
    return run_cpp_batch([snippet], timeout)[0]


def run_cpp_batch(snippets: list[Snippet], timeout: int) -> list[RunResult]:
    """Compile and run every C++ snippet concurrently instead of one at a time.

    Each snippet already compiles/links/runs in its own temp dir, fully
    independent of the others, so the only thing serializing them was running
    the loop itself. Resolve the shared setup (compiler, library dir, MSVC
    environment) once for the whole batch, then compile+run every snippet at
    the same time instead of one g++/clang++/cl.exe invocation after another.
    """
    if not snippets:
        return []

    lib_dir = _find_cpp_library()
    if lib_dir is None:
        return [
            RunResult(
                snippet=s,
                runner="cpp",
                skipped=True,
                skip_reason="fastlowess_cpp library not built (run 'make cpp' first)",
            )
            for s in snippets
        ]

    include_dir = str(REPO_ROOT / "bindings" / "cpp" / "include")
    lib_dir_str = str(lib_dir)
    use_msvc = os.name == "nt" and _is_msvc_library(lib_dir)

    if use_msvc:
        target_arch = _cpp_target_arch(lib_dir) or "x64"
        compiler = _find_msvc_compiler(target_arch)
        if compiler is None:
            return [
                RunResult(
                    snippet=s,
                    runner="cpp",
                    skipped=True,
                    skip_reason="no MSVC cl.exe found in PATH (required for MSVC-built library)",
                )
                for s in snippets
            ]
        vcvarsall = _find_vcvarsall(compiler)
        msvc_env = (
            _get_msvc_env(vcvarsall, target_arch) if vcvarsall else dict(os.environ)
        )
        _env_path = msvc_env.get("Path") or msvc_env.get("PATH", "")
        _cl = shutil.which("cl", path=_env_path) if _env_path else None
        if _cl:
            compiler = _cl
    else:
        msvc_env = None
        compiler = _find_cpp_compiler()
        if compiler is None:
            return [
                RunResult(
                    snippet=s,
                    runner="cpp",
                    skipped=True,
                    skip_reason="no C++ compiler (g++/clang++) found in PATH",
                )
                for s in snippets
            ]

    def _run_one(snippet: Snippet) -> RunResult:
        return _compile_and_run(
            snippet, timeout, include_dir, lib_dir_str, compiler, msvc_env, use_msvc
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, len(snippets))
    ) as executor:
        return list(executor.map(_run_one, snippets))


def _compile_and_run(
    snippet: Snippet,
    timeout: int,
    include_dir: str,
    lib_dir_str: str,
    compiler: str,
    msvc_env: dict[str, str] | None,
    use_msvc: bool,
) -> RunResult:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        src = os.path.join(tmpdir, "snippet.cpp")
        exe = os.path.join(tmpdir, "snippet.exe" if os.name == "nt" else "snippet")
        with open(src, "w", encoding="utf-8") as f:
            f.write(snippet.code)

        if use_msvc:
            import_lib = "fastlowess_cpp.dll.lib"
            if not (Path(lib_dir_str) / import_lib).exists():
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
                f"/Fo:{tmpdir}{os.sep}",
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
                cwd=tmpdir,
                capture_output=True,
                check=False,
                timeout=60,
                encoding="utf-8",
                errors="replace",
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
                encoding="utf-8",
                errors="replace",
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
