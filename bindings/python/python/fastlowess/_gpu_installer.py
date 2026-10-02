"""One-time downloader/installer for the opt-in GPU-enabled fastlowess build.

The GPU backend (wgpu) is not included in the wheels published to PyPI. This
module fetches a prebuilt GPU-enabled wheel from the "gpu-builds" GitHub
Release (a perpetual release holding GPU artifacts for every version, so
individual version release pages stay uncluttered). Candidate wheels are
probed for GPU support before installation. Windows installs a versioned
extension sidecar so it never overwrites a loaded .pyd file.
"""

from __future__ import annotations

import importlib.machinery
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .__version__ import __version__

_REPO = "thisisamirv/lowess-project"
_GPU_RELEASE_TAG = "gpu-builds"
_API_URL = f"https://api.github.com/repos/{_REPO}/releases/tags/{_GPU_RELEASE_TAG}"
_USER_AGENT = f"fastlowess/{__version__} (+https://github.com/{_REPO})"


def gpu_available() -> bool:
    """Return True if this installation was built with the GPU backend enabled."""
    from . import _core

    return _core.gpu_enabled()


def _current_platform_tag() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _current_arch_tag() -> str:
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        return "x86_64"
    if machine in ("arm64", "aarch64"):
        return "aarch64"
    return machine


def _linux_libc_tag() -> str | None:
    if sys.platform != "linux":
        return None
    for directory in (
        Path("/lib"),
        Path("/usr/lib"),
        Path("/lib64"),
        Path("/usr/lib64"),
    ):
        if any(directory.glob("ld-musl-*.so.1")):
            return "musl"
    try:
        if any(
            is_musl_loader(line)
            for line in Path("/proc/self/maps").read_text().splitlines()
        ):
            return "musl"
    except OSError:
        pass
    libc_name = platform.libc_ver()[0].lower()
    if "musl" in libc_name:
        return "musl"
    if "glibc" in libc_name or "gnu libc" in libc_name:
        return "glibc"
    try:
        if os.confstr("CS_GNU_LIBC_VERSION"):
            return "glibc"
    except (AttributeError, OSError, ValueError):
        pass
    return None


def is_musl_loader(line: str) -> bool:
    return "ld-musl-" in line or "libc.musl-" in line


def _asset_matches_platform(name: str) -> bool:
    name = name.lower()
    plat = _current_platform_tag()
    arch = _current_arch_tag()

    if plat == "windows":
        if arch == "aarch64":
            return "win_arm64" in name
        return "win_amd64" in name or "win32" in name
    if plat == "macos":
        if "macosx" not in name:
            return False
        if arch == "aarch64":
            return "arm64" in name or "universal2" in name
        return "x86_64" in name or "universal2" in name
    # linux
    if "linux" not in name:
        return False
    if _linux_libc_tag() != "glibc":
        return False
    return arch in name


def _asset_matches_python(name: str) -> bool:
    name = name.lower()
    if "abi3" in name:
        return True
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return tag in name


def _fetch_release_assets() -> list[dict]:
    req = urllib.request.Request(
        _API_URL,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Could not find the '{_GPU_RELEASE_TAG}' GitHub release ({_API_URL}): {e}"
        ) from e
    return data.get("assets", [])


def _asset_matches_version(name: str) -> bool:
    return f"-gpu-{__version__}-" in name.lower()


def _extract_core_extension(wheel_path: Path, directory: Path) -> Path:
    try:
        with zipfile.ZipFile(wheel_path) as wheel:
            candidates = [
                name
                for name in wheel.namelist()
                if name.startswith("fastlowess/_core")
                and any(
                    name.endswith(suffix)
                    for suffix in importlib.machinery.EXTENSION_SUFFIXES
                )
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    "wheel must contain exactly one fastlowess._core extension"
                )
            member = candidates[0]
            extension_path = directory / Path(member).name
            with wheel.open(member) as source, extension_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            return extension_path
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"Could not read GPU wheel {wheel_path}: {error}") from error


def _extension_gpu_enabled(extension_path: Path) -> bool:
    script = """
import ctypes, importlib.util, os, sys, types
if os.name == "nt":
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x8000)
package = types.ModuleType("fastlowess")
package.__path__ = []
sys.modules["fastlowess"] = package
spec = importlib.util.spec_from_file_location("fastlowess._core", sys.argv[1])
if spec is None or spec.loader is None:
    raise RuntimeError("could not load candidate extension")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(bool(module.gpu_enabled()))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(extension_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "True"


def _wheel_gpu_enabled(wheel_path: Path) -> bool:
    with tempfile.TemporaryDirectory() as temporary_directory:
        extension_path = _extract_core_extension(wheel_path, Path(temporary_directory))
        return _extension_gpu_enabled(extension_path)


def _install_gpu_wheel(wheel_path: Path) -> None:
    if not _wheel_gpu_enabled(wheel_path):
        raise RuntimeError(f"The wheel at {wheel_path} does not report GPU support.")

    if sys.platform.startswith("win"):
        with tempfile.TemporaryDirectory() as temporary_directory:
            extension_path = _extract_core_extension(
                wheel_path, Path(temporary_directory)
            )
            sidecar = Path(__file__).with_name(f"_core_gpu_{__version__}.pyd")
            staging = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp.pyd")
            try:
                shutil.copyfile(extension_path, staging)
                if not _extension_gpu_enabled(staging):
                    raise RuntimeError(
                        f"The wheel at {wheel_path} does not report GPU support."
                    )
                os.replace(staging, sidecar)
            finally:
                staging.unlink(missing_ok=True)
        return

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            str(wheel_path),
        ],
        check=True,
    )


def _find_gpu_wheel_asset() -> dict:
    assets = _fetch_release_assets()
    candidates = [
        a
        for a in assets
        if a["name"].endswith(".whl")
        and _asset_matches_version(a["name"])
        and _asset_matches_python(a["name"])
        and _asset_matches_platform(a["name"])
    ]
    if not candidates:
        raise RuntimeError(
            f"No matching GPU wheel found for fastlowess v{__version__} in the "
            f"'{_GPU_RELEASE_TAG}' release for this platform/Python version. "
            "You may need to build it locally instead — see "
            "https://lowess.readthedocs.io/api/python/#gpu-acceleration"
        )
    return candidates[0]


def install_gpu(yes: bool = False, local_path: str | None = None) -> None:
    """Install the GPU-enabled fastlowess build for this platform.

    Fetches a prebuilt wheel (built with the ``gpu`` Cargo feature) matching
    this installation's version from the "gpu-builds" GitHub Release over
    HTTPS and verifies that its native extension reports GPU support. On
    Windows the extension is installed as a versioned sidecar to avoid
    replacing the currently loaded .pyd file; on other platforms pip installs
    the wheel normally. Restart the Python process afterwards.

    Prebuilt Linux GPU wheels currently target glibc/manylinux, not musl/Alpine.

    Parameters
    ----------
    yes : bool
        Skip the interactive confirmation prompt. Must be True when stdin
        is not an interactive terminal.
    local_path : str, optional
        Path to a GPU-enabled wheel already built locally (e.g. via
        ``maturin build --release --features gpu``). When given, skips the
        GitHub Release lookup/download and installs directly from this
        path — useful for testing the installer itself, or installing an
        unreleased build.
    """
    if gpu_available():
        print("GPU backend is already installed.")
        return

    if local_path is not None:
        wheel_path = Path(local_path)
        if not wheel_path.is_file():
            raise RuntimeError(f"No such file: {wheel_path}")

        if not yes:
            if not sys.stdin.isatty():
                raise RuntimeError(
                    "install_gpu() requires confirmation. Pass yes=True to "
                    "proceed non-interactively."
                )
            answer = input(
                f"Install {wheel_path} in place of the current build? [y/N] "
            )
            if answer.strip().lower() not in ("y", "yes"):
                print("Aborted.")
                return

        print(f"Installing {wheel_path} ...")
        _install_gpu_wheel(wheel_path)
        print(
            "GPU backend installed. Restart your Python process/kernel for the "
            "change to take effect."
        )
        return

    asset = _find_gpu_wheel_asset()
    size_mb = asset.get("size", 0) / (1024 * 1024)

    if not yes:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "install_gpu() requires confirmation. Pass yes=True to "
                "proceed non-interactively."
            )
        answer = input(
            f"Download and install {asset['name']} ({size_mb:.1f} MB) from "
            f"github.com/{_REPO}? [y/N] "
        )
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return

    url = asset["browser_download_url"]
    print(f"Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with tempfile.TemporaryDirectory() as tmp:
        wheel_path = Path(tmp) / asset["name"]
        with (
            urllib.request.urlopen(req, timeout=300) as response,
            wheel_path.open("wb") as wheel_file,
        ):
            shutil.copyfileobj(response, wheel_file)

        print("Installing...")
        _install_gpu_wheel(wheel_path)

    print(
        "GPU backend installed. Restart your Python process/kernel for the "
        "change to take effect."
    )


def _cli() -> None:
    """Entry point for the `fastlowess-install-gpu` console script."""
    install_gpu()
