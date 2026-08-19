"""One-time downloader/installer for the opt-in GPU-enabled fastlowess build.

The GPU backend (wgpu) is not included in the wheels published to PyPI. This
module fetches a prebuilt GPU-enabled wheel from the matching GitHub Release
and installs it in place of the current (CPU-only) installation.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .__version__ import __version__

_REPO = "thisisamirv/lowess-project"
_API_URL = f"https://api.github.com/repos/{_REPO}/releases/tags/v{__version__}"
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
            f"Could not find a GitHub release for fastlowess v{__version__} "
            f"({_API_URL}): {e}"
        ) from e
    return data.get("assets", [])


def _find_gpu_wheel_asset() -> dict:
    assets = _fetch_release_assets()
    candidates = [
        a
        for a in assets
        if a["name"].endswith(".whl")
        and "gpu" in a["name"].lower()
        and _asset_matches_python(a["name"])
        and _asset_matches_platform(a["name"])
    ]
    if not candidates:
        raise RuntimeError(
            "No matching GPU wheel found in the fastlowess "
            f"v{__version__} release for this platform/Python version. "
            "You may need to build it locally instead — see "
            "https://lowess.readthedocs.io/api/python/#gpu-acceleration"
        )
    return candidates[0]


def install_gpu(yes: bool = False) -> None:
    """Download and install the GPU-enabled fastlowess build for this platform.

    Fetches a prebuilt wheel (built with the ``gpu`` Cargo feature) from the
    matching GitHub Release over HTTPS and installs it in place of the
    current (CPU-only) installation via pip. Restart the Python process
    afterwards — a loaded native extension cannot be swapped in place.

    Parameters
    ----------
    yes : bool
        Skip the interactive confirmation prompt. Must be True when stdin
        is not an interactive terminal.
    """
    if gpu_available():
        print("GPU backend is already installed.")
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
            urllib.request.urlopen(req, timeout=300) as resp,
            open(wheel_path, "wb") as f,
        ):
            f.write(resp.read())

        print("Installing...")
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

    print(
        "GPU backend installed. Restart your Python process/kernel for the "
        "change to take effect."
    )


def _cli() -> None:
    """Entry point for the `fastlowess-install-gpu` console script."""
    install_gpu()
