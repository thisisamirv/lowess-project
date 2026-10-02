import zipfile
from pathlib import Path

import fastlowess
import pytest
from fastlowess import _gpu_installer


def test_linux_gpu_asset_matching_rejects_musl(monkeypatch):
    monkeypatch.setattr(_gpu_installer, "_current_platform_tag", lambda: "linux")
    monkeypatch.setattr(_gpu_installer, "_current_arch_tag", lambda: "x86_64")
    monkeypatch.setattr(_gpu_installer, "_linux_libc_tag", lambda: "musl")

    assert not _gpu_installer._asset_matches_platform(
        "fastlowess-gpu-4.1.0-cp38-abi3-manylinux_2_28_x86_64.whl"
    )


def test_local_cpu_wheel_is_rejected(tmp_path):
    extension = Path(fastlowess._core.__file__)
    wheel_path = tmp_path / "fastlowess-gpu-test.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.write(extension, f"fastlowess/{extension.name}")

    assert not _gpu_installer._wheel_gpu_enabled(wheel_path)
    with pytest.raises(RuntimeError, match="does not report GPU support"):
        _gpu_installer.install_gpu(yes=True, local_path=str(wheel_path))
