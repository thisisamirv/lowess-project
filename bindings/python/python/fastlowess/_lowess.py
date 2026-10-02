"""Batch Lowess wrapper adding a friendly error for the opt-in GPU backend."""

import numpy as np

from . import _core

_GPU_HELP = (
    "GPU backend not installed in this build. Run `fastlowess.install_gpu()` "
    "once to download and install a GPU-enabled build, then restart Python. "
    "See https://lowess.readthedocs.io/api/python/#gpu-acceleration for details."
)


class Lowess(_core.Lowess):
    """Batch LOWESS model — configure once, fit many times.

    See :class:`fastlowess._core.Lowess` for the full parameter list.
    """

    def __new__(cls, *args, backend: str = "cpu", **kwargs):
        if backend == "gpu" and not _core.gpu_enabled():
            raise RuntimeError(_GPU_HELP)
        return super().__new__(cls, *args, backend=backend, **kwargs)

    def fit(self, x, y, custom_weights=None):
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if custom_weights is not None:
            custom_weights = np.ascontiguousarray(custom_weights, dtype=np.float64)
        return super().fit(x, y, custom_weights)
