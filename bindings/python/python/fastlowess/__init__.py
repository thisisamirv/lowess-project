"""
fastlowess: High-performance LOWESS Smoothing for Python.

A high-performance LOWESS (Locally Weighted Scatterplot Smoothing) implementation
with parallel execution via Rayon and NumPy integration. Built on top of the
fastLowess Rust crate.
"""

import importlib.util
import sys
import warnings
from importlib import import_module
from pathlib import Path

from .__version__ import __version__


def _load_core():
    if sys.platform.startswith("win"):
        sidecar = Path(__file__).with_name(f"_core_gpu_{__version__}.pyd")
        if sidecar.is_file():
            module_name = f"{__name__}._core"
            try:
                spec = importlib.util.spec_from_file_location(module_name, sidecar)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load GPU extension at {sidecar}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                if module.gpu_enabled():
                    return module
            except (ImportError, OSError) as error:
                warnings.warn(f"Could not load GPU extension at {sidecar}: {error}")
            sys.modules.pop(module_name, None)
    return import_module("._core", __name__)


_core = _load_core()
Diagnostics = _core.Diagnostics
LowessResult = _core.LowessResult
OnlineLowess = _core.OnlineLowess
OnlineOutput = _core.OnlineOutput
PredictOutput = _core.PredictOutput
StreamingLowess = _core.StreamingLowess

from ._gpu_installer import gpu_available, install_gpu
from ._lowess import Lowess

__all__ = [
    "Diagnostics",
    "Lowess",
    "LowessResult",
    "OnlineLowess",
    "OnlineOutput",
    "PredictOutput",
    "StreamingLowess",
    "__version__",
    "gpu_available",
    "install_gpu",
]
