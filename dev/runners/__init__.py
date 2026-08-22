"""Language runner registry."""

from __future__ import annotations

from collections.abc import Callable

from . import cpp, julia, nodejs, python, r, rust, wasm
from .cpp import run_cpp
from .julia import run_julia
from .nodejs import run_nodejs
from .python import run_python
from .r import run_r
from .rust import run_rust
from .wasm import run_wasm

RUNNERS: dict[str, Callable] = {
    "python": run_python,
    "julia": run_julia,
    "nodejs": run_nodejs,
    "r": run_r,
    "wasm": run_wasm,
    "rust": run_rust,
    "cpp": run_cpp,
}

SKIP_CHECKS: dict[str, Callable] = {
    "python": python.skip_reason,
    "julia": julia.skip_reason,
    "nodejs": nodejs.skip_reason,
    "r": r.skip_reason,
    "wasm": wasm.skip_reason,
    "rust": rust.skip_reason,
    "cpp": cpp.skip_reason,
}
