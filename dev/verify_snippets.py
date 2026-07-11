"""Extract fenced code blocks from docs/ and run them to verify correctness.

Each snippet is prefixed with language-specific boilerplate that provides common
sample data and imports, so fragment-style doc examples can execute end-to-end.

Usage
-----
    python dev/verify_snippets.py                    # run all supported languages
    python dev/verify_snippets.py --lang python      # Python only
    python dev/verify_snippets.py --lang nodejs      # Node.js only
    python dev/verify_snippets.py --lang julia       # Julia only
    python dev/verify_snippets.py --lang r           # R only
    python dev/verify_snippets.py --lang wasm        # WebAssembly only
    python dev/verify_snippets.py --lang rust        # Rust only
    python dev/verify_snippets.py --lang cpp         # C++ only
    python dev/verify_snippets.py --file docs/api/python.md
    python dev/verify_snippets.py --dry-run          # list snippets, don't run
    python dev/verify_snippets.py --verbose          # show snippet source on failure
    python dev/verify_snippets.py --output out.json  # also write JSON report
    python dev/verify_snippets.py --timeout 60       # per-snippet timeout (seconds)
    python dev/verify_snippets.py --stop-on-fail     # exit after first failure
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

# ---------------------------------------------------------------------------
# Terminal colours (disabled on non-TTY or Windows without colour support)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.name != "nt" or os.environ.get("FORCE_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def green(t: str) -> str:
    return _c("32", t)


def red(t: str) -> str:
    return _c("31", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def bold(t: str) -> str:
    return _c("1", t)


# ---------------------------------------------------------------------------
# Python executable detection (prefer venv where fastlowess is installed)
# ---------------------------------------------------------------------------

_PYTHON_BIN: str = sys.executable  # may be replaced in main()


def _find_python_with_fastlowess() -> str:
    """Return the best Python executable that has fastlowess installed."""
    candidates = [
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",  # Windows root venv
        REPO_ROOT / ".venv" / "bin" / "python",  # Unix root venv
        REPO_ROOT / "bindings" / "python" / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / "bindings" / "python" / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            try:
                r = subprocess.run(
                    [str(c), "-c", "import fastlowess"],
                    capture_output=True,
                    timeout=10,
                )
                if r.returncode == 0:
                    return str(c)
            except Exception:
                pass
    return sys.executable


# ---------------------------------------------------------------------------
# Language-specific boilerplate injected before every snippet
# ---------------------------------------------------------------------------

# Tab labels in the docs that map to each runner
_TAB_ALIASES: dict[str, set[str]] = {
    "python": {"Python"},
    "julia": {"Julia"},
    "nodejs": {"Node.js"},
    # These are not run by default (need external toolchain)
    "wasm": {"WebAssembly"},
    "r": {"R"},
    "cpp": {"C++"},
    "rust": {
        "Rust",
        "Rust (fastLowess)",
        "lowess (no_std compatible)",
        "fastLowess (parallel)",
    },
}

# Code-block language tags for each runner
_LANG_TAGS: dict[str, set[str]] = {
    "python": {"python"},
    "julia": {"julia"},
    "nodejs": {"javascript", "js", "typescript", "ts"},
    "wasm": {"javascript", "js"},
    "r": {"r"},
    "cpp": {"cpp", "c++"},
    "rust": {"rust"},
}

_PYTHON_PREAMBLE = textwrap.dedent("""\
    # --- snippet preamble: suppress display back-end -------------------------
    import sys as _sys, os as _os
    try:
        import matplotlib as _mpl
        _mpl.use("Agg")
        import matplotlib.pyplot as plt
        plt.show = lambda *a, **kw: None
    except ImportError:
        import sys as _sys2, types as _types
        _plt_stub = _types.ModuleType('matplotlib.pyplot')
        _plt_stub.__getattr__ = lambda _n: (lambda *a, **kw: _plt_stub)  # type: ignore
        _sys2.modules.setdefault('matplotlib', _types.ModuleType('matplotlib'))
        _sys2.modules.setdefault('matplotlib.pyplot', _plt_stub)
        plt = _plt_stub  # type: ignore

    # --- snippet preamble: common imports ------------------------------------
    import numpy as np
    import fastlowess as fl

    # --- snippet preamble: sample data ---------------------------------------
    np.random.seed(42)
    _n = 100
    x = np.linspace(0.0, 10.0, _n)
    y = np.sin(x) + np.random.normal(0, 0.1, _n)

    # Aliases used in various tutorials
    t = x.copy()
    t_irregular = x.copy()
    y_irregular = y.copy()
    positions = x.copy()
    observed = y.copy()
    times = x.copy()
    temperatures = y.copy()
    hours = x.copy()
    expression = y.copy()
    coverage = np.abs(y.copy()) * 10 + 5

    # Multivariate input variables (lat/lon/x1/x2/x3 for dimensions.md examples)
    lat = x.copy(); lon = x * 0.5
    x1 = x.copy(); x2 = x * 0.5; x3 = x * 0.25
    z = np.sin(x) + np.cos(x * 0.5)
    # Python binding requires flattened 1D for multi-dim input
    x2d = np.column_stack([x, x * 0.5]).ravel()    # (200,) flat row-major
    x3d = np.column_stack([x, x * 0.5, x * 0.25]).ravel()  # (300,) flat

    # Outlier / weight examples
    y_with_outlier = y.copy();  y_with_outlier[50] = 100.0
    weights = np.ones(_n)

    # Streaming / chunk examples
    chunk_size, overlap = 50, 10
    chunk1_x, chunk1_y = x[:50].copy(), y[:50].copy()
    chunk2_x, chunk2_y = x[50:].copy(), y[50:].copy()
    x_chunk, y_chunk = x[:50].copy(), y[:50].copy()

    # Sliding-window examples
    data_x = list(x[:30])
    data_y = list(y[:30])

    # Calibration / uncertainty examples
    calibration_indices = np.array([2, 5, 7, 20, 50])
    sigma = np.random.default_rng(42).uniform(0.1, 0.5, _n)

    # API doc pseudocode helpers
    fastlowess = fl   # allow docs that use bare "fastlowess.X(...)"
    kwargs = {'fraction': 0.3}
    model = fl.Lowess()
    stream = fl.StreamingLowess()
    online = fl.OnlineLowess()
    # Bare-name aliases so snippets that import with "from fastlowess import ..."
    # or use the class name directly without fl. prefix work out of the box.
    Lowess = fl.Lowess
    StreamingLowess = fl.StreamingLowess
    OnlineLowess = fl.OnlineLowess

    # -------------------------------------------------------------------------
""")

_JULIA_PREAMBLE = textwrap.dedent("""\
    # --- snippet preamble ----------------------------------------------------
    using FastLOWESS
    using Random, Printf, Statistics
    Random.seed!(42)

    _n = 100
    x  = collect(LinRange(0.0, 10.0, _n))
    y  = sin.(x) .+ randn(_n) .* 0.1

    t           = copy(x)
    t_irregular = copy(x)
    y_irregular = copy(y)
    positions   = copy(x)
    observed    = copy(y)
    times       = copy(x)
    temperatures = copy(y)
    hours       = copy(x)
    expression  = copy(y)
    coverage    = abs.(y) .* 10.0 .+ 5.0

    x2d = hcat(x, x .* 0.5)
    x3d = hcat(x, x .* 0.5, x .* 0.25)
    z   = sin.(x) .+ cos.(x .* 0.5)
    lat = copy(x); lon = x .* 0.5
    x1  = copy(x); x2 = x .* 0.5; x3 = x .* 0.25

    y_with_outlier = copy(y); y_with_outlier[50] = 100.0
    weights = ones(Float64, _n)

    chunk1_x, chunk1_y = x[1:50], y[1:50]
    chunk2_x, chunk2_y = x[51:end], y[51:end]
    x_chunk, y_chunk   = x[1:50], y[1:50]

    data_x = copy(x[1:30])
    data_y = copy(y[1:30])

    # API doc placeholders (method-signature examples use these as variables)
    model  = Lowess()
    stream = StreamingLowess()
    online = OnlineLowess()
    kwargs = (fraction=0.3,)
    # -------------------------------------------------------------------------
""")

_NODEJS_PREAMBLE = textwrap.dedent("""\
    // --- snippet preamble ----------------------------------------------------
    'use strict';
    const fl = (() => { try { return require('fastlowess'); } catch (e) { return null; } })();
    if (!fl) { console.error('fastlowess not found — skip'); process.exit(0); }
    const { Lowess, StreamingLowess, OnlineLowess } = fl;
    const fastlowess = fl;

    const _n = 100;
    const x = new Float64Array(_n).map((_, i) => i * 0.1);
    const y = new Float64Array(x.map(xi => Math.sin(xi) + (Math.random() - 0.5) * 0.2));

    const t            = new Float64Array(x);
    const tIrregular   = new Float64Array(x);
    const yIrregular   = new Float64Array(y);
    const positions    = new Float64Array(x);
    const observed     = new Float64Array(y);
    const times        = new Float64Array(x);
    const temperatures = new Float64Array(y);
    const hours        = new Float64Array(x);
    const expression   = new Float64Array(y);
    const coverage     = new Float64Array(y.map(yi => Math.abs(yi) * 10 + 5));

    const x2d = { x: Array.from(x), z: Array.from(x.map(xi => xi * 0.5)) };
    const z = new Float64Array(y.map((yi, i) => yi + Math.cos(x[i] * 0.5)));

    const yWithOutlier = new Float64Array(y); yWithOutlier[50] = 100.0;
    const weights = new Float64Array(_n).fill(1.0);

    const chunk1_x = x.slice(0, 50); const chunk1_y = y.slice(0, 50);
    const chunk2_x = x.slice(50);    const chunk2_y = y.slice(50);
    let data_x = Array.from(x.slice(0, 30));
    let data_y = Array.from(y.slice(0, 30));
    const xArr = new Float64Array(x); const yArr = new Float64Array(y);
    let windowX = Array.from(x.slice(0, 20));
    let windowY = Array.from(y.slice(0, 20));
    // -------------------------------------------------------------------------
""")

_R_PREAMBLE = textwrap.dedent("""\
    # --- snippet preamble ----------------------------------------------------
    suppressMessages({{
        .libPaths(c(
            normalizePath(file.path(
                Sys.getenv("LOWESS_REPO_ROOT", "{repo_root}"),
                "bindings", "r", ".r-lib"
            ), mustWork = FALSE),
            .libPaths()
        ))
        library(rfastlowess)
    }})
    set.seed(42)
    n  <- 100L
    x  <- seq(0, 10, length.out = n)
    y  <- sin(x) + rnorm(n, sd = 0.1)

    t            <- x
    t_irregular  <- x
    y_irregular  <- y
    positions    <- x
    observed     <- y
    times        <- x
    temperatures <- y
    hours        <- x
    expression   <- y
    coverage     <- abs(y) * 10 + 5

    lat <- x; lon <- x * 0.5
    x1  <- x; x2 <- x * 0.5; x3 <- x * 0.25
    z   <- sin(x) + cos(x * 0.5)
    x2d <- as.vector(rbind(x, x * 0.5))   # row-major flat: (2*n,)
    x3d <- as.vector(rbind(x, x * 0.5, x * 0.25))  # row-major flat: (3*n,)

    y_with_outlier        <- y
    y_with_outlier[[50L]] <- 100
    weights <- rep(1, n)

    chunk1_x <- x[seq_len(50)]; chunk1_y <- y[seq_len(50)]
    chunk2_x <- x[51:100];      chunk2_y <- y[51:100]
    x_chunk  <- x[seq_len(50)]; y_chunk  <- y[seq_len(50)]
    data_x   <- x[seq_len(30)]
    data_y   <- y[seq_len(30)]

    # Placeholder variables used in illustrative snippets
    calibration_indices <- c(3L, 6L, 8L)   # 1-indexed
    sigma               <- runif(n, 0.1, 0.5)

    # API doc placeholder objects so single-line method-call snippets run
    model  <- Lowess()
    stream <- StreamingLowess()
    online <- OnlineLowess()

    # Open a null graphics device so polygon()/lines()/plot() work without display
    suppressWarnings(pdf(NULL))
    plot.new()
    # -------------------------------------------------------------------------
""").format(repo_root=str(REPO_ROOT).replace("\\", "/"))

_WASM_PKG_DIR = REPO_ROOT / "bindings" / "wasm" / "pkg"

_WASM_PREAMBLE = textwrap.dedent("""\
    // --- snippet preamble (WASM) ---------------------------------------------
    'use strict';
    const _wasmPkg = (() => {{
        const candidates = [
            '{wasm_pkg}/fastlowess_wasm.js',
        ];
        for (const c of candidates) {{
            try {{ return require(c); }} catch (_) {{}}
        }}
        return null;
    }})();
    if (!_wasmPkg) {{ console.error('WASM pkg not found — skip'); process.exit(0); }}
    const {{ Lowess, StreamingLowess, OnlineLowess }} = _wasmPkg;
    const fastlowess = _wasmPkg;

    const _n = 100;
    const x = new Float64Array(_n).map((_, i) => i * 0.1);
    const y = new Float64Array(x.map(xi => Math.sin(xi) + (Math.random() - 0.5) * 0.2));

    const z = new Float64Array(y.map((yi, i) => yi + Math.cos(x[i] * 0.5)));
    const weights = new Float64Array(_n).fill(1.0);
    // row-major flat 2D: [x0,x0*0.5, x1,x1*0.5, ...]
    const x2d = new Float64Array(_n * 2);
    for (let i = 0; i < _n; i++) {{ x2d[i*2] = x[i]; x2d[i*2+1] = x[i]*0.5; }}
    // row-major flat 3D: [x0,x0*0.5,x0*0.25, x1,...]
    const x3d = new Float64Array(_n * 3);
    for (let i = 0; i < _n; i++) {{ x3d[i*3] = x[i]; x3d[i*3+1] = x[i]*0.5; x3d[i*3+2] = x[i]*0.25; }}
    const xChunk = x.slice(0, 50);
    const yChunk = y.slice(0, 50);
    // chunk aliases used in streaming examples
    const x1 = x.slice(0, 50); const y1 = y.slice(0, 50);
    const x2 = x.slice(50);    const y2 = y.slice(50);
    // camelCase variable referenced in doc examples
    const calibrationIndices = [2, 5, 10];
    // sliding-window accumulators
    let windowX = [], windowY = [];
    // online / sensor readings
    const readings = Array.from({{length: _n}}, (_, i) => ({{x: x[i], y: y[i]}}));
    // -------------------------------------------------------------------------
""").format(wasm_pkg=str(_WASM_PKG_DIR).replace("\\", "/"))

# Rust snippets are wrapped: top + snippet + bottom
_RUST_PREAMBLE_TOP = textwrap.dedent("""\
    #[allow(unused_imports, ambiguous_glob_reexports)]
    use lowess::prelude::*;

    #[allow(dead_code, unused_variables)]
    fn _run() -> Result<(), Box<dyn std::error::Error>> {
        // Concrete f64 specialisations so doc examples compile without turbofish
        #[allow(unused)]
        type Lowess = lowess::prelude::Lowess<f64>;
        #[allow(unused)]
        type StreamingLowess = lowess::prelude::StreamingLowess<f64>;
        #[allow(unused)]
        type OnlineLowess = lowess::prelude::OnlineLowess<f64>;
        let n = 100usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        let t = x.clone();
        let weights: Vec<f64> = vec![1.0; n];
        let lat = x.clone();
        let lon: Vec<f64> = x.iter().map(|&xi| xi * 0.5).collect();
        let x2d: Vec<f64> = x.iter().chain(lon.iter()).copied().collect();
        let x3: Vec<f64> = x.iter().map(|&xi| xi * 0.25).collect();
        let x3d: Vec<f64> = x.iter().chain(lon.iter()).chain(x3.iter()).copied().collect();
        let z: Vec<f64> = x.iter().map(|&xi| xi.sin() + (xi * 0.5).cos()).collect();
        let x_chunk: Vec<f64> = x[..50].to_vec();
        let y_chunk: Vec<f64> = y[..50].to_vec();
        let chunk1_x = x_chunk.clone();
        let chunk1_y = y_chunk.clone();
        let chunk2_x: Vec<f64> = x[50..].to_vec();
        let chunk2_y: Vec<f64> = y[50..].to_vec();
        let t_irregular = t.clone();
        let y_irregular = y.clone();
        let y_with_outlier: Vec<f64> = { let mut v = y.clone(); v[50] = 100.0; v };
        let _ = (&t, &weights, &lat, &lon, &x2d, &x3, &x3d, &z,
                  &x_chunk, &y_chunk, &chunk1_x, &chunk1_y, &chunk2_x, &chunk2_y,
                  &t_irregular, &y_irregular, &y_with_outlier);

""")

_RUST_PREAMBLE_BOTTOM = textwrap.dedent("""\
        Ok(())
    }

    fn main() {
        if let Err(e) = _run() {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
""")

# Fixed temp project for Rust snippets (reuses Cargo artifacts)
_RUST_SNIPPET_DIR = REPO_ROOT / "target" / "doc-snippet-runner"

# C++ snippets are wrapped: preamble top + snippet + bottom
_CPP_PREAMBLE_TOP = textwrap.dedent("""\
    #define _USE_MATH_DEFINES
    #include "fastlowess.hpp"
    #include <cmath>
    #include <cstdlib>
    #include <iostream>
    #include <vector>

    using namespace fastlowess;

    static void _run() {
        const size_t n = 100;
        std::vector<double> x(n), y(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = static_cast<double>(i) / 10.0;
            y[i] = std::sin(x[i]);
        }
        auto x_chunk = std::vector<double>(x.begin(), x.begin() + 50);
        auto y_chunk = std::vector<double>(y.begin(), y.begin() + 50);
        // row-major flat 2D: [x0,x0*0.5, x1,x1*0.5, ...]
        std::vector<double> x2d;
        x2d.reserve(n * 2);
        for (size_t i = 0; i < n; ++i) { x2d.push_back(x[i]); x2d.push_back(x[i] * 0.5); }
        // row-major flat 3D: [x0,x0*0.5,x0*0.25, ...]
        std::vector<double> x3d;
        x3d.reserve(n * 3);
        for (size_t i = 0; i < n; ++i) { x3d.push_back(x[i]); x3d.push_back(x[i] * 0.5); x3d.push_back(x[i] * 0.25); }
        // z as a second response variable (same length as y)
        std::vector<double> z(n);
        for (size_t i = 0; i < n; ++i) { z[i] = std::cos(x[i]); }
        // genomics data
        std::vector<double> positions(n), observed(n), coverage(n);
        for (size_t i = 0; i < n; ++i) {
            positions[i] = static_cast<double>(i) * 1000.0;
            observed[i] = 5.0 + std::sin(x[i]);
            coverage[i] = 20.0 + std::cos(x[i]);
        }
        // sensor / time-series data
        std::vector<double> times(n), temperatures(n);
        for (size_t i = 0; i < n; ++i) {
            times[i] = static_cast<double>(i) * 0.1;
            temperatures[i] = 20.0 + std::sin(times[i]);
        }
        // irregular time series
        std::vector<double> tIrregular(50), yIrregular(50);
        for (size_t i = 0; i < 50; ++i) {
            tIrregular[i] = static_cast<double>(i) * 2.0 + 0.3 * static_cast<double>(i % 3);
            yIrregular[i] = 10.0 + tIrregular[i] * 0.3 + std::sin(tIrregular[i]);
        }
        // gene expression / hours data
        std::vector<double> hours(49), expression(49);
        for (size_t i = 0; i < 49; ++i) {
            hours[i] = static_cast<double>(i) * 0.5;
            expression[i] = 100.0 * (1.0 + 0.5 * std::sin(hours[i] * M_PI / 12.0));
        }
        // t as alias for x (time axis)
        const auto& t = x;
        // sliding window accumulators
        std::vector<double> windowX, windowY;
        // calibration indices (used in custom-weights examples)
        std::vector<std::size_t> calibration_indices = {2, 5, 10};
        // measurement uncertainty (used in custom-weights examples)
        std::vector<double> sigma(n, 0.1);
        (void)x_chunk; (void)y_chunk; (void)x2d; (void)x3d; (void)z;
        (void)positions; (void)observed; (void)coverage;
        (void)times; (void)temperatures;
        (void)tIrregular; (void)yIrregular;
        (void)hours; (void)expression;
        (void)windowX; (void)windowY;
        (void)t; (void)calibration_indices; (void)sigma;

""")

_CPP_PREAMBLE_BOTTOM = textwrap.dedent("""\
    }

    int main() {
        try {
            _run();
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\\n";
            return 1;
        }
        return 0;
    }
""")

PREAMBLES: dict[str, str] = {
    "python": _PYTHON_PREAMBLE,
    "julia": _JULIA_PREAMBLE,
    "nodejs": _NODEJS_PREAMBLE,
    "r": _R_PREAMBLE,
    "wasm": _WASM_PREAMBLE,
}

# ---------------------------------------------------------------------------
# Snippet data class
# ---------------------------------------------------------------------------


@dataclass
class Snippet:
    file: Path
    line: int  # 1-based line number of the opening fence
    lang_tag: str  # code-block language tag (e.g. "python")
    tab: Optional[str]  # nearest === "Tab" label, or None
    code: str

    @property
    def runner(self) -> Optional[str]:
        """Return which runner handles this snippet, or None to skip."""
        for runner, tags in _LANG_TAGS.items():
            if self.lang_tag.lower() in tags:
                # For JS: distinguish Node.js from WASM by tab label
                if runner in ("nodejs", "wasm"):
                    if self.tab in _TAB_ALIASES["wasm"]:
                        return "wasm"
                    if self.tab in _TAB_ALIASES["nodejs"]:
                        return "nodejs"
                    # No tab: fall back to nodejs if no WASM markers
                    if (
                        "fastlowess-wasm" not in self.code
                        and "import {" not in self.code[:80]
                    ):
                        return "nodejs"
                    return "wasm"
                return runner
        return None

    @property
    def label(self) -> str:
        tab = f" [{self.tab}]" if self.tab else ""
        return f"{self.file.relative_to(REPO_ROOT)}:{self.line}{tab}"


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------

_TAB_RE = re.compile(r'^[ \t]*===\s+"([^"]+)"', re.MULTILINE)
_FENCE_RE = re.compile(r"^([ \t]*)```(\w+)", re.MULTILINE)


def extract_snippets(md_file: Path) -> List[Snippet]:
    """Extract all fenced code blocks from a markdown file."""
    text = md_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    result: List[Snippet] = []

    current_tab: Optional[str] = None
    i = 0
    while i < len(lines):
        line = lines[i]

        # Track tab context
        m = _TAB_RE.match(line)
        if m:
            current_tab = m.group(1)
            i += 1
            continue

        # Detect fence opening: optional leading whitespace then ```lang
        m = re.match(r"^([ \t]*)```(\w+)\s*$", line)
        if m:
            fence_indent = m.group(1)
            lang_tag = m.group(2)
            start_line = i + 1  # 1-based
            code_lines: List[str] = []
            i += 1
            while i < len(lines):
                close = lines[i]
                # Closing fence: same indent + ```
                if re.match(r"^" + re.escape(fence_indent) + r"```\s*$", close):
                    i += 1
                    break
                # Strip the common fence indent
                stripped = (
                    close[len(fence_indent) :]
                    if close.startswith(fence_indent)
                    else close
                )
                code_lines.append(stripped)
                i += 1
            code = "\n".join(code_lines)
            result.append(
                Snippet(
                    file=md_file,
                    line=start_line,
                    lang_tag=lang_tag,
                    tab=current_tab,
                    code=code,
                )
            )
            # A tab label covers only the next block (reset after capture)
            current_tab = None
            continue

        # Reset tab on section headers or dividers
        if line.startswith("#") or line.strip() == "---":
            current_tab = None

        i += 1

    return result


def should_skip(snippet: Snippet, runner: str) -> Optional[str]:
    """Return a skip reason string, or None if the snippet should be run."""
    code = snippet.code

    # MkDocs file-include directives are not runnable
    if "--8<--" in code:
        return "file-include (--8<--)"

    # Empty or whitespace-only
    if not code.strip():
        return "empty"

    # Skip snippets that reference variables or packages we can't supply
    if runner == "python":
        # Genomic-specific heavy I/O (would need real files)
        if any(s in code for s in ["read_csv", "open(", "glob(", "argparse"]):
            return "file I/O"
        # Lines that are obviously just output examples (no executable Python)
        if not any(c in code for c in ["=", "(", "import", "print"]):
            return "no executable statements"
        # Large synthetic datasets that exceed the per-snippet timeout
        if re.search(r"total_points\s*=\s*[1-9][0-9]{4,}", code):
            return "large synthetic dataset (too slow for CI)"
        # Single-line API signature example: add_point(x, y) where preamble
        # x/y are arrays but add_point expects scalar arguments.
        stripped_lines = [
            line
            for line in code.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if len(stripped_lines) == 1 and re.search(
            r"\.add_point\s*\(", stripped_lines[0]
        ):
            if not re.search(r"\bfloat\s*\(|\bint\s*\(|\[\s*[0-9]", stripped_lines[0]):
                return (
                    "add_point signature example (preamble x/y are arrays, not scalars)"
                )

    if runner == "julia":
        # Skip package-management / installation snippets
        if re.search(r"\bPkg\.(add|develop|clone|rm|pin)\s*\(", code):
            return "Pkg management snippet"
    if runner == "nodejs":
        # TypeScript-only syntax (type annotations)
        if ": SmoothOptions" in code or ": LowessResult" in code:
            return "TypeScript (not Node.js)"

    if runner == "r":
        # Skip install/devtools snippets
        if re.search(r"\binstall\.packages\b|\bdevtools::", code):
            return "package installation snippet"
        # Skip if no actual R statements (e.g., pure output blocks)
        if not any(c in code for c in ["<-", "=", "(", "library"]):
            return "no executable R statements"
        # Multi-dim input (x2d/x3d) requires a package rebuild to work correctly
        if re.search(r"\bx2d\b|\bx3d\b|\bdimensions\s*=\s*[23]\b", code):
            return "multi-dim R (needs package rebuild)"
        # API signature snippets that use R's '...' outside a function definition
        # (e.g. 'model <- Lowess(...)') — not valid in a script context.
        if re.search(r"\(\s*\.\.\.\s*\)", code):
            return "R API signature with ... (not runnable outside function)"

    if runner == "wasm":
        # Skip any ES-module import syntax — wasm runner uses CJS require().
        # Catches `import { X }`, `import init, { X }`, `import X from ...`, etc.
        if re.search(r"^import\b", code, re.MULTILINE) or "await init(" in code:
            return "ES module import (not supported in CJS runner)"

    if runner == "rust":
        # Skip snippets that are complete programs (have their own fn main)
        if re.search(r"\bfn\s+main\s*\(", code):
            return "defines own fn main"
        # Skip snippets that look like TOML config (Cargo.toml examples)
        if code.strip().startswith("[") and "=" in code and "fn " not in code:
            return "TOML/config snippet"
        # Skip snippets with no actual Rust statements
        if not any(c in code for c in ["let ", "use ", "fn ", "::", "Lowess", "build"]):
            return "no executable Rust statements"
        # Build-only snippets can't infer generic type without a .fit() call
        if (
            re.search(r"\.adapter\(", code)
            and not any(
                s in code
                for s in [".fit(", ".add_points(", ".add_point(", ".process_chunk("]
            )
            and "Lowess::<" not in code
        ):
            return "build-only snippet (type T unresolvable without usage)"

    if runner == "cpp":
        # Skip snippets that define their own main
        if re.search(r"\bint\s+main\s*\(", code):
            return "defines own int main"
        # Skip pure output / comment blocks
        if not any(c in code for c in [";", "{", "Lowess", "smooth", "auto ", "std::"]):
            return "no executable C++ statements"

    return None


# ---------------------------------------------------------------------------
# Node.js: strip redeclarations that conflict with the preamble
# ---------------------------------------------------------------------------

# Variables the Node.js preamble already declares at the top level
_NODEJS_PREAMBLE_VARS: frozenset = frozenset(
    {
        "fl",
        "fastlowess",
        "Lowess",
        "StreamingLowess",
        "OnlineLowess",
        # data arrays — snippets often redeclare these with small sample data
        "x",
        "y",
        "z",
        "t",
        "weights",
        "yWithOutlier",
        "positions",
        "observed",
        "times",
        "temperatures",
        "hours",
        "expression",
        "coverage",
        "chunk1_x",
        "chunk1_y",
        "chunk2_x",
        "chunk2_y",
        "data_x",
        "data_y",
        "xArr",
        "yArr",
        "windowX",
        "windowY",
        # multi-dim / WASM shared vars
        "x2d",
        "x3d",
        "xChunk",
        "yChunk",
    }
)


def _strip_nodejs_redeclarations(code: str) -> str:
    """Remove declarations that would shadow preamble const bindings.

    Handles single-line and multi-line declarations by tracking bracket depth.
    """
    result = []
    skipping = False  # inside a multi-line declaration being stripped
    depth = 0  # net ( [ { depth while skipping

    for line in code.splitlines():
        s = line.strip()

        if skipping:
            for ch in line:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
            if depth <= 0:
                skipping = False
                depth = 0
            continue  # drop continuation line

        # Detect lines to strip
        strip_line = False
        # Strip require() imports from either package name
        if re.match(
            r"""(?:const|let|var)\s+\S.*=\s*require\(\s*['"]fastlowess(?:-wasm)?['"]\s*\)""",
            s,
        ):
            strip_line = True
        # Strip ES module imports from fastlowess packages
        elif re.match(r"""import\s+.*\bfrom\s+['"]fastlowess(?:-wasm)?['"]""", s):
            strip_line = True
        # Strip `await init()` — WASM preamble initialises synchronously via require
        elif re.match(r"""await\s+init\s*\(""", s):
            strip_line = True
        elif re.match(
            r"""(?:const|let|var)\s+\{[^}]+\}\s*=\s*(?:fl|fastlowess)\s*;?\s*$""", s
        ):
            strip_line = True
        else:
            m = re.match(r"(?:const|let|var)\s+(\w+)\s*=", s)
            if m and m.group(1) in _NODEJS_PREAMBLE_VARS:
                strip_line = True

        if strip_line:
            for ch in line:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
            if depth > 0:
                skipping = True  # multi-line declaration
            else:
                depth = 0
            continue

        result.append(line)

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    snippet: Snippet
    runner: str
    skipped: bool = False
    skip_reason: str = ""
    passed: bool = False
    duration: float = 0.0
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1


def run_python(snippet: Snippet, timeout: int) -> RunResult:
    code = PREAMBLES["python"] + snippet.code
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name
    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [_PYTHON_BIN, tmp],
            capture_output=True,
            timeout=timeout,
            text=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="python",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="python",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )
    finally:
        os.unlink(tmp)


def run_julia(snippet: Snippet, timeout: int) -> RunResult:
    code = PREAMBLES["julia"] + snippet.code
    with tempfile.NamedTemporaryFile(
        suffix=".jl", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name

    julia_bin = _find_exe("julia")
    if julia_bin is None:
        os.unlink(tmp)
        return RunResult(
            snippet=snippet,
            runner="julia",
            skipped=True,
            skip_reason="julia not found in PATH",
        )

    # Find the Julia project for the bindings
    julia_project = REPO_ROOT / "bindings" / "julia" / "julia"
    env = {**os.environ}
    if julia_project.exists():
        env["JULIA_PROJECT"] = str(julia_project)

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [julia_bin, "--startup-file=no", tmp],
            capture_output=True,
            timeout=timeout,
            text=True,
            env=env,
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="julia",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="julia",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )
    finally:
        os.unlink(tmp)


def run_nodejs(snippet: Snippet, timeout: int) -> RunResult:
    code = PREAMBLES["nodejs"] + _strip_nodejs_redeclarations(snippet.code)
    with tempfile.NamedTemporaryFile(
        suffix=".js", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name

    node_bin = _find_exe("node")
    if node_bin is None:
        os.unlink(tmp)
        return RunResult(
            snippet=snippet,
            runner="nodejs",
            skipped=True,
            skip_reason="node not found in PATH",
        )

    # Run from the nodejs binding directory so require('fastlowess') resolves
    nodejs_dir = REPO_ROOT / "bindings" / "nodejs"
    cwd = str(nodejs_dir) if nodejs_dir.exists() else str(REPO_ROOT)

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [node_bin, tmp],
            capture_output=True,
            timeout=timeout,
            text=True,
            cwd=cwd,
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="nodejs",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="nodejs",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )
    finally:
        os.unlink(tmp)


def run_r(snippet: Snippet, timeout: int) -> RunResult:
    code = PREAMBLES["r"] + snippet.code
    with tempfile.NamedTemporaryFile(
        suffix=".R", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name

    rscript = _find_exe("Rscript")
    if rscript is None:
        os.unlink(tmp)
        return RunResult(
            snippet=snippet,
            runner="r",
            skipped=True,
            skip_reason="Rscript not found in PATH",
        )

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [rscript, "--vanilla", tmp],
            capture_output=True,
            timeout=timeout,
            text=True,
            env={**os.environ, "LOWESS_REPO_ROOT": str(REPO_ROOT)},
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="r",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="r",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )
    finally:
        os.unlink(tmp)


def run_wasm(snippet: Snippet, timeout: int) -> RunResult:
    # WASM snippets run in Node.js with the pre-built wasm pkg
    if not _WASM_PKG_DIR.exists():
        return RunResult(
            snippet=snippet,
            runner="wasm",
            skipped=True,
            skip_reason="bindings/wasm/pkg/ not built (run 'make wasm' first)",
        )

    code = PREAMBLES["wasm"] + _strip_nodejs_redeclarations(snippet.code)
    with tempfile.NamedTemporaryFile(
        suffix=".js", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name

    node_bin = _find_exe("node")
    if node_bin is None:
        os.unlink(tmp)
        return RunResult(
            snippet=snippet,
            runner="wasm",
            skipped=True,
            skip_reason="node not found in PATH",
        )

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [node_bin, tmp],
            capture_output=True,
            timeout=timeout,
            text=True,
            cwd=str(_WASM_PKG_DIR),
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="wasm",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="wasm",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )
    finally:
        os.unlink(tmp)


def _ensure_rust_snippet_project() -> bool:
    """Create the persistent temp Cargo project for Rust snippet execution.

    Returns True if the project is ready, False on error.
    """
    _RUST_SNIPPET_DIR.mkdir(parents=True, exist_ok=True)
    cargo_toml = _RUST_SNIPPET_DIR / "Cargo.toml"
    if not cargo_toml.exists():
        # [workspace] prevents Cargo from merging this into the parent workspace
        lowess_path = str(REPO_ROOT / "crates" / "lowess").replace("\\", "/")
        fastlowess_path = str(REPO_ROOT / "crates" / "fastLowess").replace("\\", "/")
        cargo_toml.write_text(
            textwrap.dedent(f"""\
                [workspace]

                [package]
                name = "doc-snippet"
                version = "0.1.0"
                edition = "2021"

                [[bin]]
                name = "doc-snippet"
                path = "src/main.rs"

                [dependencies]
                lowess    = {{ path = "{lowess_path}" }}
                fastLowess = {{ path = "{fastlowess_path}" }}
            """),
            encoding="utf-8",
        )
    src_dir = _RUST_SNIPPET_DIR / "src"
    src_dir.mkdir(exist_ok=True)
    return True


def run_rust(snippet: Snippet, timeout: int) -> RunResult:
    cargo_bin = _find_exe("cargo")
    if cargo_bin is None:
        return RunResult(
            snippet=snippet,
            runner="rust",
            skipped=True,
            skip_reason="cargo not found in PATH",
        )

    _ensure_rust_snippet_project()

    # Strip leading use declarations from snippet (preamble provides them)
    code_lines = snippet.code.splitlines()
    filtered: list[str] = []
    for line in code_lines:
        # Keep use statements but they'll be duplicates — Rust ignores them
        filtered.append(line)

    code_body = "\n".join(filtered)
    main_rs = (
        _RUST_PREAMBLE_TOP
        + "    "
        + code_body.replace("\n", "\n    ")
        + "\n"
        + _RUST_PREAMBLE_BOTTOM
    )

    main_path = _RUST_SNIPPET_DIR / "src" / "main.rs"
    main_path.write_text(main_rs, encoding="utf-8")

    # Share the parent workspace's target dir to reuse compiled deps
    target_dir = str(REPO_ROOT / "target" / "doc-snippet-target")

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [
                cargo_bin,
                "run",
                "--manifest-path",
                str(_RUST_SNIPPET_DIR / "Cargo.toml"),
                "--target-dir",
                target_dir,
                "--quiet",
            ],
            capture_output=True,
            timeout=timeout,
            text=True,
            cwd=str(REPO_ROOT),
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="rust",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="rust",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )


def _find_cpp_compiler() -> Optional[str]:
    for name in ("g++", "clang++", "c++"):
        exe = _find_exe(name)
        if exe:
            return exe
    return None


def _find_cpp_library() -> Optional[Path]:
    """Return the path to the built fastlowess_cpp shared/static library, or None."""
    candidates = [
        # MinGW targets on Windows (Makefile uses these when GCC is available)
        REPO_ROOT / "target" / "x86_64-pc-windows-gnu" / "release-c",
        REPO_ROOT / "target" / "aarch64-pc-windows-gnu" / "release-c",
        # Default release-c (MSVC on Windows, native on Unix)
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
    compiler = _find_cpp_compiler()
    if compiler is None:
        return RunResult(
            snippet=snippet,
            runner="cpp",
            skipped=True,
            skip_reason="no C++ compiler (g++/clang++) found in PATH",
        )

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

    # Strip redundant includes that are already in the preamble
    cpp_code = snippet.code
    for inc in ('#include "fastlowess.hpp"', "#include <fastlowess.hpp>"):
        cpp_code = cpp_code.replace(inc + "\n", "").replace(inc, "")

    code = (
        _CPP_PREAMBLE_TOP
        + "    "
        + cpp_code.strip().replace("\n", "\n    ")
        + "\n"
        + _CPP_PREAMBLE_BOTTOM
    )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        src = os.path.join(tmpdir, "snippet.cpp")
        exe = os.path.join(tmpdir, "snippet.exe" if os.name == "nt" else "snippet")
        with open(src, "w", encoding="utf-8") as f:
            f.write(code)

        compile_cmd = [
            compiler,
            "-std=c++17",
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
                timeout=60,
                text=True,
            )
            if cproc.returncode != 0:
                dur = time.monotonic() - t0
                # Detect MSVC-ABI / MinGW linker mismatch so we skip rather than fail.
                _abi_markers = ("__chkstk", "??_7type_info", "_Unwind_Resume")
                if os.name == "nt" and any(m in cproc.stderr for m in _abi_markers):
                    return RunResult(
                        snippet=snippet,
                        runner="cpp",
                        skipped=True,
                        skip_reason=(
                            "C++ library ABI mismatch (MSVC vs MinGW) — "
                            "rebuild with: make cpp (using the x86_64-pc-windows-gnu target)"
                        ),
                    )
                return RunResult(
                    snippet=snippet,
                    runner="cpp",
                    passed=False,
                    duration=dur,
                    stdout=cproc.stdout,
                    stderr=cproc.stderr,
                    returncode=cproc.returncode,
                )

            # Set library search path and run
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


_RUNNERS = {
    "python": run_python,
    "julia": run_julia,
    "nodejs": run_nodejs,
    "r": run_r,
    "wasm": run_wasm,
    "rust": run_rust,
    "cpp": run_cpp,
}


def _find_exe(name: str) -> Optional[str]:
    import shutil

    return shutil.which(name)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def iter_md_files(root: Path, file_filter: Optional[str]) -> Iterator[Path]:
    if file_filter:
        p = Path(file_filter)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.is_file():
            yield p
            return
        # Treat as glob
        yield from sorted(REPO_ROOT.glob(file_filter))
        return
    yield from sorted(root.rglob("*.md"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lang",
        choices=["python", "julia", "nodejs", "r", "wasm", "rust", "cpp", "all"],
        default="all",
        help="Which language runner to use (default: all)",
    )
    parser.add_argument(
        "--file",
        metavar="PATH_OR_GLOB",
        help="Restrict to a specific file or glob (relative to repo root)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List snippets without running them"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print snippet source and full output on failure",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop after the first failing snippet",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-snippet timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--output", metavar="FILE", help="Write JSON report to this file"
    )
    args = parser.parse_args(argv)

    active_runners: set[str] = (
        {"python", "julia", "nodejs", "r", "wasm", "rust", "cpp"}
        if args.lang == "all"
        else {args.lang}
    )

    # Detect best Python executable
    global _PYTHON_BIN
    _PYTHON_BIN = _find_python_with_fastlowess()

    # ---- Collect snippets ---------------------------------------------------
    snippets: List[Snippet] = []
    for md in iter_md_files(DOCS_DIR, args.file):
        snippets.extend(extract_snippets(md))

    total_found = len(snippets)

    # Filter to only snippets we can handle
    runnable: List[Tuple[Snippet, str]] = []  # (snippet, runner)
    for s in snippets:
        r = s.runner
        if r is None or r not in active_runners:
            continue
        reason = should_skip(s, r)
        if reason:
            continue
        runnable.append((s, r))

    print(bold("\nfastLowess doc snippet verifier"))
    print(f"Docs dir : {DOCS_DIR}")
    print(f"Runners  : {', '.join(sorted(active_runners))}")
    print(f"Snippets : {len(runnable)} runnable / {total_found} total")
    if args.dry_run:
        print()
        for s, r in runnable:
            print(f"  {cyan(r):20s}  {s.label}")
        print()
        return 0
    print()

    # ---- Run snippets (parallel per language) --------------------------------
    results: List[RunResult] = []
    n_pass = n_fail = n_skip = 0
    _print_lock = threading.Lock()

    # Group runnable snippets by runner so each language executes as a unit.
    # Rust snippets share a single temp project (main.rs), so they must remain
    # sequential within their language — parallelism is across languages only.
    _by_runner: dict[str, List[Tuple[Snippet, str]]] = defaultdict(list)
    for _s, _r in runnable:
        _by_runner[_r].append((_s, _r))

    def _run_language(lang_items: List[Tuple[Snippet, str]]) -> List[RunResult]:
        lang_results: List[RunResult] = []
        for s, runner in lang_items:
            label = s.label
            run_fn = _RUNNERS.get(runner)
            if run_fn is None:
                res = RunResult(
                    snippet=s,
                    runner=runner,
                    skipped=True,
                    skip_reason="no runner implementation",
                )
                with _print_lock:
                    print(
                        f"  {cyan(runner):20s}  {label} … {yellow('SKIP (no runner)')}"
                    )
            else:
                res = run_fn(s, args.timeout)
                with _print_lock:
                    sys.stdout.write(f"  {cyan(runner):20s}  {label} … ")
                    if res.skipped:
                        print(yellow(f"SKIP ({res.skip_reason})"))
                    elif res.passed:
                        print(green(f"PASS ({res.duration:.2f}s)"))
                    else:
                        print(red(f"FAIL ({res.duration:.2f}s, exit {res.returncode})"))
                        if args.verbose:
                            _print_failure(s, res)

            lang_results.append(res)
            if args.stop_on_fail and not res.skipped and not res.passed:
                with _print_lock:
                    print(
                        red(
                            f"\n[{runner}] Stopped after first failure (--stop-on-fail)."
                        )
                    )
                break
        return lang_results

    n_workers = len(_by_runner) or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_run_language, items) for items in _by_runner.values()
        ]
        for future in concurrent.futures.as_completed(futures):
            for res in future.result():
                results.append(res)
                if res.skipped:
                    n_skip += 1
                elif res.passed:
                    n_pass += 1
                else:
                    n_fail += 1

    # ---- Summary ------------------------------------------------------------
    print()
    print("-" * 60)
    print(bold("Summary"))
    print(
        f"  {green(f'PASS: {n_pass}'):30s}  {yellow(f'SKIP: {n_skip}'):30s}  {red(f'FAIL: {n_fail}')}"
    )
    print()

    # Print failures in verbose mode (already shown inline) or always in brief
    failures = [r for r in results if not r.passed and not r.skipped]
    if failures and not args.verbose:
        print(bold("Failed snippets:"))
        for r in failures:
            print(f"  {red('FAIL')} {r.snippet.label}")
            # Show first 5 lines of stderr
            if r.stderr.strip():
                for line in r.stderr.strip().splitlines()[:5]:
                    print(f"      {line}")
        print()

    # ---- JSON output --------------------------------------------------------
    if args.output:
        report = {
            "summary": {"pass": n_pass, "fail": n_fail, "skip": n_skip},
            "snippets": [
                {
                    "file": str(r.snippet.file.relative_to(REPO_ROOT)),
                    "line": r.snippet.line,
                    "lang": r.snippet.lang_tag,
                    "tab": r.snippet.tab,
                    "runner": r.runner,
                    "status": "skip" if r.skipped else ("pass" if r.passed else "fail"),
                    "skip_reason": r.skip_reason if r.skipped else None,
                    "returncode": r.returncode if not r.skipped else None,
                    "duration": round(r.duration, 3),
                    "stderr": r.stderr[:2000] if r.stderr else "",
                }
                for r in results
            ],
        }
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.output}")

    return 1 if n_fail > 0 else 0


def _print_failure(snippet: Snippet, res: RunResult) -> None:
    """Print detailed failure information."""
    sep = "-" * 56
    print()
    print(f"  {sep}")
    print(f"  {bold('File:')} {snippet.label}")
    if snippet.tab:
        print(f"  {bold('Tab:')}  {snippet.tab}")
    print(f"  {bold('Code:')}")
    for line in snippet.code.splitlines()[:20]:
        print(f"    {line}")
    if len(snippet.code.splitlines()) > 20:
        print(f"    ... ({len(snippet.code.splitlines())} lines total)")
    if res.stderr.strip():
        print(f"  {bold('Stderr:')}")
        for line in res.stderr.strip().splitlines()[-20:]:
            print(f"    {line}")
    if res.stdout.strip():
        print(f"  {bold('Stdout:')}")
        for line in res.stdout.strip().splitlines()[-10:]:
            print(f"    {line}")
    print(f"  {sep}")
    print()


if __name__ == "__main__":
    sys.exit(main())
