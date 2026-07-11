#!/usr/bin/env python3
"""
fix_doc_snippets.py — Transform doc snippets to be self-contained.

Adds minimal boilerplate to each code snippet that would otherwise be skipped
because it lacks imports, fn main, int main, require(), etc.

Usage:
    python3 dev/fix_doc_snippets.py [--dry-run] [--lang LANG]

Without --dry-run, modifies the doc files in place.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dev.verify_snippets import (  # noqa: E402
    DOCS_DIR,
    Snippet,
    extract_snippets,
    should_skip,
)

# ---------------------------------------------------------------------------
# Standard boilerplate per language
# ---------------------------------------------------------------------------

# Python: standard imports + 100-pt noisy sine wave
_PY_STD = """\
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

"""

# Python: when the snippet uses matplotlib (time-series tutorial)
_PY_TIMESERIES = """\
import fastlowess as fl
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
t = np.linspace(0, 100, 500)
trend_true = 10 + 0.5 * t + 3 * np.sin(t / 10)
y = trend_true + np.random.normal(0, 3, len(t))

"""

# Python: genomics tutorial (uses positions / coverage)
_PY_GENOMICS = """\
import fastlowess as fl
import numpy as np

np.random.seed(42)
positions = np.arange(0, 10000, 10, dtype=float)
coverage = np.random.poisson(50, len(positions)).astype(float)

"""

# Python: merge.md (uses x_chunk / y_chunk)
_PY_MERGE = """\
import fastlowess as fl
import numpy as np

x_chunk = np.linspace(0, np.pi, 50)
y_chunk = np.sin(x_chunk) + np.random.default_rng(42).normal(0, 0.1, 50)

"""

# Python: custom-weights.md extra vars (calibration_indices / sigma / Lowess)
_PY_CUSTOM_WEIGHTS_EXTRA = """\
import fastlowess as fl
from fastlowess import Lowess
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)
calibration_indices = [5, 20, 40, 60, 80]
sigma = rng.uniform(0.1, 0.5, 100)

"""

# Node.js: standard require + data
_JS_STD = """\
const {{ {imports} }} = require('fastlowess');

const n = 100;
const x = Float64Array.from({{ length: n }}, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

"""

# Node.js: merge.md (uses xChunk / yChunk)
_JS_MERGE = """\
const { StreamingLowess } = require('fastlowess');

const n = 50;
const xChunk = Float64Array.from({ length: n }, (_, i) => i * Math.PI / (n - 1));
const yChunk = Float64Array.from(xChunk, xi => Math.sin(xi));

"""

# Node.js: time-series (uses t / y)
_JS_TIMESERIES = """\
const fl = require('fastlowess');

const n = 500;
const t = Float64Array.from({ length: n }, (_, i) => i * 100 / (n - 1));
const y = Float64Array.from(t, ti => 10 + 0.5 * ti + 3 * Math.sin(ti / 10) + (Math.random() - 0.5) * 6);

"""

# Node.js: custom-weights.md (uses calibrationIndices / sigma)
_JS_CUSTOM_WEIGHTS = """\
const fastlowess = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + (Math.random() - 0.5) * 0.6);
const calibrationIndices = [5, 20, 40, 60, 80];
const sigma = Float64Array.from({ length: n }, () => 0.1 + Math.random() * 0.4);

"""

# WASM: standard require (run from _WASM_PKG_DIR)
_WASM_STD_REQUIRE = "./fastlowess_wasm.js"

# Rust: fn main wrapper
_RS_TOP = """\
use fastLowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {{
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

"""
_RS_BOTTOM = """
    Ok(())
}}
"""

# C++: int main wrapper
_CPP_TOP = """\
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {{
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {{
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }}

"""
_CPP_BOTTOM = """
    return 0;
}}
"""


# ---------------------------------------------------------------------------
# Per-snippet transformation functions
# ---------------------------------------------------------------------------

# R standard data used by most snippets
_R_STD = """\
library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

"""

_R_TIMESERIES = """\
library(rfastlowess)
set.seed(42)
t <- seq(0, 100, length.out = 500)
trend_true <- 10 + 0.5 * t + 3 * sin(t / 10)
y <- trend_true + rnorm(500, sd = 3)

"""

_R_GENOMICS = """\
library(rfastlowess)
set.seed(42)
positions <- seq(0, 10000, by = 10)
observed <- 50 + sin(positions / 100) * 20 + rnorm(length(positions), sd = 5)

"""


def _r_needs_data(code: str) -> bool:
    """Return True if an R snippet uses x/y/t without defining them."""
    import re as _re

    defines_x = bool(_re.search(r"(?:^|\n)\s*x\s*<-", code))
    defines_y = bool(_re.search(r"(?:^|\n)\s*y\s*<-", code))
    defines_t = bool(_re.search(r"(?:^|\n)\s*t\s*<-", code))
    # Common usage patterns
    uses_x = bool(_re.search(r"\$fit\(x[,\)]|\bx,\s*y\b", code))
    uses_y = bool(_re.search(r"\bfit\(x,\s*y\b", code))
    uses_t = bool(_re.search(r"\$fit\(t[,\)]|\bfit\(t,\s*y\b", code))
    return (
        (uses_x and not defines_x)
        or (uses_y and not defines_y)
        or (uses_t and not defines_t)
    )


def transform_r(code: str, doc_rel: str) -> str | None:
    """Add R boilerplate to snippets that lack library() or data definitions."""
    import re as _re

    has_lib = "library(" in code
    has_x = bool(_re.search(r"(?:^|\n)\s*x\s*<-", code))
    has_t = bool(_re.search(r"(?:^|\n)\s*t\s*<-", code))

    uses_t = bool(_re.search(r"\$fit\(t[,\)]|\bt,\s*y\b", code))
    uses_pos = "positions" in code or "observed" in code

    if has_lib and has_x:
        return None  # already self-contained

    # Build preamble
    if uses_pos:
        preamble = _R_GENOMICS
    elif uses_t and not has_t:
        preamble = _R_TIMESERIES
    else:
        preamble = _R_STD

    # Handle extra tutorial vars
    extra: list[str] = []
    if "x_chunk" in code and "x_chunk <-" not in code:
        extra.append("x_chunk <- x[seq_len(50)]")
        extra.append("y_chunk <- y[seq_len(50)]")
    if "chunk1_x" in code and "chunk1_x <-" not in code:
        extra.append("chunk1_x <- x[seq_len(50)]; chunk1_y <- y[seq_len(50)]")
        extra.append("chunk2_x <- x[51:100]; chunk2_y <- y[51:100]")
    if "positions" in code and "positions <-" not in code and uses_pos:
        pass  # already handled by _R_GENOMICS above

    if extra:
        preamble = preamble.rstrip("\n") + "\n" + "\n".join(extra) + "\n\n"

    return preamble + code


# Julia standard data
_JL_STD = """\
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2\u03c0, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

"""

_JL_TIMESERIES = """\
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
t = collect(range(0, 100, length=500))
y = 10.0 .+ 0.5 .* t .+ 3.0 .* sin.(t ./ 10.0) .+ randn(rng, 500) .* 3.0

"""

_JL_GENOMICS = """\
using FastLOWESS
using Random

rng = MersenneTwister(42)
positions = collect(0.0:10.0:10000.0)
observed = 50.0 .+ sin.(positions ./ 100.0) .* 20.0 .+ randn(rng, length(positions)) .* 5.0

"""


def transform_julia(code: str, doc_rel: str) -> str | None:
    """Add Julia boilerplate to snippets that lack using/data definitions."""
    import re as _re

    has_using = "using FastLOWESS" in code
    has_x = bool(_re.search(r"^x\s*=", code, _re.MULTILINE))
    has_t = bool(_re.search(r"^t\s*=", code, _re.MULTILINE))

    uses_t = bool(_re.search(r"\bfit\(\w+,\s*t\b|\bt,\s*y\b", code))
    uses_pos = "positions" in code or "observed" in code

    if has_using and (has_x or has_t):
        return None  # already self-contained

    # Choose preamble
    if uses_pos:
        preamble = _JL_GENOMICS
    elif uses_t and not has_t:
        preamble = _JL_TIMESERIES
    else:
        preamble = _JL_STD

    # Handle tutorial-specific extra variables
    extra: list[str] = []
    if (
        "calibration_indices" in code
        and "calibration_indices" not in code.split("=")[0]
    ):
        extra.append("calibration_indices = [5, 20, 40, 60, 80]")
    if _re.search(r"\bsigma\b", code) and "sigma =" not in code:
        extra.append("sigma = rand(rng, 100) .* 0.4 .+ 0.1")
    if "x_chunk" in code and "x_chunk =" not in code:
        extra.append("x_chunk = x[1:50]; y_chunk = y[1:50]")

    if extra:
        preamble = preamble.rstrip("\n") + "\n" + "\n".join(extra) + "\n\n"

    return preamble + code

    """Detect which fastlowess symbols are used in a Node.js snippet."""
    names = []
    for sym in ("Lowess", "StreamingLowess", "OnlineLowess", "smooth"):
        if re.search(r"\b" + sym + r"\b", code):
            names.append(sym)
    if not names:
        names = ["Lowess"]
    return names


def transform_python(code: str, doc_rel: str) -> str | None:
    """Return transformed Python code, or None if no transformation needed."""
    if "import fastlowess" in code or "from fastlowess" in code:
        return (
            None  # already has an import — but may still need data (handled by caller)
        )

    # Choose the right boilerplate based on what the snippet uses
    if "positions" in code or "coverage" in code:
        preamble = _PY_GENOMICS
    elif "plt." in code or "matplotlib" in code:
        preamble = _PY_TIMESERIES
    elif (
        "t," in code
        or "\nt " in code
        or code.startswith("t ")
        or " t\n" in code
        or re.search(r"\bt\b", code[:50])
    ):
        # Uses variable `t` (time) rather than `x`
        preamble = _PY_TIMESERIES
    elif "x_chunk" in code or "y_chunk" in code:
        preamble = _PY_MERGE
    elif (
        "calibration_indices" in code
        or "sigma" in code
        or (re.search(r"\bLowess\s*\(", code) and "fl." not in code)
    ):
        preamble = _PY_CUSTOM_WEIGHTS_EXTRA
    else:
        preamble = _PY_STD
        # Add numpy if needed but not present
        if "np." in code and "import numpy" not in code:
            pass  # already included in _PY_STD

    return preamble + code


def transform_python_existing(code: str, doc_rel: str) -> str | None:
    """Transform Python snippets that already have imports but missing data variables."""
    # Check if snippet has imports but uses undefined variables
    has_import = bool(re.search(r"^(import|from)\s", code, re.MULTILINE))
    if not has_import:
        return None

    # If snippet uses x_chunk / y_chunk and doesn't define them
    if (
        ("x_chunk" in code or "y_chunk" in code)
        and "x_chunk" not in code.split("import")[0]
        and "x_chunk =" not in code
    ):
        # Add the chunk variables if not already defined
        if "x_chunk =" not in code:
            extra = (
                "import numpy as np\n\n"
                "x_chunk = np.linspace(0, np.pi, 50)\n"
                "y_chunk = np.sin(x_chunk) + np.random.default_rng(42).normal(0, 0.1, 50)\n\n"
            )
            # Remove duplicate numpy import
            if "import numpy" in code:
                extra = extra.replace("import numpy as np\n\n", "")
            return extra + code

    return None


def _detect_js_imports(code: str) -> list[str]:
    """Detect which fastlowess symbols are used in a Node.js snippet."""
    names = [
        sym
        for sym in ("Lowess", "StreamingLowess", "OnlineLowess", "smooth")
        if re.search(r"\b" + sym + r"\b", code)
    ]
    return names or ["Lowess"]


def _js_add_missing_data(code: str, doc_rel: str) -> str | None:
    """For a Node.js snippet that already has require(), add missing data variables.

    Returns updated code, or None if no changes are needed.
    """
    original = code

    # Fix: snippet uses fl. but fl is not bound (e.g. const { Lowess } = require(...))
    if re.search(r"\bfl\.", code) and not re.search(r"\bconst\s+fl\s*=", code):
        code = re.sub(
            r"const\s*\{[^}]+\}\s*=\s*require\('fastlowess'\);",
            "const fl = require('fastlowess');",
            code,
        )

    # ---- Detect tutorial-specific missing variables ----
    extra: list[str] = []

    def _missing(name: str) -> bool:
        """True if `name` is referenced but not declared in this snippet."""
        return bool(re.search(r"\b" + re.escape(name) + r"\b", code)) and not re.search(
            r"(?:const|let|var)\s+" + re.escape(name) + r"\s*[=[]", code
        )

    # Tutorial-specific variables
    if _missing("dataChunks"):
        extra.append(
            "const dataChunks = Array.from({ length: 5 }, (_, ci) => ({\n"
            "    x: Float64Array.from({ length: 20 }, (_, i) => ci * 20 + i),\n"
            "    y: Float64Array.from({ length: 20 }, (_, i) => Math.sin((ci * 20 + i) * 0.1))\n"
            "}));"
        )
    if _missing("tIrregular"):
        extra.append(
            "const tIrregular = Float64Array.from({ length: 200 }, () => Math.random() * 100).sort((a,b)=>a-b);\n"
            "const yIrregular = Float64Array.from(tIrregular, t => 10 + 0.3 * t + Math.random() * 2);"
        )
    if _missing("hours"):
        extra.append(
            "const hours = Float64Array.from({ length: 49 }, (_, i) => i * 0.5);\n"
            "const expression = Float64Array.from(hours, h => 100*(1+0.5*Math.sin(h*Math.PI/12))+(Math.random()-0.5)*20);"
        )
    if _missing("positions"):
        extra.append(
            "const positions = Float64Array.from({ length: 1000 }, (_, i) => i * 10.0);\n"
            "const observed = Float64Array.from(positions, p => 50 + Math.sin(p/100)*20 + Math.random()*5);"
        )
    if _missing("genomicData"):
        extra.append(
            "const genomicData = {\n"
            "    positions: Float64Array.from({ length: 100 }, (_, i) => i * 100.0),\n"
            "    methylation: Float64Array.from({ length: 100 }, () => 0.5 + (Math.random()-0.5)*0.3)\n"
            "};"
        )
    if _missing("chunk1_x"):
        extra.append(
            "const chunk1_x = Float64Array.from({ length: 50 }, (_, i) => i);\n"
            "const chunk1_y = Float64Array.from(chunk1_x, v => Math.sin(v * 0.1));\n"
            "const chunk2_x = Float64Array.from({ length: 50 }, (_, i) => i + 50);\n"
            "const chunk2_y = Float64Array.from(chunk2_x, v => Math.sin(v * 0.1));"
        )
    if _missing("sensorStream"):
        # Add a minimal EventEmitter-like stub
        extra.append(
            "const EventEmitter = require('events');\n"
            "const sensorStream = new EventEmitter();\n"
            "// Emit sample readings immediately after snippet runs\n"
            "setImmediate(() => { for (let i = 0; i < 10; i++) sensorStream.emit('data', { t: i, value: Math.sin(i * 0.3) }); });"
        )

    # ---- Standard x / y / t data ----
    defines_x = bool(re.search(r"(?:const|let|var)\s+x\s*[=[]", code))
    defines_y = bool(re.search(r"(?:const|let|var)\s+y\s*[=[]", code))

    defines_n = bool(re.search(r"(?:const|let|var)\s+n\s*=", code))

    uses_t = _missing("t")
    uses_x = _missing("x")
    uses_y = _missing("y")

    if uses_t:
        extra.insert(
            0,
            "const n = 500;\n"
            "const t = Float64Array.from({ length: n }, (_, i) => i * 100 / (n - 1));\n"
            "const y = Float64Array.from(t, ti => 10 + 0.5 * ti + 3 * Math.sin(ti / 10) + (Math.random()-0.5)*6);",
        )
    elif uses_x or uses_y:
        n_line = "" if defines_n else "const n = 100;\n"
        x_line = (
            ""
            if defines_x
            else "const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));\n"
        )
        y_line = (
            ""
            if defines_y
            else "const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);\n"
        )
        if n_line or x_line or y_line:
            extra.insert(0, (n_line + x_line + y_line).rstrip())

    if not extra and code == original:
        return None  # nothing to change

    # Insert extra definitions after the last require() line
    lines = code.splitlines()
    last_require = max(
        (i for i, ln in enumerate(lines) if "require(" in ln),
        default=0,
    )
    insert_block = "\n" + "\n".join(extra)
    lines.insert(last_require + 1, insert_block)
    return "\n".join(lines)


def transform_nodejs(code: str, doc_rel: str) -> str | None:
    """Return transformed Node.js code, or None if already has require."""
    if "require(" in code:
        # Already has require — but might still need data variables or fl. fix
        return _js_add_missing_data(code, doc_rel)

    # ---- Detect context for preamble selection ----
    if "xChunk" in code or "yChunk" in code:
        preamble = _JS_MERGE
    elif re.search(r"\btIrregular\b|\bhours\b", code):
        preamble = _JS_TIMESERIES
    elif re.search(r"\bt\b", code) and re.search(r"\.fit\(t[, \)]", code):
        preamble = _JS_TIMESERIES
    elif "calibrationIndices" in code or re.search(r"\bsigma\b", code):
        preamble = _JS_CUSTOM_WEIGHTS
    elif re.search(r"\bfl\.", code):
        # Code uses fl.Xxx namespace
        need_data = re.search(r"\bx\b|\by\b", code) and not re.search(
            r"(?:const|let|var)\s+[xy]\s*[=[]", code
        )
        preamble = "const fl = require('fastlowess');\n\n"
        if need_data:
            preamble += (
                "const n = 100;\n"
                "const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));\n"
                "const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);\n\n"
            )
    elif "fastlowess." in code:
        # Code uses fastlowess.Xxx namespace
        need_data = re.search(r"\bx\b|\by\b", code) and not re.search(
            r"(?:const|let|var)\s+[xy]\s*[=[]", code
        )
        preamble = "const fastlowess = require('fastlowess');\n\n"
        if need_data:
            preamble += (
                "const n = 100;\n"
                "const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));\n"
                "const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);\n\n"
            )
    else:
        imports = _detect_js_imports(code)
        # Don't add x/y if snippet defines them itself
        if re.search(r"(?:const|let|var)\s+[xnt]\s*=", code, re.MULTILINE):
            return (
                f"const {{ {', '.join(imports)} }} = require('fastlowess');\n\n" + code
            )
        preamble = _JS_STD.format(imports=", ".join(imports))

    # Also check if tutorial-specific variables need to be added after preamble
    # (these get inserted by _js_add_missing_data after transformation)
    return preamble + code


def transform_wasm(code: str, doc_rel: str) -> str | None:
    """Convert WASM snippets from ES module to CJS, or add require()."""
    if "require(" in code:
        return None

    # Detect what's imported in ES module style
    es_imports: set[str] = set()
    for m in re.finditer(
        r"^import\s+(?:init\s*,\s*)?\{([^}]+)\}\s+from\s+['\"]fastlowess[_-]?wasm['\"];?",
        code,
        re.MULTILINE,
    ):
        for name in m.group(1).split(","):
            es_imports.add(name.strip())

    # Also handle: import { Lowess } from "fastlowess_wasm.js"
    for m in re.finditer(
        r"^import\s+(?:init\s*,\s*)?\{([^}]+)\}\s+from\s+['\"][^'\"]+['\"];?",
        code,
        re.MULTILINE,
    ):
        for name in m.group(1).split(","):
            es_imports.add(name.strip())

    # Remove ES module import lines
    code = re.sub(
        r"^import\s+(?:init\s*,\s*)?(?:\{[^}]+\}\s+)?from\s+['\"][^'\"]+['\"];?\n?",
        "",
        code,
        flags=re.MULTILINE,
    )
    # Remove bare `import init ...` lines
    code = re.sub(
        r"^import\s+\w+\s*,\s*\{[^}]+\}\s+from.*\n?", "", code, flags=re.MULTILINE
    )

    # Remove await init() calls
    code = re.sub(r"^\s*await\s+init\s*\(\s*\)\s*;?\n?", "", code, flags=re.MULTILINE)

    # If the snippet has its own data definitions, just add the require
    has_own_data = bool(
        re.search(r"(?:^|[\n;])\s*(?:const|let|var)\s+x\s*=", code, re.MULTILINE)
    )

    # Determine what to import from the WASM package
    if not es_imports:
        # Detect from usage
        for sym in (
            "Lowess",
            "StreamingLowess",
            "OnlineLowess",
            "smooth",
            "smoothStreaming",
            "smoothOnline",
        ):
            if re.search(r"\b" + sym + r"\b", code):
                es_imports.add(sym)
    if not es_imports:
        es_imports = {"smooth"}

    require_line = f"const {{ {', '.join(sorted(es_imports))} }} = require('{_WASM_STD_REQUIRE}');\n"

    code = code.lstrip("\n")

    if has_own_data:
        return require_line + "\n" + code

    # Check if snippet uses x, y
    uses_x_y = bool(re.search(r"\bx\b|\by\b", code))
    if uses_x_y:
        data = (
            "const n = 100;\n"
            "const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));\n"
            "const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);\n"
        )
        return require_line + "\n" + data + "\n" + code
    else:
        return require_line + "\n" + code


def transform_rust(code: str, doc_rel: str) -> str | None:
    """Wrap a Rust fragment in fn main."""
    if "fn main" in code:
        return None

    lines = code.splitlines()

    # Separate use/extern lines from body lines
    use_lines: list[str] = []
    body_lines: list[str] = []
    for line in lines:
        s = line.strip()
        if s.startswith("use ") or s.startswith("extern crate"):
            use_lines.append(line.rstrip())
        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip()

    # Build use section at top (avoiding duplicates)
    has_prelude = any("prelude" in ln for ln in use_lines)
    has_tau = any(("TAU" in ln or "consts" in ln) for ln in use_lines)

    top_uses: list[str] = []
    if not has_prelude:
        top_uses.append("use fastLowess::prelude::*;")
    top_uses.extend(use_lines)
    uses_x_y = bool(re.search(r"\bx\b|\by\b", body))
    needs_tau = bool(re.search(r"\bTAU\b", body)) or uses_x_y
    if not has_tau and needs_tau:
        top_uses.append("use std::f64::consts::TAU;")

    result_parts = ["\n".join(top_uses), ""]
    result_parts.append("fn main() -> Result<(), LowessError> {")

    if uses_x_y:
        result_parts.append("    let n = 100usize;")
        result_parts.append(
            "    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();"
        )
        result_parts.append(
            "    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();"
        )
        result_parts.append("")

    # Indent body lines by 4 spaces
    for line in body_lines:
        if line.strip():
            result_parts.append("    " + line)
        else:
            result_parts.append("")

    result_parts.append("")
    result_parts.append("    Ok(())")
    result_parts.append("}")
    result_parts.append("")

    return "\n".join(result_parts)


def transform_cpp(code: str, doc_rel: str) -> str | None:
    """Wrap a C++ fragment in int main."""
    if "int main" in code:
        return None

    lines = code.splitlines()

    # Separate #include lines from body lines
    include_lines: list[str] = []
    body_lines: list[str] = []
    for line in lines:
        s = line.strip()
        if s.startswith("#include"):
            include_lines.append(line.rstrip())
        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip()

    # Build includes (always include the standard set, avoid duplicates)
    needed = [
        "#include <fastlowess.hpp>",
        "#include <cmath>",
        "#include <iostream>",
        "#include <vector>",
    ]
    existing_hdrs = set()
    for inc in include_lines:
        m = re.search(r"[<\"]([^>\"]+)[>\"]", inc)
        if m:
            existing_hdrs.add(m.group(1))
    final_includes = list(needed)
    for inc in include_lines:
        m = re.search(r"[<\"]([^>\"]+)[>\"]", inc)
        nm = {re.search(r"[<\"]([^>\"]+)[>\"]", n) for n in needed}
        existing_needed = {x.group(1) for x in nm if x}
        if m and m.group(1) not in existing_needed:
            final_includes.append(inc)

    uses_x_y = bool(
        re.search(r"\bx\[|\by\[|\bx,\s*y|\(x,\s*y\)|\(x\.data|\by\.data", body)
    )

    result_parts = final_includes + ["", "int main() {"]

    if uses_x_y:
        result_parts += [
            "    const int n = 100;",
            "    std::vector<double> x(n), y(n);",
            "    for (int i = 0; i < n; ++i) {",
            "        x[i] = i * 2 * M_PI / (n - 1);",
            "        y[i] = std::sin(x[i]) + 0.1;",
            "    }",
            "",
        ]

    for line in body_lines:
        if line.strip():
            result_parts.append("    " + line)
        else:
            result_parts.append("")

    result_parts += ["", "    return 0;", "}", ""]

    return "\n".join(result_parts)


# ---------------------------------------------------------------------------
# Doc file transformer
# ---------------------------------------------------------------------------

RUNNERS = ("python", "nodejs", "wasm", "rust", "cpp")

_FIXABLE_REASONS = {
    "fastlowess not imported (snippet is not self-contained)",
    "no require() \u2014 snippet must load fastlowess itself",
    "no require() \u2014 snippet must load the WASM package itself",
    "ES module import (not supported in CJS runner)",
    "fragment \u2014 no fn main (not a standalone Rust program)",
    "fragment \u2014 no int main (not a standalone C++ program)",
}


def _fix_rust_tau(code: str) -> str | None:
    """Add missing TAU import to already-complete Rust snippets that use it."""
    if "TAU" not in code:
        return None
    if "use std::f64::consts::TAU" in code:
        return None
    lines = code.splitlines()
    last_use = max(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("use ")),
        default=-1,
    )
    if last_use < 0:
        return None
    new_lines = (
        lines[: last_use + 1] + ["use std::f64::consts::TAU;"] + lines[last_use + 1 :]
    )
    return "\n".join(new_lines)


def _transform(code: str, runner: str, doc_rel: str) -> str | None:
    if runner == "python":
        return transform_python(code, doc_rel)
    if runner == "r":
        return transform_r(code, doc_rel)
    if runner == "julia":
        return transform_julia(code, doc_rel)
    if runner == "nodejs":
        return transform_nodejs(code, doc_rel)
    if runner == "wasm":
        return transform_wasm(code, doc_rel)
    if runner == "rust":
        result = transform_rust(code, doc_rel)
        if result is None:
            result = _fix_rust_tau(code)
        return result
    if runner == "cpp":
        return transform_cpp(code, doc_rel)
    return None


def fix_file(md: Path, dry_run: bool = False, lang: str | None = None) -> int:
    """Fix one Markdown file. Returns number of snippets transformed."""
    rel = str(md.relative_to(DOCS_DIR))

    snippets_to_fix: list[tuple[Snippet, str, str]] = []
    for s in extract_snippets(md):
        r = s.runner
        if r is None or r == "bash":
            continue
        if lang and r != lang:
            continue
        reason = should_skip(s, r)
        if reason and reason in _FIXABLE_REASONS:
            snippets_to_fix.append((s, r, reason))
        elif r == "nodejs" and not reason and "require(" in s.code:
            # Already runnable but might need data or fl. fix
            if _js_add_missing_data(s.code, rel) is not None:
                snippets_to_fix.append((s, r, "nodejs-needs-data-or-fl-fix"))
        elif r == "r" and not reason:
            # R: not skipped, but might need library/data
            transformed = transform_r(s.code, rel)
            if transformed is not None and transformed != s.code:
                snippets_to_fix.append((s, r, "r-needs-data"))
        elif r == "julia" and not reason:
            # Julia: not skipped, but might need using/data
            transformed = transform_julia(s.code, rel)
            if transformed is not None and transformed != s.code:
                snippets_to_fix.append((s, r, "julia-needs-data"))
        elif r == "rust" and not reason:
            # Rust: already runnable but might be missing TAU import
            if _fix_rust_tau(s.code) is not None:
                snippets_to_fix.append((s, r, "rust-needs-tau"))

    if not snippets_to_fix:
        return 0

    # Read file lines
    raw = md.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)

    # Process in REVERSE order so that earlier line-numbers stay valid
    changed = 0
    for s, runner, reason in sorted(snippets_to_fix, key=lambda t: -t[0].line):
        transformed = _transform(s.code, runner, rel)
        if transformed is None or transformed == s.code:
            continue

        # s.line is the 1-based line number of the opening fence (```python etc.).
        # In MkDocs tab-indented blocks the structure is:
        #   [indent]```python   <- fence_open_idx = s.line - 1 (0-based)
        #   [indent]code line 1 <- code_start_idx = s.line     (0-based)
        #   [indent]...
        #   [indent]```         <- code_end_idx   (0-based, NOT included in replacement)

        fence_open_idx = s.line - 1  # 0-based index of opening fence line
        code_start_idx = s.line  # 0-based index of first code line

        # Detect indentation from the opening fence (e.g. "    ```python")
        fence_line = lines[fence_open_idx]
        indent = len(fence_line) - len(fence_line.lstrip())
        indent_str = " " * indent

        # Find the closing fence: strip() == "```"  (handles any indentation level)
        code_end_idx = code_start_idx
        while code_end_idx < len(lines) and lines[code_end_idx].strip() != "```":
            code_end_idx += 1

        if code_end_idx >= len(lines):
            # Closing fence not found -- skip this snippet safely
            continue

        # Build replacement lines, re-applying the detected indentation
        new_code_lines: list[str] = []
        for line in transformed.rstrip("\n").splitlines():
            if line.strip():
                new_code_lines.append(indent_str + line + "\n")
            else:
                new_code_lines.append("\n")

        lines[code_start_idx:code_end_idx] = new_code_lines

        changed += 1
        if dry_run:
            print(f"  [DRY] {rel}:{s.line} [{runner}] {reason[:50]}")
        else:
            print(f"  FIX  {rel}:{s.line} [{runner}]")

    if not dry_run and changed:
        md.write_text("".join(lines), encoding="utf-8")

    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Don't write files")
    ap.add_argument("--lang", help="Only fix snippets for this language")
    args = ap.parse_args()

    total = 0
    for md in sorted(DOCS_DIR.rglob("*.md")):
        rel = str(md.relative_to(DOCS_DIR))
        if "installation" in rel:
            continue
        n = fix_file(md, dry_run=args.dry_run, lang=args.lang)
        if n:
            print(f"  => {n} snippet(s) in {rel}")
        total += n

    print(f"\nTotal: {total} snippet(s) {'would be ' if args.dry_run else ''}fixed.")


if __name__ == "__main__":
    main()
