"""Shared types, constants, and utilities for snippet runners."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = REPO_ROOT / "bindings" / "python" / "docs"
JULIA_DOCS_DIR = REPO_ROOT / "bindings" / "julia" / "julia" / "docs" / "src"
VIGNETTES_DIRS: list[Path] = [REPO_ROOT / "bindings" / "r" / "vignettes"]
RUST_CRATE_DOCS_DIRS: list[Path] = [
    REPO_ROOT / "crates" / "fastLowess" / "docs",
    REPO_ROOT / "crates" / "lowess" / "docs",
]
NODEJS_BINDING_DOCS_DIR = REPO_ROOT / "bindings" / "nodejs" / "src" / "content" / "docs"
WASM_BINDING_DOCS_DIR = REPO_ROOT / "bindings" / "wasm" / "src" / "content" / "docs"
CPP_BINDING_DOCS_DIR = REPO_ROOT / "bindings" / "cpp" / "docs"
GO_BINDING_DOCS_DIR = REPO_ROOT / "bindings" / "go" / "docs"
JAVA_BINDING_DOCS_DIR = REPO_ROOT / "bindings" / "java" / "docs"

_TAB_ALIASES: dict[str, set[str]] = {
    "python": {"Python"},
    "julia": {"Julia"},
    "nodejs": {"Node.js"},
    "wasm": {"WebAssembly"},
    "r": {"R"},
    "cpp": {"C++"},
    "go": {"Go"},
    "java": {"Java"},
    "rust": {
        "Rust",
        "Rust (fastLowess)",
        "lowess (no_std compatible)",
        "fastLowess (parallel)",
    },
}

_LANG_TAGS: dict[str, set[str]] = {
    "python": {"python"},
    "julia": {"julia"},
    "nodejs": {"javascript", "js"},
    "wasm": {"javascript", "js"},
    "r": {"r"},
    "cpp": {"cpp", "c++"},
    "go": {"go"},
    "java": {"java"},
    "rust": {"rust"},
}


@dataclass
class Snippet:
    file: Path
    line: int  # 1-based line number of the opening fence
    lang_tag: str  # code-block language tag (e.g. "python")
    tab: str | None  # nearest === "Tab" label, or None
    code: str

    @property
    def runner(self) -> str | None:
        """Return which runner handles this snippet, or None to skip."""
        for runner, tags in _LANG_TAGS.items():
            if self.lang_tag.lower() in tags:
                if runner in ("nodejs", "wasm"):
                    if self.tab in _TAB_ALIASES["wasm"]:
                        return "wasm"
                    if self.tab in _TAB_ALIASES["nodejs"]:
                        return "nodejs"
                    if (
                        "fastlowess-wasm" not in self.code
                        and "fastlowess_wasm" not in self.code
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


def _find_exe(name: str) -> str | None:
    return shutil.which(name)
