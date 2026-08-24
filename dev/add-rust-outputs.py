#!/usr/bin/env python3
"""Compile and run each ```rust snippet in crates/fastLowess/docs/ (or fastLoess),
inject ```output blocks -- mirrors dev/add-cpp-outputs.py for Rust.

Usage:
    python dev/add-rust-outputs.py [DOCS_DIR]

DOCS_DIR defaults to crates/fastLowess/docs.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else REPO_ROOT / "crates" / "fastLowess" / "docs"
)
CRATE_DIR = DOCS_DIR.parent  # fastLowess or fastLoess crate
TIMEOUT = 30

FENCE_RE = re.compile(r"```rust\n([\s\S]*?)```", re.MULTILINE)
OUTPUT_RE = re.compile(r"\n\n```output\n[\s\S]*?```")

CARGO_TOML = """\
[package]
name = "snippet-runner"
version = "0.1.0"
edition = "2024"

[dependencies]
{crate} = {{ path = "{path}" }}
{extra_deps}"""


def preprocess(code: str) -> str:
    """Strip Rust doctest hidden-line prefixes so code compiles as a plain file.

    Lines starting with '# ' are hidden in rendered docs but included in
    compilation — the prefix is doctest syntax, not valid plain Rust.
    """
    out = []
    for line in code.splitlines():
        if line.startswith("# "):
            out.append(line[2:])
        elif line == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def build_and_run_all(
    snippets: list[tuple[Path, int, str]],
) -> dict[tuple[Path, int], str | None]:
    """Compile ALL snippets sharing one Cargo project.

    fastLowess is compiled once (first bin build); subsequent bins reuse the
    cached rlib so each only needs to compile its small main file.
    Per-snippet failures are isolated — a broken snippet does not block others.
    """
    crate_name = CRATE_DIR.name
    results: dict[tuple[Path, int], str | None] = {}

    with tempfile.TemporaryDirectory(prefix="rust-snippets-") as tmp:
        tmp_path = Path(tmp)
        bin_dir = tmp_path / "src" / "bin"
        bin_dir.mkdir(parents=True)

        for i, (_md, _start, code) in enumerate(snippets):
            (bin_dir / f"s{i}.rs").write_text(preprocess(code), encoding="utf-8")

        # Include all sibling crates so snippets can reference any crate in the workspace
        extra_deps = ""
        for sibling in sorted(CRATE_DIR.parent.iterdir()):
            cargo_toml = sibling / "Cargo.toml"
            if sibling == CRATE_DIR or not cargo_toml.exists():
                continue
            name_m = re.search(
                r'^name\s*=\s*"([^"]+)"',
                cargo_toml.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
            if name_m:
                sib_name = name_m.group(1)
                sib_path = str(sibling).replace("\\", "/")
                extra_deps += f'{sib_name} = {{ path = "{sib_path}" }}\n'

        (tmp_path / "Cargo.toml").write_text(
            CARGO_TOML.format(
                crate=crate_name,
                path=str(CRATE_DIR).replace("\\", "/"),
                extra_deps=extra_deps,
            ),
            encoding="utf-8",
        )

        env = {**os.environ, "CARGO_TERM_COLOR": "never"}
        suffix = ".exe" if sys.platform == "win32" else ""

        for i, (md, start, _code) in enumerate(snippets):
            # Each call reuses the target dir — fastLowess is only compiled once
            build = subprocess.run(
                ["cargo", "build", "--bin", f"s{i}", "--quiet"],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if build.returncode != 0:
                # Show first 800 chars of compiler output to aid debugging
                print(
                    f"  snippet {i} ({md.name}) build error — skipping\n"
                    f"{build.stderr[:800]}",
                    file=sys.stderr,
                )
                results[(md, start)] = None
                continue

            binary = tmp_path / "target" / "debug" / f"s{i}{suffix}"
            if not binary.exists():
                results[(md, start)] = None
                continue

            proc = subprocess.run(
                [str(binary)],
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                check=False,
            )
            if proc.returncode != 0:
                print(
                    f"  snippet {i} ({md.name}) runtime error — skipping\n"
                    f"{proc.stderr[:600]}",
                    file=sys.stderr,
                )
            results[(md, start)] = proc.stdout if proc.returncode == 0 else None

    return results


def collect_snippets(md_path: Path) -> list[tuple[int, int, str]]:
    """Return (start, end, code) for every ```rust fn main( block."""
    text = md_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return [
        (m.start(), m.end(), m.group(1))
        for m in FENCE_RE.finditer(text)
        if "fn main(" in m.group(1)
        # Skip GPU-only snippets — Backend::GPU requires the `gpu` feature and produces no stdout
        and "Backend::GPU" not in m.group(1)
    ]


def process_file(md_path: Path, outputs: dict[tuple[Path, int], str | None]) -> bool:
    original = md_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    result, pos, changed = "", 0, False
    for m in FENCE_RE.finditer(original):
        if "fn main(" not in m.group(1):
            result += original[pos : m.end()]
            pos = m.end()
            continue
        result += original[pos : m.end()]
        pos = m.end()
        stale = OUTPUT_RE.match(original, pos)
        if stale:
            pos += len(stale.group())
        stdout = outputs.get((md_path, m.start()))
        if stdout and stdout.strip():
            result += f"\n\n```output\n{stdout.rstrip()}\n```"
            changed = True
        elif stale:
            result += stale.group()
    result += original[pos:]
    if result != original:
        md_path.write_text(result, encoding="utf-8")
        return True
    return changed


def main() -> None:
    if not DOCS_DIR.exists():
        sys.exit(f"Docs directory not found: {DOCS_DIR}")

    md_files = sorted(DOCS_DIR.glob("*.md"))
    tasks: list[tuple[Path, int, str]] = [
        (md, start, code)
        for md in md_files
        for start, _end, code in collect_snippets(md)
    ]

    if not tasks:
        print("No executable snippets found.")
        return

    print(f"Building {len(tasks)} snippet(s) in one cargo invocation...")
    outputs = build_and_run_all(tasks)

    updated = [md.name for md in md_files if process_file(md, outputs)]
    for name in updated:
        print(f"  updated: {name}")
    print(f"\nDone -- {len(updated)} file(s) updated.")


if __name__ == "__main__":
    main()
