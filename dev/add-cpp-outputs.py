#!/usr/bin/env python3
"""Compile and run each ```cpp snippet in bindings/cpp/docs/, inject ```output blocks."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Allow running from repo root or directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dev"))

from runners.base import Snippet
from runners.cpp import run_cpp

CPP_DIR = (
    Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO_ROOT / "bindings" / "cpp"
)
DOCS_DIR = CPP_DIR / "docs"
TIMEOUT = 30

FENCE_RE = re.compile(r"```cpp\n([\s\S]*?)```", re.MULTILINE)
OUTPUT_RE = re.compile(r"\n\n```output\n[\s\S]*?```")


def process_file(md_path: Path, failures: list[str]) -> bool:
    original = md_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    result = ""
    pos = 0

    for m in FENCE_RE.finditer(original):
        code = m.group(1)
        # Only run complete programs
        if "int main(" not in code:
            result += original[pos : m.end()]
            pos = m.end()
            continue

        result += original[pos : m.end()]
        pos = m.end()

        # Consume any existing output block immediately following
        tail = original[pos:]
        existing = OUTPUT_RE.match(tail)
        if existing:
            pos += len(existing.group())

        snippet = Snippet(file=md_path, line=0, lang_tag="cpp", tab=None, code=code)
        run_result = run_cpp(snippet, timeout=TIMEOUT)

        if run_result.skipped:
            if existing:
                result += existing.group()
        elif run_result.passed:
            if run_result.stdout:
                result += f"\n\n```output\n{run_result.stdout.rstrip()}\n```"
        else:
            err = (run_result.stderr or "").strip()
            failures.append(f"  FAIL {md_path.name}\n{err}")
            if existing:
                result += existing.group()

    result += original[pos:]

    if result != original:
        md_path.write_text(result, encoding="utf-8")
        return True
    return False


files = sorted(f for f in DOCS_DIR.glob("*.md") if f.name != "index.md")
failures: list[str] = []
updated = 0
for f in files:
    sys.stdout.write(f"  {f.name}...")
    changed = process_file(f, failures)
    sys.stdout.write(" updated\n" if changed else "\n")
    if changed:
        updated += 1

print(f"add-cpp-outputs: {updated}/{len(files)} file(s) updated")

if failures:
    for msg in failures:
        print(msg, file=sys.stderr)
    sys.exit(1)
