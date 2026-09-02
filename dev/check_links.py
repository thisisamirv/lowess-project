#!/usr/bin/env python3
"""Validate relative cross-reference links across every binding/crate's docs.

Checks Markdown `[text](target)` links and AsciiDoc `xref:target[]` references,
resolving them relative to the linking file and failing if the target doesn't
exist. Image references (`![alt](src)`, `image::name[]`) are intentionally
skipped: Doxygen (C++), Hugo (Go), and Antora (Java) each resolve images via
their own asset-path mechanism (`IMAGE_PATH`, static mounts, basename search)
rather than literal relative paths, so validating them here would be noisy
and incorrect. External URLs, anchors-only links, rustdoc intra-doc paths
(`crate::doc::api`), and Documenter.jl `@ref`/`@id` targets are also skipped,
since none of those are filesystem paths.

Usage:
    python dev/check_links.py                # check every binding/crate
    python dev/check_links.py --lang cpp      # a single binding/crate
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from runners.base import (
    CPP_BINDING_DOCS_DIR,
    DOCS_DIR,
    GO_BINDING_DOCS_DIR,
    JAVA_BINDING_DOCS_DIR,
    JULIA_DOCS_DIR,
    NODEJS_BINDING_DOCS_DIR,
    REPO_ROOT,
    RUST_CRATE_DOCS_DIRS,
    VIGNETTES_DIRS,
    WASM_BINDING_DOCS_DIR,
)

TARGETS: dict[str, list[Path]] = {
    "python": [DOCS_DIR],
    "julia": [JULIA_DOCS_DIR],
    "r": VIGNETTES_DIRS,
    "nodejs": [NODEJS_BINDING_DOCS_DIR],
    "wasm": [WASM_BINDING_DOCS_DIR],
    "cpp": [CPP_BINDING_DOCS_DIR],
    "go": [GO_BINDING_DOCS_DIR],
    "java": [JAVA_BINDING_DOCS_DIR],
    "rust": RUST_CRATE_DOCS_DIRS,
}

_MD_LINK_RE = re.compile(r"(!?)\[[^\]]*\]\(([^)]+)\)")
_XREF_RE = re.compile(r"xref:([^\[]+)\[")


def _is_skippable(target: str) -> bool:
    t = target.strip()
    return (
        not t
        or t.startswith(("http://", "https://", "mailto:", "#", "@"))
        or "::" in t  # rustdoc intra-doc link, e.g. crate::doc::api
    )


def _check_markdown_file(f: Path) -> list[str]:
    if f.name in ("NEWS.md", "news.md"):
        return []  # auto-generated changelog prose, not real navigable links
    errors = []
    text = f.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.split("\n"), 1):
        for m in _MD_LINK_RE.finditer(line):
            is_image, target = m.group(1), m.group(2)
            if is_image or _is_skippable(target):
                continue
            t = target.split("#")[0].split("?")[0].strip()
            if not t or t.startswith("/"):
                continue
            resolved = (f.parent / t).resolve()
            if not (resolved.exists() or Path(f"{resolved}.md").exists()):
                errors.append(
                    f"{f.relative_to(REPO_ROOT)}:{lineno}: broken link -> {target}"
                )
    return errors


def _check_adoc_file(f: Path) -> list[str]:
    errors = []
    # Antora resolves nav.adoc xrefs relative to the module's pages/ directory,
    # not nav.adoc's own physical location (which sits one level above it).
    base_dir = f.parent / "pages" if f.name == "nav.adoc" else f.parent
    text = f.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.split("\n"), 1):
        for m in _XREF_RE.finditer(line):
            target = m.group(1)
            if target.startswith("http"):
                continue
            t = target.split("#")[0]
            if not t or ":" in t:  # component:module:page - cross-component, skip
                continue
            t = t.removeprefix("./")
            resolved = (base_dir / t).resolve()
            if not resolved.exists():
                errors.append(
                    f"{f.relative_to(REPO_ROOT)}:{lineno}: broken xref -> {target}"
                )
    return errors


def check_lang(lang: str) -> list[str]:
    errors: list[str] = []
    for root in TARGETS[lang]:
        if not root.exists():
            continue
        for ext in ("*.md", "*.mdx", "*.Rmd"):
            for f in sorted(root.rglob(ext)):
                errors.extend(_check_markdown_file(f))
        for f in sorted(root.rglob("*.adoc")):
            errors.extend(_check_adoc_file(f))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lang", choices=sorted(TARGETS), help="Check a single binding/crate"
    )
    args = parser.parse_args()

    langs = [args.lang] if args.lang else sorted(TARGETS)
    all_errors: list[str] = []
    for lang in langs:
        all_errors.extend(check_lang(lang))

    if all_errors:
        print(f"Found {len(all_errors)} broken cross-reference link(s):\n")
        for e in all_errors:
            print(f"  {e}")
        sys.exit(1)

    print(f"No broken cross-reference links found ({len(langs)} target(s) checked).")


if __name__ == "__main__":
    main()
