#!/usr/bin/env python3
"""Embed a binding's top-level README.md as its Hugo docs-site home page.

Writes {binding_dir}/docs/_index.md (the content root mounted by
docs-site/hugo.toml, i.e. Hugo's "home" page kind) from README.md, stripping
its redundant top-level `# LOWESS Project` heading since the page's own
frontmatter `title` already covers it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BINDING_DIR = (
    Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO_ROOT / "bindings" / "go"
)
README_PATH = BINDING_DIR / "README.md"
HUGO_CONFIG_PATH = BINDING_DIR / "docs-site" / "hugo.toml"
INDEX_PATH = BINDING_DIR / "docs" / "_index.md"

# Drops the leading "# LOWESS Project" (optionally preceded by an HTML
# comment, e.g. a markdownlint-disable directive) so the README's own H1
# isn't duplicated below the page's frontmatter `title`.
H1_RE = re.compile(r"^(<!--[^\n]*-->\n)?# .+\n\n?")


def read_site_title() -> str:
    text = HUGO_CONFIG_PATH.read_text(encoding="utf-8")
    m = re.search(r'^title\s*=\s*"([^"]*)"', text, re.MULTILINE)
    if not m:
        raise ValueError(f"No `title` found in {HUGO_CONFIG_PATH}")
    return m.group(1)


def main() -> None:
    title = read_site_title()
    readme = README_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    body = H1_RE.sub(lambda m: m.group(1) or "", readme, count=1)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(f'---\ntitle: "{title}"\n---\n\n{body}', encoding="utf-8")
    print(f"Embedded README.md into {INDEX_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
