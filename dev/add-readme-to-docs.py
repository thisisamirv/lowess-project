#!/usr/bin/env python3
"""Embed a binding's top-level README.md as its docs-site home page.

Auto-detects the docs-site flavor from what exists in the binding directory:
  - Hugo (bindings/go, bindings/java): writes docs/_index.md (the content
    root mounted by docs-site/hugo.toml, i.e. Hugo's "home" page kind) with
    generated frontmatter (title read from docs-site/hugo.toml).
  - Starlight (bindings/nodejs, bindings/wasm): rewrites the body of the
    already-existing src/content/docs/index.md, preserving its own
    hand-authored frontmatter (hero, etc.) above it.

In both cases the README's redundant top-level `# LOWESS Project` heading is
stripped, since the page's own title (Hugo frontmatter / Starlight hero)
already covers it.

Usage:
    python dev/add-readme-to-docs.py <binding_dir>
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

# Drops the leading "# LOWESS Project" (optionally preceded by an HTML
# comment, e.g. a markdownlint-disable directive) so the README's own H1
# isn't duplicated below the page's title.
H1_RE = re.compile(r"^(<!--[^\n]*-->\n)?# .+\n\n?")


def _read_readme() -> str:
    return README_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")


def _embed_hugo(hugo_toml: Path, index_path: Path) -> None:
    text = hugo_toml.read_text(encoding="utf-8")
    m = re.search(r'^title\s*=\s*"([^"]*)"', text, re.MULTILINE)
    if not m:
        raise ValueError(f"No `title` found in {hugo_toml}")
    title = m.group(1)

    body = H1_RE.sub(lambda m: m.group(1) or "", _read_readme(), count=1)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(f'---\ntitle: "{title}"\n---\n\n{body}', encoding="utf-8")


def _embed_starlight(index_path: Path) -> None:
    existing = index_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    fm_match = re.match(r"^---\n[\s\S]*?\n---\n", existing)
    if not fm_match:
        raise ValueError(f"No frontmatter block found in {index_path}")
    frontmatter = fm_match.group(0)

    body = H1_RE.sub(lambda m: (m.group(1) or "") + "\n", _read_readme(), count=1)
    index_path.write_text(f"{frontmatter}\n{body}", encoding="utf-8")


def main() -> None:
    hugo_toml = BINDING_DIR / "docs-site" / "hugo.toml"
    starlight_index = BINDING_DIR / "src" / "content" / "docs" / "index.md"

    if hugo_toml.exists():
        index_path = BINDING_DIR / "docs" / "_index.md"
        _embed_hugo(hugo_toml, index_path)
    elif starlight_index.exists():
        index_path = starlight_index
        _embed_starlight(index_path)
    else:
        sys.exit(
            f"Don't know how to embed README for {BINDING_DIR}: neither "
            f"{hugo_toml} nor {starlight_index} exists."
        )

    print(f"Embedded README.md into {index_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
