#!/usr/bin/env python3
"""Embed a binding's top-level README.md as its docs-site home page.

Auto-detects the docs-site flavor from what exists in the binding directory:
  - Hugo (bindings/go): writes docs/_index.md (the content root mounted by
    docs-site/hugo.toml, i.e. Hugo's "home" page kind) with generated
    frontmatter (title read from docs-site/hugo.toml).
  - Antora (bindings/java): converts the README to AsciiDoc and writes
    docs/modules/ROOT/pages/index.adoc (title read from docs/antora.yml).
  - Starlight (bindings/nodejs, bindings/wasm): rewrites the body of the
    already-existing src/content/docs/index.md, preserving its own
    hand-authored frontmatter (hero, etc.) above it.
  - Sphinx (bindings/python): rewrites docs/index.md with the README body,
    preserving the existing hidden `:::{toctree}` block at the end (Sphinx's
    site navigation is defined inline in the root doc, unlike the other
    flavors' separate nav files).

In all cases the README's redundant top-level `# LOWESS Project` heading is
stripped, since the page's own title (Hugo frontmatter / Antora doctitle /
Starlight hero) already covers it.

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


def _embed_sphinx(index_path: Path) -> None:
    existing = index_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    toctree_match = re.search(r":::\{toctree\}[\s\S]*?\n:::\n?", existing)
    if not toctree_match:
        raise ValueError(f"No toctree block found in {index_path}")
    toctree = toctree_match.group(0)

    # Unlike Hugo/Antora/Starlight, Sphinx has no separate title mechanism
    # (frontmatter/hero) that renders into the page body, so the README's
    # own H1 is kept as the page's visible title instead of being stripped.
    body = _read_readme()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(f"{body.rstrip()}\n\n{toctree}", encoding="utf-8")


# --- Markdown -> AsciiDoc conversion, for the Antora flavor ---------------

_ADOC_CODE_FENCE_RE = re.compile(r"^```(\S*)\s*$")
_ADOC_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_ADOC_TABLE_SEP_RE = re.compile(r"^\s*\|?(\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
_ADOC_BULLET_RE = re.compile(r"^(\s*)-\s+(.*)$")
_ADOC_NUM_RE = re.compile(r"^(\s*)\d+\.\s+(.*)$")
_ADOC_BLOCKQUOTE_RE = re.compile(r"^>\s?(.*)$")
_ADOC_IMAGE_LINE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
_ADOC_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_ADOC_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ADOC_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_ADOC_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_ADOC_DISPLAY_MATH_LINE_RE = re.compile(r"^\$\$\s*$")
_ADOC_DISPLAY_MATH_ONE_LINE_RE = re.compile(r"^\$\$(.+)\$\$\s*$")
_ADOC_INLINE_MATH_RE = re.compile(r"\$([^$\n]+)\$")
_ADOC_HTML_COMMENT_RE = re.compile(r"^<!--.*-->\s*$")
_ADOC_LANG_MAP = {"output": "console", "": None}


def _adoc_protect_code_spans(text: str) -> tuple[str, list[str]]:
    saved: list[str] = []

    def repl(m: re.Match) -> str:
        # Escape literal braces so Asciidoctor doesn't treat them as an
        # attribute reference (e.g. a code span mentioning `{toctree}`).
        saved.append(m.group(0).replace("{", "\\{").replace("}", "\\}"))
        return f"\x00{len(saved) - 1}\x00"

    return _ADOC_CODE_SPAN_RE.sub(repl, text), saved


def _adoc_restore_code_spans(text: str, saved: list[str]) -> str:
    for i, span in enumerate(saved):
        text = text.replace(f"\x00{i}\x00", span)
    return text


def _adoc_convert_links(text: str) -> str:
    def repl(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        if target.startswith(("http://", "https://", "mailto:")):
            return f"{target}[{label}]"
        path, _, anchor = target.partition("#")
        path = path.replace(".md", ".adoc")
        xref = path + (f"#{anchor}" if anchor else "")
        return f"xref:{xref}[{label}]"

    return _ADOC_LINK_RE.sub(repl, text)


def _adoc_strip_nested_heading(text: str) -> str:
    """Blockquotes can't contain AsciiDoc section headings; render as bold text."""
    m = _ADOC_HEADING_RE.match(text)
    if m is not None:
        return _adoc_inline_full(f"**{m.group(2)}**")
    return _adoc_inline_full(text)


def _adoc_inline_full(text: str) -> str:
    text, saved = _adoc_protect_code_spans(text)
    bold_saved: list[str] = []

    def bold_repl(m: re.Match) -> str:
        bold_saved.append(m.group(1))
        return f"\x01{len(bold_saved) - 1}\x01"

    text = _ADOC_BOLD_RE.sub(bold_repl, text)
    text = _ADOC_ITALIC_RE.sub(r"_\1_", text)
    for i, content in enumerate(bold_saved):
        text = text.replace(f"\x01{i}\x01", f"*{content}*")
    text = _adoc_convert_links(text)
    text = _ADOC_INLINE_MATH_RE.sub(r"stem:[\1]", text)
    text = _adoc_restore_code_spans(text, saved)
    return text


def _adoc_convert_table(lines: list[str], start: int) -> tuple[list[str], int]:
    def split_row(raw: str) -> list[str]:
        raw = raw.strip().strip("|")
        raw = raw.replace(r"\|", "\x02")
        return [c.strip().replace("\x02", r"\|") for c in raw.split("|")]

    header = split_row(lines[start])
    sep = split_row(lines[start + 1])
    aligns = []
    for c in sep:
        left, right = c.startswith(":"), c.endswith(":")
        if left and right:
            aligns.append("^")
        elif right:
            aligns.append(">")
        else:
            aligns.append("")
    cols = ",".join(f"{a}1" for a in aligns) or ",".join("1" for _ in header)
    out = [f'[cols="{cols}",options="header"]', "|==="]
    out.append("|" + " |".join(_adoc_inline_full(h) for h in header))
    out.append("")
    i = start + 2
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = split_row(lines[i])
        out.append("|" + " |".join(_adoc_inline_full(c) for c in row))
        i += 1
    out.append("|===")
    return out, i


def _adoc_convert_body(body: str) -> str:
    lines = body.split("\n")
    out: list[str] = []
    i = 0
    in_code = False
    while i < len(lines):
        line = lines[i]
        fence = _ADOC_CODE_FENCE_RE.match(line)
        if fence is not None:
            if not in_code:
                mapped = _ADOC_LANG_MAP.get(fence.group(1), fence.group(1))
                if mapped:
                    out.append(f"[source,{mapped}]")
                out.append("----")
                in_code = True
            else:
                out.append("----")
                in_code = False
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue
        one_line_math = _ADOC_DISPLAY_MATH_ONE_LINE_RE.match(line)
        if one_line_math is not None:
            out.extend(["[stem]", "++++", one_line_math.group(1), "++++"])
            i += 1
            continue
        if _ADOC_DISPLAY_MATH_LINE_RE.match(line):
            out.extend(["[stem]", "++++"])
            i += 1
            while i < len(lines) and not _ADOC_DISPLAY_MATH_LINE_RE.match(lines[i]):
                out.append(lines[i])
                i += 1
            out.append("++++")
            i += 1
            continue
        if _ADOC_HTML_COMMENT_RE.match(line):
            i += 1
            continue
        if line.lstrip().startswith("<") and not line.lstrip().startswith("<!--"):
            html_lines = []
            while i < len(lines) and lines[i].strip() != "":
                html_lines.append(lines[i])
                i += 1
            out.append("++++")
            out.extend(html_lines)
            out.append("++++")
            continue
        if line.strip() == "---":
            out.append("'''")
            i += 1
            continue
        m = _ADOC_HEADING_RE.match(line)
        if m is not None:
            out.append("=" * len(m.group(1)) + " " + _adoc_inline_full(m.group(2)))
            i += 1
            continue
        if (
            line.startswith("|")
            and i + 1 < len(lines)
            and _ADOC_TABLE_SEP_RE.match(lines[i + 1])
        ):
            table_out, i = _adoc_convert_table(lines, i)
            out.extend(table_out)
            continue
        im = _ADOC_IMAGE_LINE_RE.match(line)
        if im is not None:
            out.append(f"image::{Path(im.group(2)).name}[{im.group(1)}]")
            i += 1
            continue
        if line.startswith(">"):
            quote_lines = []
            while i < len(lines) and (
                lines[i].startswith(">") or lines[i].strip() == ""
            ):
                if lines[i].strip() == "":
                    if i + 1 < len(lines) and lines[i + 1].startswith(">"):
                        quote_lines.append("")
                        i += 1
                        continue
                    break
                qm = _ADOC_BLOCKQUOTE_RE.match(lines[i])
                quote_lines.append(qm.group(1) if qm else "")
                i += 1
            body_text = "\n".join(
                _adoc_strip_nested_heading(q) for q in quote_lines
            ).strip()
            out.extend(["[NOTE]", "====", body_text, "===="])
            continue
        bm = _ADOC_BULLET_RE.match(line)
        if bm is not None:
            out.append("* " + _adoc_inline_full(bm.group(2)))
            i += 1
            continue
        nm = _ADOC_NUM_RE.match(line)
        if nm is not None:
            out.append(". " + _adoc_inline_full(nm.group(2)))
            i += 1
            continue
        out.append(_adoc_inline_full(line))
        i += 1
    return "\n".join(out)


def _embed_antora(antora_yml: Path, index_path: Path) -> None:
    text = antora_yml.read_text(encoding="utf-8")
    m = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
    if not m:
        raise ValueError(f"No `title` found in {antora_yml}")
    title = m.group(1).strip()

    body = H1_RE.sub(lambda m: m.group(1) or "", _read_readme(), count=1)
    converted = _adoc_convert_body(body.lstrip("\n"))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(f"= {title}\n\n{converted}\n", encoding="utf-8")


def main() -> None:
    hugo_toml = BINDING_DIR / "docs-site" / "hugo.toml"
    antora_yml = BINDING_DIR / "docs" / "antora.yml"
    starlight_index = BINDING_DIR / "src" / "content" / "docs" / "index.md"
    sphinx_conf = BINDING_DIR / "docs" / "conf.py"

    if hugo_toml.exists():
        index_path = BINDING_DIR / "docs" / "_index.md"
        _embed_hugo(hugo_toml, index_path)
    elif antora_yml.exists():
        index_path = BINDING_DIR / "docs" / "modules" / "ROOT" / "pages" / "index.adoc"
        _embed_antora(antora_yml, index_path)
    elif starlight_index.exists():
        index_path = starlight_index
        _embed_starlight(index_path)
    elif sphinx_conf.exists():
        index_path = BINDING_DIR / "docs" / "index.md"
        _embed_sphinx(index_path)
    else:
        sys.exit(
            f"Don't know how to embed README for {BINDING_DIR}: none of "
            f"{hugo_toml}, {antora_yml}, {starlight_index}, {sphinx_conf} exist."
        )

    print(f"Embedded README.md into {index_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
