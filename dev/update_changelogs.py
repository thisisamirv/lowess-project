#!/usr/bin/env python3
"""Regenerate per-binding/crate changelog files from the root CHANGELOG.md.

The root CHANGELOG.md documents every language/binding together, grouped as:

    ## <version>
    ### <Added|Changed|Fixed|Removed>
    **<Language>:**
    - <bullet>

For each target below, this script extracts only the bullets listed under its
matching "**Label:**" heading(s) for every version, and writes the result to
the target's output file using a CRAN-NEWS-style format:

    # <package> <version>
    ## <Added|Changed|Fixed|Removed>
    * <bullet>

Versions with no matching entries are skipped entirely.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"
DESCRIPTION_PATH = REPO_ROOT / "bindings" / "r" / "DESCRIPTION"
REPO_URL = "https://github.com/thisisamirv/lowess-project"

VERSION_RE = re.compile(r"^## (\[Unreleased\]|\S+)")
SECTION_RE = re.compile(r"^### (\w+)")
LANG_RE = re.compile(r"^\*\*([^*]+):\*\*$")
BULLET_RE = re.compile(r"^- (.*)$")

# Labels merged into every target in addition to its own labels below, since
# these apply to all bindings/crates rather than one specific language.
COMMON_LABELS = ["docs", "Monorepo"]

# Each target's `labels` are matched against the "**Label:**" headings in
# CHANGELOG.md (case-sensitive, exact match). `package` is the display name
# used in the "# <package> <version>" heading; use None to resolve it from
# bindings/r/DESCRIPTION's Package field instead.
TARGETS = [
    {"labels": ["R"], "package": None, "output": "bindings/r/NEWS.md"},
    {
        "labels": ["C++"],
        "package": "fastlowess (C++)",
        "output": "bindings/cpp/docs/NEWS.md",
        "doxygen_page": "\\page news News",
    },
    {
        "labels": ["Julia"],
        "package": "FastLOWESS.jl",
        "output": "bindings/julia/julia/docs/src/NEWS.md",
    },
    {
        "labels": ["Node.js"],
        "package": "fastlowess (Node.js)",
        "output": "bindings/nodejs/src/content/docs/NEWS.md",
        "frontmatter_title": "News",
    },
    {
        "labels": ["Python"],
        "package": "fastlowess (Python)",
        "output": "bindings/python/docs/NEWS.md",
    },
    {
        "labels": ["WASM", "WebAssembly"],
        "package": "fastlowess-wasm",
        "output": "bindings/wasm/src/content/docs/NEWS.md",
        "frontmatter_title": "News",
    },
    {
        "labels": ["fastLowess", "lowess and fastLowess", "Rust"],
        "package": "fastLowess",
        "output": "crates/fastLowess/docs/news.md",
    },
    {
        "labels": ["lowess", "lowess and fastLowess", "Rust"],
        "package": "lowess",
        "output": "crates/lowess/docs/news.md",
    },
]


def read_description_field(name: str) -> str | None:
    text = DESCRIPTION_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith(f"{name}:"):
            return line.split(":", 1)[1].strip()
    return None


def changelog_url() -> str:
    bug_reports = read_description_field("BugReports")
    if bug_reports and bug_reports.endswith("/issues"):
        return bug_reports[: -len("issues")] + "blob/main/CHANGELOG.md"
    return f"{REPO_URL}/blob/main/CHANGELOG.md"


def parse_entries(
    changelog_text: str, labels: list[str]
) -> list[tuple[str, dict[str, list[str]]]]:
    """Return an ordered list of (version, {section: [bullets]}) tuples for any of `labels`."""
    versions: list[tuple[str, dict[str, list[str]]]] = []
    current_section = None
    current_lang = None

    for line in changelog_text.splitlines():
        if m := VERSION_RE.match(line):
            version = m.group(1)
            if version == "[Unreleased]":
                version = "(development version)"
            versions.append((version, {}))
            current_section = None
            current_lang = None
            continue
        if m := SECTION_RE.match(line):
            current_section = m.group(1)
            current_lang = None
            continue
        if m := LANG_RE.match(line.strip()):
            current_lang = m.group(1).strip()
            continue
        if (
            current_lang in labels
            and current_section is not None
            and versions
            and (m := BULLET_RE.match(line))
        ):
            versions[-1][1].setdefault(current_section, []).append(m.group(1))

    return versions


def render_news(
    package: str,
    versions: list[tuple[str, dict[str, list[str]]]],
    frontmatter_title: str | None = None,
    doxygen_page: str | None = None,
) -> str:
    header = "<!-- markdownlint-disable MD024 MD025 -->"
    if doxygen_page:
        header = f"{doxygen_page}\n\n{header}"
    if frontmatter_title:
        header = f"---\ntitle: {frontmatter_title}\n---\n{header}"
    blocks = []
    for version, sections in versions:
        if not sections:
            continue
        lines = [f"# {package} {version}", ""]
        for section, bullets in sections.items():
            lines.append(f"## {section}")
            lines.append("")
            lines.extend(f"* {bullet}" for bullet in bullets)
            lines.append("")
        blocks.append("\n".join(lines).rstrip() + "\n")

    footer = f"For the full changelog, see:\n<{changelog_url()}>\n"
    body = "\n".join(blocks) + "\n" + footer if blocks else footer
    return header + "\n" + body


def main() -> None:
    changelog_text = CHANGELOG_PATH.read_text(encoding="utf-8")
    for target in TARGETS:
        package = target["package"] or read_description_field("Package") or "package"
        versions = parse_entries(changelog_text, target["labels"] + COMMON_LABELS)
        output_path = REPO_ROOT / target["output"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            render_news(
                package,
                versions,
                target.get("frontmatter_title"),
                target.get("doxygen_page"),
            ),
            encoding="utf-8",
        )
        print(
            f"Wrote {output_path.relative_to(REPO_ROOT)} from {len(versions)} changelog version(s)"
        )


if __name__ == "__main__":
    main()
