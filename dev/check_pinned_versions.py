#!/usr/bin/env python3
"""Check hardcoded tool/library version pins against their latest GitHub release.

Unlike Renovate/Dependabot, this does NOT open PRs or modify any files -- it only
reports which pins are outdated and exits non-zero so CI can fail the build.
Covers version pins that live outside any package manifest Dependabot understands
(CMake FetchContent, install scripts, `go install`, vendored static files).

Usage:
    python dev/check_pinned_versions.py

Requires network access to api.github.com. Set GITHUB_TOKEN to avoid the
unauthenticated rate limit (60 requests/hour vs 5000/hour with a token).
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PINS = [
    {
        "name": "Corrosion (CMake FetchContent)",
        "file": REPO_ROOT / "bindings/cpp/CMakeLists.txt",
        "pattern": re.compile(r"GIT_TAG\s+v(?P<version>\d+\.\d+\.\d+)"),
        "repo": "corrosion-rs/corrosion",
    },
    {
        "name": "doxygen-awesome-css (vendored theme)",
        "file": REPO_ROOT / "bindings/cpp/Doxyfile",
        "pattern": re.compile(
            r"# renovate: doxygen-awesome-css version: v(?P<version>\d+\.\d+\.\d+)"
        ),
        "repo": "jothepro/doxygen-awesome-css",
    },
    {
        "name": "Checkstyle (standalone jar)",
        "file": REPO_ROOT / "bindings/java/Makefile",
        "pattern": re.compile(r"CHECKSTYLE_VERSION\s*:=\s*(?P<version>\d+\.\d+\.\d+)"),
        "repo": "checkstyle/checkstyle",
        "tag_prefix": "checkstyle-",
    },
    {
        "name": "golangci-lint (install script)",
        "file": REPO_ROOT / "bindings/go/Makefile",
        "pattern": re.compile(
            r"GOLANGCI_LINT_VERSION\s*:=\s*v(?P<version>\d+\.\d+\.\d+)"
        ),
        "repo": "golangci/golangci-lint",
    },
    {
        "name": "Hugo (go install in docs.yml)",
        "file": REPO_ROOT / ".github/workflows/docs.yml",
        "pattern": re.compile(r"gohugoio/hugo@v(?P<version>\d+\.\d+\.\d+)"),
        "repo": "gohugoio/hugo",
    },
    {
        "name": "rextendr scaffold (DESCRIPTION)",
        "file": REPO_ROOT / "bindings/r/DESCRIPTION",
        "pattern": re.compile(r"Config/rextendr/version:\s*(?P<version>\d+\.\d+\.\d+)"),
        "repo": "extendr/rextendr",
    },
    {
        "name": "roxygen2 (DESCRIPTION)",
        "file": REPO_ROOT / "bindings/r/DESCRIPTION",
        "pattern": re.compile(r"Config/roxygen2/version:\s*(?P<version>\d+\.\d+\.\d+)"),
        "repo": "r-lib/roxygen2",
    },
    {
        "name": "KaTeX (lowess crate CDN header)",
        "file": REPO_ROOT / "crates/lowess/katex-header.html",
        "pattern": re.compile(r"katex@(?P<version>\d+\.\d+\.\d+)"),
        "repo": "KaTeX/KaTeX",
    },
    {
        "name": "KaTeX (fastLowess crate CDN header)",
        "file": REPO_ROOT / "crates/fastLowess/katex-header.html",
        "pattern": re.compile(r"katex@(?P<version>\d+\.\d+\.\d+)"),
        "repo": "KaTeX/KaTeX",
    },
]


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split("."))


def _latest_release(repo: str, tag_prefix: str = "") -> str:
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "lowess-project-version-check",
        },
    )
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)
    tag = data["tag_name"]
    if tag_prefix and tag.startswith(tag_prefix):
        tag = tag[len(tag_prefix) :]
    return tag.lstrip("v")


def main() -> None:
    outdated: list[str] = []
    errors: list[str] = []

    for pin in PINS:
        text = pin["file"].read_text(encoding="utf-8")
        m = pin["pattern"].search(text)
        if not m:
            errors.append(f"{pin['name']}: could not find version pin in {pin['file']}")
            continue
        current = m.group("version")
        try:
            latest = _latest_release(pin["repo"], pin.get("tag_prefix", ""))
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            errors.append(
                f"{pin['name']}: failed to fetch latest release for {pin['repo']}: {exc}"
            )
            continue

        if _version_tuple(latest) > _version_tuple(current):
            outdated.append(
                f"{pin['name']}: {current} -> {latest} "
                f"(https://github.com/{pin['repo']}/releases/tag/"
                f"{pin.get('tag_prefix', '')}{latest})"
            )
        else:
            print(f"OK: {pin['name']} is up to date ({current})")

    if outdated:
        print(f"\n{len(outdated)} outdated pin(s) found:\n")
        for line in outdated:
            print(f"  {line}")
    if errors:
        print(f"\n{len(errors)} check error(s):\n")
        for line in errors:
            print(f"  {line}")

    if outdated or errors:
        sys.exit(1)
    print("\nAll pinned versions are up to date.")


if __name__ == "__main__":
    main()
