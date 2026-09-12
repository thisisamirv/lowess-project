#!/usr/bin/env python3
"""Bump the project version across every crate, binding, and manifest in the repo.

Covers: Cargo.toml package versions (Rust crates + all bindings), the internal
fastLowess/lowess path-dependency version requirements (major.minor), each
binding's own version file (package.json, pyproject-adjacent __version__.py,
pom.xml, DESCRIPTION, Project.toml (incl. the fastlowess_jll compat floor),
version.go, CMakeLists.txt, FastLowess.java), CITATION.cff, and the Spack
recipe's example `url`. Also updates the Go module's `/vN` major-version
suffix (go.mod files, doc snippets, README/docs badges, the doc-snippet
runner) whenever a major version bump changes it -- see
https://go.dev/ref/mod#major-version-suffixes.

Does NOT touch: CHANGELOG.md (write that by hand), generated NEWS.md/docs-site
content (regenerated via `make <lang>-dev` / dev/update_changelogs.py), or the
Spack recipe's `version()`/`sha256` block and the conda-forge feedstock -- those
require a published release tarball to hash, so release-cpp.yml/release-conda.yml
update them after the fact, not before.

Usage:
    python dev/bump_version.py --version 3.3.0
    python dev/bump_version.py --version 3.3.0 --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

# Go requires any module tagged v2.0.0+ to end its module path with /vN, or
# the Go toolchain (and pkg.go.dev) silently ignores those tags and falls
# back to commit pseudo-versions. These are the files that reference the Go
# module's import path and must be kept in sync with its current major.
GO_MODULE_BASE_PATH = "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
GO_FASTLOWESS_GOMOD = "bindings/go/fastlowess/go.mod"
GO_TESTS_GOMOD = "bindings/go/tests/go.mod"
GO_TESTS_MAIN = "bindings/go/tests/lowess_test.go"
GO_DOC_RUNNER = "dev/runners/go.py"
GO_README = "bindings/go/README.md"
GO_DOCS_INDEX = "bindings/go/docs/_index.md"
ROOT_README = "README.md"

# Cargo.toml files whose bare `[package]` `version = "X.Y.Z"` line should track
# the release version 1:1.
CARGO_PACKAGE_FILES = [
    "crates/lowess/Cargo.toml",
    "crates/fastLowess/Cargo.toml",
    "bindings/cpp/Cargo.toml",
    "bindings/go/Cargo.toml",
    "bindings/java/Cargo.toml",
    "bindings/julia/Cargo.toml",
    "bindings/nodejs/Cargo.toml",
    "bindings/python/Cargo.toml",
    "bindings/r/src/Cargo.toml",
    "bindings/wasm/Cargo.toml",
]

# Cargo.toml files with a `fastLowess = { path = "../../crates/fastLowess",
# version = "X.Y", ... }` internal path-dependency requirement to keep in sync
# (major.minor only -- these are SemVer caret requirements, not exact pins).
FASTLOWESS_DEP_FILES = [
    "bindings/cpp/Cargo.toml",
    "bindings/go/Cargo.toml",
    "bindings/java/Cargo.toml",
    "bindings/julia/Cargo.toml",
    "bindings/nodejs/Cargo.toml",
    "bindings/python/Cargo.toml",
    "bindings/wasm/Cargo.toml",
]

# Node.js binding platform subpackages (each just has its own bare version).
NODEJS_NPM_PACKAGES = [
    "darwin-arm64",
    "darwin-x64",
    "linux-arm-gnueabihf",
    "linux-arm64-gnu",
    "linux-arm64-musl",
    "linux-x64-gnu",
    "linux-x64-musl",
    "win32-arm64-msvc",
    "win32-x64-msvc",
]


def _replace(
    path: Path,
    pattern: re.Pattern[str],
    replacement: str,
    dry_run: bool,
    count: int = 1,
) -> bool:
    """Apply one regex substitution to `path`, reporting what happened."""
    rel = path.relative_to(REPO_ROOT)
    if not path.exists():
        print(f"  SKIP (missing): {rel}")
        return False
    text = path.read_text(encoding="utf-8")
    new_text, n = pattern.subn(replacement, text, count=count)
    if n == 0:
        print(f"  WARNING: no match in {rel}")
        return False
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    verb = "Would update" if dry_run else "Updated"
    print(f"  {verb} ({n}x): {rel}")
    return True


def _replace_literal_all(
    path: Path, old: str, new: str, dry_run: bool, required: bool = True
) -> bool:
    """Replace every literal occurrence of `old` with `new` in `path`.

    When `required` is False, a file with no occurrences is not an error (used
    for glob-discovered doc pages that may not contain the pattern at all).
    """
    rel = path.relative_to(REPO_ROOT)
    if not path.exists():
        print(f"  SKIP (missing): {rel}")
        return False
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        if not required:
            return True
        print(f"  WARNING: no match in {rel}")
        return False
    if not dry_run:
        path.write_text(text.replace(old, new), encoding="utf-8")
    verb = "Would update" if dry_run else "Updated"
    print(f"  {verb} ({count}x): {rel}")
    return True


def _current_go_module_suffix() -> str:
    """Return the Go module's current major-version suffix, e.g. '/v4', or '' for v0/v1."""
    gomod = REPO_ROOT / GO_FASTLOWESS_GOMOD
    text = gomod.read_text(encoding="utf-8")
    m = re.search(
        rf"^module {re.escape(GO_MODULE_BASE_PATH)}(/v\d+)?$", text, re.MULTILINE
    )
    if m is None:
        print(f"  WARNING: could not parse module line in {GO_FASTLOWESS_GOMOD}")
        return ""
    return m.group(1) or ""


def apply_go_module_suffix(new_major: str, dry_run: bool) -> bool:
    """Update the Go module's `/vN` major-version suffix if this bump crosses
    a major version boundary (see https://go.dev/ref/mod#major-version-suffixes)."""
    old_suffix = _current_go_module_suffix()
    new_suffix = "" if new_major in ("0", "1") else f"/v{new_major}"
    if old_suffix == new_suffix:
        return True

    old_path = GO_MODULE_BASE_PATH + old_suffix
    new_path = GO_MODULE_BASE_PATH + new_suffix
    placeholder = f"v{new_major}.0.0" if new_suffix else "v0.0.0"
    print(
        f"Go module major version suffix change: "
        f"{old_suffix or '(none)'} -> {new_suffix or '(none)'}"
    )

    # Plain path-only files: one literal find/replace each.
    files = [GO_FASTLOWESS_GOMOD, GO_TESTS_MAIN, GO_README, GO_DOCS_INDEX, ROOT_README]
    all_ok = True
    for rel in files:
        ok = _replace_literal_all(REPO_ROOT / rel, old_path, new_path, dry_run)
        all_ok = all_ok and ok

    # Doc pages: not every page has a Go snippet, so a page without the
    # pattern is not an error. Skip GO_DOCS_INDEX (already handled above).
    doc_files = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in (REPO_ROOT / "bindings/go/docs").rglob("*.md")
        if str(p.relative_to(REPO_ROOT)).replace("\\", "/") != GO_DOCS_INDEX
    )
    for rel in doc_files:
        ok = _replace_literal_all(
            REPO_ROOT / rel, old_path, new_path, dry_run, required=False
        )
        all_ok = all_ok and ok

    # tests/go.mod's require line also carries a version number that must
    # match the new major; match against `old_path` (not `new_path`) so this
    # stays correct under --dry-run, where prior writes aren't persisted.
    all_ok = (
        _replace(
            REPO_ROOT / GO_TESTS_GOMOD,
            re.compile(
                r"^require " + re.escape(old_path) + r" v\d+\.\d+\.\d+$", re.MULTILINE
            ),
            f"require {new_path} {placeholder}",
            dry_run,
        )
        and all_ok
    )
    all_ok = (
        _replace(
            REPO_ROOT / GO_TESTS_GOMOD,
            re.compile(
                r"^replace " + re.escape(old_path) + r" => \.\./fastlowess$",
                re.MULTILINE,
            ),
            f"replace {new_path} => ../fastlowess",
            dry_run,
        )
        and all_ok
    )

    # dev/runners/go.py: MODULE_PATH constant, plus its hardcoded placeholder
    # version on the generated `require {MODULE_PATH} vX.0.0` scaffold line.
    all_ok = (
        _replace_literal_all(REPO_ROOT / GO_DOC_RUNNER, old_path, new_path, dry_run)
        and all_ok
    )
    all_ok = (
        _replace(
            REPO_ROOT / GO_DOC_RUNNER,
            re.compile(r"(require \{MODULE_PATH\} )v\d+\.\d+\.\d+"),
            rf"\g<1>{placeholder}",
            dry_run,
        )
        and all_ok
    )
    return all_ok


def build_targets(
    new_version: str, new_major_minor: str
) -> list[tuple[str, re.Pattern[str], str, int]]:
    """Return (relative_path, pattern, replacement, expected_count) tuples."""
    targets: list[tuple[str, re.Pattern[str], str, int]] = []

    cargo_pkg_pattern = re.compile(r'^version = "\d+\.\d+\.\d+"$', re.MULTILINE)
    for rel in CARGO_PACKAGE_FILES:
        targets.append((rel, cargo_pkg_pattern, f'version = "{new_version}"', 1))

    # CONTRIBUTING.md's "Individual crate Cargo.toml" example snippet.
    targets.append(
        (
            "CONTRIBUTING.md",
            re.compile(r'(name = "lowess"\nversion = )"\d+\.\d+\.\d+"'),
            rf'\g<1>"{new_version}"',
            1,
        )
    )

    targets.append(
        (
            "crates/fastLowess/Cargo.toml",
            re.compile(r'(lowess = \{ path = "\.\./lowess", version = ")\d+\.\d+(")'),
            rf"\g<1>{new_major_minor}\g<2>",
            1,
        )
    )
    fastlowess_dep_pattern = re.compile(
        r'(fastLowess = \{ path = "\.\./\.\./crates/fastLowess", version = ")\d+\.\d+(")'
    )
    for rel in FASTLOWESS_DEP_FILES:
        targets.append(
            (rel, fastlowess_dep_pattern, rf"\g<1>{new_major_minor}\g<2>", 1)
        )

    targets.append(
        (
            "bindings/python/python/fastlowess/__version__.py",
            re.compile(r'__version__ = "\d+\.\d+\.\d+"'),
            f'__version__ = "{new_version}"',
            1,
        )
    )

    package_json_version_pattern = re.compile(r'"version": "\d+\.\d+\.\d+"')
    targets.append(
        (
            "bindings/nodejs/package.json",
            package_json_version_pattern,
            f'"version": "{new_version}"',
            1,
        )
    )
    targets.append(
        (
            "bindings/nodejs/package.json",
            re.compile(r'("fastlowess-[\w-]+": "\^)\d+\.\d+\.\d+(")'),
            rf"\g<1>{new_version}\g<2>",
            0,
        )
    )
    for pkg in NODEJS_NPM_PACKAGES:
        targets.append(
            (
                f"bindings/nodejs/npm/{pkg}/package.json",
                package_json_version_pattern,
                f'"version": "{new_version}"',
                1,
            )
        )
    targets.append(
        (
            "bindings/wasm/package.json",
            package_json_version_pattern,
            f'"version": "{new_version}"',
            1,
        )
    )

    targets.append(
        (
            "bindings/java/pom.xml",
            re.compile(
                r"(<artifactId>fastlowess</artifactId>\s*\n\s*)<version>\d+\.\d+\.\d+</version>"
            ),
            rf"\g<1><version>{new_version}</version>",
            1,
        )
    )
    targets.append(
        (
            "bindings/java/src/main/java/fastlowess/FastLowess.java",
            re.compile(r'public static final String VERSION = "\d+\.\d+\.\d+";'),
            f'public static final String VERSION = "{new_version}";',
            1,
        )
    )

    # docs/installation.adoc's Maven dependency example snippet.
    targets.append(
        (
            "bindings/java/docs/modules/ROOT/pages/introduction/installation.adoc",
            re.compile(
                r"(<artifactId>fastlowess</artifactId>\s*\n\s*)<version>\d+\.\d+\.\d+</version>"
            ),
            rf"\g<1><version>{new_version}</version>",
            1,
        )
    )

    targets.append(
        (
            "bindings/r/DESCRIPTION",
            re.compile(r"^Version: \d+\.\d+\.\d+$", re.MULTILINE),
            f"Version: {new_version}",
            1,
        )
    )

    targets.append(
        (
            "bindings/julia/julia/Project.toml",
            re.compile(r'^version = "\d+\.\d+\.\d+"$', re.MULTILINE),
            f'version = "{new_version}"',
            1,
        )
    )
    # `make julia-dev`/CI relax this compat floor to an OR-list of the actual
    # latest-registered JLL version at test-time (see bindings/julia/Makefile),
    # so it's safe to bump this to the not-yet-registered version here.
    targets.append(
        (
            "bindings/julia/julia/Project.toml",
            re.compile(r'^fastlowess_jll = "\d+\.\d+\.\d+"$', re.MULTILINE),
            f'fastlowess_jll = "{new_version}"',
            1,
        )
    )

    targets.append(
        (
            "bindings/go/fastlowess/version.go",
            re.compile(r'const version = "\d+\.\d+\.\d+"'),
            f'const version = "{new_version}"',
            1,
        )
    )

    targets.append(
        (
            "bindings/cpp/CMakeLists.txt",
            re.compile(
                r"project\(fastlowess-cpp VERSION \d+\.\d+\.\d+ LANGUAGES CXX\)"
            ),
            f"project(fastlowess-cpp VERSION {new_version} LANGUAGES CXX)",
            1,
        )
    )

    targets.append(
        (
            "CITATION.cff",
            re.compile(r'^version: "\d+\.\d+\.\d+"$', re.MULTILINE),
            f'version: "{new_version}"',
            1,
        )
    )

    targets.append(
        (
            "bindings/cpp/spack/package.py",
            re.compile(r"archive/refs/tags/v\d+\.\d+\.\d+\.tar\.gz"),
            f"archive/refs/tags/v{new_version}.tar.gz",
            1,
        )
    )

    targets.append(
        (
            "bindings/r/inst/CITATION",
            re.compile(r'note\s*=\s*"R package version \d+\.\d+\.\d+"'),
            f'note    = "R package version {new_version}"',
            1,
        )
    )

    return targets


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", required=True, help="New version, e.g. 3.3.0")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without writing files"
    )
    args = parser.parse_args()

    if not VERSION_RE.match(args.version):
        parser.error(f"--version must be a plain semver X.Y.Z, got: {args.version!r}")

    new_version = args.version
    major, minor, _patch = new_version.split(".")
    new_major_minor = f"{major}.{minor}"

    print(
        f"Bumping version to {new_version} (path-dependency requirements -> {new_major_minor})"
    )
    if args.dry_run:
        print("[dry-run] no files will be written")
    print()

    all_ok = True
    for rel, pattern, replacement, count in build_targets(new_version, new_major_minor):
        ok = _replace(REPO_ROOT / rel, pattern, replacement, args.dry_run, count=count)
        all_ok = all_ok and ok

    print()
    all_ok = apply_go_module_suffix(major, args.dry_run) and all_ok

    print()
    if not all_ok:
        print(
            "Some files were not updated -- check the WARNING/SKIP lines above.",
            file=sys.stderr,
        )
        return 1

    print(f"Done{' (dry run)' if args.dry_run else ''}. Next steps:")
    print("  1. Add a new section to CHANGELOG.md for this version.")
    print(
        "  2. Run `python dev/update_changelogs.py <lang>` (or `make <lang>-dev`) to regenerate each binding's NEWS.md."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
