#!/usr/bin/env python3
"""Bump the project version across every crate, binding, and manifest in the repo.

Covers: Cargo.toml package versions (Rust crates + all bindings), the internal
fastLowess/lowess path-dependency version requirements (major.minor), each
binding's own version file (package.json, pyproject-adjacent __version__.py,
pom.xml, DESCRIPTION, Project.toml, version.go, CMakeLists.txt, FastLowess.java),
CITATION.cff, and the Spack recipe's example `url`.

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


def build_targets(
    new_version: str, new_major_minor: str
) -> list[tuple[str, re.Pattern[str], str, int]]:
    """Return (relative_path, pattern, replacement, expected_count) tuples."""
    targets: list[tuple[str, re.Pattern[str], str, int]] = []

    cargo_pkg_pattern = re.compile(r'^version = "\d+\.\d+\.\d+"$', re.MULTILINE)
    for rel in CARGO_PACKAGE_FILES:
        targets.append((rel, cargo_pkg_pattern, f'version = "{new_version}"', 1))

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
