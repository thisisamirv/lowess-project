#!/usr/bin/env python3
"""Updates a Spack package.py's version() list for a new release.

Downloads the GitHub release source tarball for the given tag, computes its
sha256, and inserts a `version("X.Y.Z", sha256="...")` line at the top of the
version list in the given package.py (Spack lists versions newest-first). If
a version() line for that version already exists, its sha256 is refreshed
in place instead of duplicating the entry.

Usage:
    python dev/spack_release.py --repo OWNER/REPO --tag v1.2.3 --package-file bindings/cpp/spack/package.py
"""

import argparse
import hashlib
import re
import sys
import urllib.request


def compute_sha256(url: str) -> str:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def update_package_file(package_file: str, version: str, sha256: str) -> None:
    with open(package_file, "r", encoding="utf-8") as f:
        content = f.read()

    existing_pattern = re.compile(
        rf'version\(\s*"{re.escape(version)}"\s*,\s*sha256\s*=\s*"[0-9a-f]*"\s*\)'
    )
    new_line = f'version("{version}", sha256="{sha256}")'

    if existing_pattern.search(content):
        content = existing_pattern.sub(new_line, content)
    else:
        first_version_pattern = re.compile(r"^(\s*)version\(", re.MULTILINE)
        match = first_version_pattern.search(content)
        if not match:
            print(
                f"error: no version() directive found in {package_file}",
                file=sys.stderr,
            )
            sys.exit(1)
        indent = match.group(1)
        insert_at = match.start()
        content = content[:insert_at] + f"{indent}{new_line}\n" + content[insert_at:]

    with open(package_file, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", required=True, help="GitHub repo, e.g. thisisamirv/lowess-project"
    )
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v3.1.0")
    parser.add_argument(
        "--package-file", required=True, help="Path to the Spack package.py to update"
    )
    args = parser.parse_args()

    version = args.tag.lstrip("v")
    url = f"https://github.com/{args.repo}/archive/refs/tags/{args.tag}.tar.gz"

    print(f"Downloading {url} to compute sha256...")
    sha256 = compute_sha256(url)
    print(f"sha256: {sha256}")

    update_package_file(args.package_file, version, sha256)
    print(f"Updated {args.package_file} with version {version}")


if __name__ == "__main__":
    main()
