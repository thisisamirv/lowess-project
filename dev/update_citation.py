#!/usr/bin/env python3
"""Update version in R CITATION file from Cargo.toml workspace."""

import argparse
import re
import sys
from pathlib import Path


def get_workspace_version(cargo_toml_path: Path) -> str:
    """Extract version from workspace Cargo.toml."""
    content = cargo_toml_path.read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError(f"Could not find version in {cargo_toml_path}")
    return match.group(1)


def update_citation(citation_path: Path, version: str, quiet: bool = False) -> bool:
    """Update the version in R CITATION file."""
    content = citation_path.read_text()

    # Match: note = "R package version X.Y.Z"
    pattern = r'(note\s*=\s*"R package version )[^"]+"'
    replacement = rf'\g<1>{version}"'

    new_content, count = re.subn(pattern, replacement, content)

    if count == 0:
        if not quiet:
            print(f"Warning: No version pattern found in {citation_path}")
        return False

    if new_content != content:
        citation_path.write_text(new_content)
        if not quiet:
            print(f"Updated CITATION to version {version}")
        return True
    else:
        if not quiet:
            print(f"CITATION already at version {version}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update R CITATION file version from Cargo.toml"
    )
    parser.add_argument(
        "cargo_toml",
        type=Path,
        help="Path to root Cargo.toml with workspace version",
    )
    parser.add_argument(
        "citation_file",
        type=Path,
        help="Path to R CITATION file",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output",
    )
    args = parser.parse_args()

    if not args.cargo_toml.exists():
        print(f"Error: {args.cargo_toml} not found", file=sys.stderr)
        sys.exit(1)

    if not args.citation_file.exists():
        print(f"Error: {args.citation_file} not found", file=sys.stderr)
        sys.exit(1)

    try:
        version = get_workspace_version(args.cargo_toml)
        update_citation(args.citation_file, version, args.quiet)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
