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


def update_description(description_path: Path, version: str, quiet: bool = False) -> bool:
    """Update the version in R DESCRIPTION file."""
    if not description_path.exists():
        if not quiet:
            print(f"Warning: DESCRIPTION file not found at {description_path}")
        return False
        
    content = description_path.read_text()
    
    # Match: Version: X.Y.Z
    pattern = r'(Version:\s*).*'
    replacement = rf'\g<1>{version}'
    
    new_content, count = re.subn(pattern, replacement, content)
    
    if count == 0:
        if not quiet:
            print(f"Warning: No Version field found in {description_path}")
        return False

    if new_content != content:
        description_path.write_text(new_content)
        if not quiet:
            print(f"Updated DESCRIPTION to version {version}")
        return True
    else:
        if not quiet:
            print(f"DESCRIPTION already at version {version}")
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
        "-d", "--description_file",
        type=Path,
        help="Path to R DESCRIPTION file",
    )
    parser.add_argument(
        "-p", "--python_version_file",
        type=Path,
        help="Path to Python __version__.py file",
    )
    parser.add_argument(
        "-c", "--cff_file",
        type=Path,
        help="Path to CITATION.cff file",
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
        if args.description_file:
            update_description(args.description_file, version, args.quiet)
        if args.python_version_file:
            update_python_version(args.python_version_file, version, args.quiet)
        if args.cff_file:
            update_cff_version(args.cff_file, version, args.quiet)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def update_cff_version(cff_path: Path, version: str, quiet: bool = False) -> bool:
    """Update the version in CITATION.cff file."""
    if not cff_path.exists():
        if not quiet:
            print(f"Warning: CITATION.cff file not found at {cff_path}")
        return False
        
    content = cff_path.read_text()
    
    # Match: version: "X.Y.Z"
    pattern = r'(version:\s*")[^"]+(")'
    replacement = rf'\g<1>{version}\g<2>'
    
    new_content, count = re.subn(pattern, replacement, content)
    
    if count == 0:
        if not quiet:
            print(f"Warning: No version field found in {cff_path}")
        return False

    if new_content != content:
        cff_path.write_text(new_content)
        if not quiet:
            print(f"Updated CITATION.cff version to {version}")
        return True
    else:
        if not quiet:
            print(f"CITATION.cff version already at {version}")
        return False


def update_python_version(version_file_path: Path, version: str, quiet: bool = False) -> bool:
    """Update the version in Python __version__.py file."""
    if not version_file_path.exists():
        if not quiet:
            print(f"Warning: Python version file not found at {version_file_path}")
        return False
        
    content = version_file_path.read_text()
    
    # Match: __version__ = "X.Y.Z"
    pattern = r'(__version__\s*=\s*")[^"]+(")'
    replacement = rf'\g<1>{version}\g<2>'
    
    new_content, count = re.subn(pattern, replacement, content)
    
    if count == 0:
        if not quiet:
            print(f"Warning: No __version__ variable found in {version_file_path}")
        return False

    if new_content != content:
        version_file_path.write_text(new_content)
        if not quiet:
            print(f"Updated Python version to {version}")
        return True
    else:
        if not quiet:
            print(f"Python version already at {version}")
        return False


if __name__ == "__main__":
    main()
