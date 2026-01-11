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
        description="Update version in various project files from Cargo.toml"
    )
    parser.add_argument(
        "cargo_toml",
        type=Path,
        help="Path to root Cargo.toml with workspace version",
    )
    parser.add_argument(
        "-r", "--r_citation_file",
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
        "-j", "--julia_project_file",
        type=Path,
        help="Path to Julia Project.toml file",
    )
    parser.add_argument(
        "-b", "--build_tarballs_file",
        type=Path,
        help="Path to Julia build_tarballs.jl file",
    )
    parser.add_argument(
        "-n", "--nodejs_package_file",
        type=Path,
        help="Path to Node.js package.json file",
    )
    parser.add_argument(
        "-N", "--nodejs_npm_dir",
        type=Path,
        help="Path to Node.js npm subpackages directory",
    )
    parser.add_argument(
        "-w", "--workflow_file",
        type=Path,
        help="Path to GitHub workflow file (e.g. release-julia.yml)",
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

    try:
        version = get_workspace_version(args.cargo_toml)
        if args.r_citation_file:
            update_citation(args.r_citation_file, version, args.quiet)
        if args.description_file:
            update_description(args.description_file, version, args.quiet)
        if args.python_version_file:
            update_python_version(args.python_version_file, version, args.quiet)
        if args.cff_file:
            update_cff_version(args.cff_file, version, args.quiet)
        if args.julia_project_file:
            update_julia_project_version(args.julia_project_file, version, args.quiet)
        if args.build_tarballs_file:
            update_build_tarballs_version(args.build_tarballs_file, version, args.quiet)
        if args.nodejs_package_file:
            update_package_json(args.nodejs_package_file, version, args.quiet)
        if args.nodejs_npm_dir:
            update_npm_subpackages(args.nodejs_npm_dir, version, args.quiet)
        if args.workflow_file:
            update_workflow_file(args.workflow_file, version, args.quiet)
            
        # Always update the dependencies in Cargo.toml itself
        update_cargo_toml_dependencies(args.cargo_toml, version, args.quiet)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def update_cargo_toml_dependencies(cargo_toml_path: Path, version: str, quiet: bool = False) -> bool:
    """Update internal dependency versions in Cargo.toml."""
    content = cargo_toml_path.read_text()
    
    # Update lowess/fastLowess dependencies
    # Match: crate = { ... version = "old_version" }
    patterns = [
        r'(lowess\s*=\s*{[^}]*version\s*=\s*")[^"]+(")',
        r'(fastLowess\s*=\s*{[^}]*version\s*=\s*")[^"]+(")',
    ]
    
    replacement = rf'\g<1>{version}\g<2>'
    
    new_content = content
    count_total = 0
    
    for pattern in patterns:
        new_content, count = re.subn(pattern, replacement, new_content)
        count_total += count
    
    if count_total == 0:
        if not quiet:
             print(f"Warning: No lowess/fastLowess dependency patterns found in {cargo_toml_path}")
        return False

    if new_content != content:
        cargo_toml_path.write_text(new_content)
        if not quiet:
            print(f"Updated Cargo.toml internal dependencies to version {version}")
        return True
    else:
        if not quiet:
             print(f"Cargo.toml internal dependencies already at version {version}")
        return False


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


def update_julia_project_version(project_path: Path, version: str, quiet: bool = False) -> bool:
    """Update the version in Julia Project.toml file."""
    if not project_path.exists():
        if not quiet:
            print(f"Warning: Julia Project.toml file not found at {project_path}")
        return False
        
    content = project_path.read_text()
    
    # Match: version = "X.Y.Z"
    pattern = r'(version\s*=\s*")[^"]+(")'
    replacement = rf'\g<1>{version}\g<2>'
    
    new_content, count = re.subn(pattern, replacement, content)
    
    if count == 0:
        if not quiet:
            print(f"Warning: No version field found in {project_path}")
        return False

    if new_content != content:
        project_path.write_text(new_content)
        if not quiet:
            print(f"Updated Julia Project.toml version to {version}")
        return True
    else:
        if not quiet:
            print(f"Julia Project.toml version already at {version}")
        return False


def update_build_tarballs_version(build_path: Path, version: str, quiet: bool = False) -> bool:
    """Update the version in Julia build_tarballs.jl file."""
    if not build_path.exists():
        if not quiet:
            print(f"Warning: build_tarballs.jl file not found at {build_path}")
        return False
        
    content = build_path.read_text()
    
    # Match: version = v"X.Y.Z"
    pattern = r'(version\s*=\s*v")[^"]+(")'
    replacement = rf'\g<1>{version}\g<2>'
    
    new_content, count = re.subn(pattern, replacement, content)
    
    if count == 0:
        if not quiet:
            print(f"Warning: No version field found in {build_path}")
        return False

    if new_content != content:
        build_path.write_text(new_content)
        if not quiet:
            print(f"Updated build_tarballs.jl version to {version}")
        return True
    else:
        if not quiet:
            print(f"build_tarballs.jl version already at {version}")
        return False


def update_package_json(package_path: Path, version: str, quiet: bool = False) -> bool:
    """Update the version in Node.js package.json file."""
    if not package_path.exists():
        if not quiet:
            print(f"Warning: package.json file not found at {package_path}")
        return False
        
    content = package_path.read_text()
    
    # Match: "version": "X.Y.Z"
    pattern = r'("version"\s*:\s*")[^"]+(")'
    replacement = rf'\g<1>{version}\g<2>'
    
    new_content, count = re.subn(pattern, replacement, content)

    # Match: "fastlowess-.*": "^X.Y.Z" (in optionalDependencies)
    # matching specifically fastlowess- prefixed packages to avoid touching others
    dep_pattern = r'("fastlowess-[a-z0-9-]+"\s*:\s*"\^)[^"]+(")'
    dep_replacement = rf'\g<1>{version}\g<2>'
    
    new_content, dep_count = re.subn(dep_pattern, dep_replacement, new_content)
    
    if count == 0:
        if not quiet:
            print(f"Warning: No version field found in {package_path}")
        return False
    if new_content != content:
        package_path.write_text(new_content)
        if not quiet:
            print(f"Updated package.json version to {version}")
        return True
    else:
        if not quiet:
            print(f"package.json version already at {version}")
        return False


def update_npm_subpackages(npm_dir: Path, version: str, quiet: bool = False) -> None:
    """Update version in all package.json files within subdirectories."""
    if not npm_dir.exists():
        if not quiet:
            print(f"Warning: npm directory not found at {npm_dir}")
        return

    for pkg_dir in npm_dir.iterdir():
        if pkg_dir.is_dir():
            pkg_json = pkg_dir / "package.json"
            if pkg_json.exists():
                update_package_json(pkg_json, version, quiet)


if __name__ == "__main__":
    main()
