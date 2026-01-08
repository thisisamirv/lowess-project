#!/usr/bin/env python3
"""
Prepare the R package's Cargo.toml for vendoring and building.

This script handles:
1. Cleaning up workspace/patch sections before vendoring
2. Adding workspace isolation after vendoring
3. Temporarily excluding R package from root workspace during vendoring
"""

import argparse
import re
import sys
from pathlib import Path


def clean_for_vendoring(cargo_toml: Path) -> None:
    """Remove [workspace] and [patch.crates-io] sections for clean vendoring."""
    content = cargo_toml.read_text()
    
    # Remove [workspace] line
    content = re.sub(r'^\[workspace\]\s*\n', '', content, flags=re.MULTILINE)
    
    # Remove [patch.crates-io] section (multiline)
    content = re.sub(
        r'^\[patch\.crates-io\]\s*\n(?:.*\n)*?(?=\[|\Z)',
        '',
        content,
        flags=re.MULTILINE
    )
    
    # Remove trailing whitespace
    content = content.rstrip() + '\n'
    
    cargo_toml.write_text(content)


def add_workspace_isolation(cargo_toml: Path) -> None:
    """Add [workspace] and [patch.crates-io] sections for isolated build."""
    content = cargo_toml.read_text().rstrip()
    
    # Add workspace and patch sections
    content += '\n\n[workspace]\n\n[patch.crates-io]\nlowess = { path = "vendor/lowess" }\n'
    
    cargo_toml.write_text(content)


def exclude_from_root_workspace(root_toml: Path, member: str) -> Path:
    """Temporarily exclude a member from root workspace. Returns backup path."""
    backup = root_toml.with_suffix('.vendor-backup')
    
    # Create backup
    content = root_toml.read_text()
    backup.write_text(content)
    
    # Comment out the member
    pattern = rf'(\s*)("{member}",)'
    replacement = r'\1# \2  # temporarily excluded for vendoring'
    content = re.sub(pattern, replacement, content)
    
    root_toml.write_text(content)
    
    return backup


def restore_root_workspace(backup: Path, root_toml: Path) -> None:
    """Restore root workspace from backup."""
    if backup.exists():
        content = backup.read_text()
        root_toml.write_text(content)
        backup.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare R package Cargo.toml for vendoring/building"
    )
    parser.add_argument(
        "action",
        choices=["clean", "isolate", "exclude", "restore"],
        help="Action to perform",
    )
    parser.add_argument(
        "cargo_toml",
        type=Path,
        help="Path to Cargo.toml to modify",
    )
    parser.add_argument(
        "--member",
        type=str,
        default="bindings/r/src",
        help="Workspace member to exclude (for exclude action)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output",
    )
    
    args = parser.parse_args()
    
    if not args.cargo_toml.exists():
        print(f"Error: Cargo.toml not found: {args.cargo_toml}", file=sys.stderr)
        return 1
    
    if args.action == "clean":
        if not args.quiet:
            print(f"Cleaning {args.cargo_toml} for vendoring...")
        clean_for_vendoring(args.cargo_toml)
        
    elif args.action == "isolate":
        if not args.quiet:
            print(f"Adding workspace isolation to {args.cargo_toml}...")
        add_workspace_isolation(args.cargo_toml)
        
    elif args.action == "exclude":
        if not args.quiet:
            print(f"Excluding {args.member} from {args.cargo_toml}...")
        exclude_from_root_workspace(args.cargo_toml, args.member)
        
    elif args.action == "restore":
        backup = args.cargo_toml.with_suffix('.vendor-backup')
        if not args.quiet:
            print(f"Restoring {args.cargo_toml} from backup...")
        restore_root_workspace(backup, args.cargo_toml)
    
    if not args.quiet:
        print("Done")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
