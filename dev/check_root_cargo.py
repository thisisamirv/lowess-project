#!/usr/bin/env python3
"""Guard the root Cargo workspace manifest against accidental isolation edits."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_MEMBERS = [
    "crates/lowess",
    "crates/fastLowess",
    "bindings/python",
    "bindings/julia",
    "bindings/nodejs",
    "bindings/wasm",
    "bindings/cpp",
    "examples",
    "tests",
    "validation/fastLowess",
    "benchmarks/fastLowess",
]


def run_git(repo_root: Path, *args: str) -> str:
    """Run a git command in the repository root and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def repo_root_from_script() -> Path:
    """Return the repository root derived from this script location."""
    return Path(__file__).resolve().parents[1]


def restore_backup_if_present(repo_root: Path) -> bool:
    """Restore the root manifest from `Cargo.toml.bak` if it exists."""
    backup = repo_root / "Cargo.toml.bak"
    cargo_toml = repo_root / "Cargo.toml"
    if not backup.exists():
        return False

    shutil.move(str(backup), str(cargo_toml))
    return True


def read_working_tree(repo_root: Path) -> str:
    """Read the working-tree root `Cargo.toml` contents."""
    return (repo_root / "Cargo.toml").read_text(encoding="utf-8")


def read_git_version(repo_root: Path, spec: str) -> str:
    """Read `Cargo.toml` from a git object spec such as `:` or `HEAD`."""
    object_spec = ":Cargo.toml" if spec == ":" else f"{spec}:Cargo.toml"
    return run_git(repo_root, "show", object_spec)


def validate_manifest(content: str, label: str) -> list[str]:
    """Validate that all required workspace members are active in the manifest."""
    errors: list[str] = []
    members_match = re.search(r"members\s*=\s*\[(?P<body>.*?)\]", content, re.DOTALL)
    if not members_match:
        return [f"{label}: could not find the workspace members list in Cargo.toml"]

    body = members_match.group("body")
    active_members = set(re.findall(r'^[ \t]*"([^"]+)"\s*,', body, re.MULTILINE))
    commented_members = set(re.findall(r'^[ \t]*#\s*"([^"]+)"\s*,', body, re.MULTILINE))

    for member in REQUIRED_MEMBERS:
        if member in commented_members:
            errors.append(
                f'{label}: workspace member "{member}" is commented out in Cargo.toml'
            )
        elif member not in active_members:
            errors.append(
                f'{label}: workspace member "{member}" is missing from Cargo.toml'
            )

    return errors


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the hook guard."""
    parser = argparse.ArgumentParser(
        description="Validate that the root Cargo workspace manifest is not left isolated."
    )
    parser.add_argument(
        "--hook",
        choices=("pre-commit", "pre-push"),
        required=True,
        help="Which git hook is invoking the validation.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the requested manifest validation for the invoking git hook."""
    args = parse_args()
    repo_root = repo_root_from_script()

    if restore_backup_if_present(repo_root):
        print(
            "Restored Cargo.toml from Cargo.toml.bak. Review the restored manifest, "
            "stage it again if needed, and retry.",
            file=sys.stderr,
        )
        return 1

    checks: list[tuple[str, str]] = [("working tree", read_working_tree(repo_root))]
    if args.hook == "pre-commit":
        checks.append(("staged", read_git_version(repo_root, ":")))
    else:
        checks.append(("HEAD", read_git_version(repo_root, "HEAD")))

    errors: list[str] = []
    for label, content in checks:
        errors.extend(validate_manifest(content, label))

    if errors:
        print(
            "Refusing to proceed because the root Cargo workspace manifest is still isolated:",
            file=sys.stderr,
        )
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print(
            "Run your Makefile target again or restore Cargo.toml before committing/pushing.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
