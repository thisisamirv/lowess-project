#!/usr/bin/env python3
"""
Patch vendored Cargo.toml files for R package build.

This script replaces workspace inheritance with concrete values and removes
GPU dependencies that are not needed for the R binding.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

import tomli_w


def load_workspace_values(workspace_toml: Path) -> Dict[str, Any]:
    """Load workspace package values from root Cargo.toml."""
    with open(workspace_toml, "rb") as f:
        data = tomllib.load(f)
    
    workspace = data.get("workspace", {})
    package = workspace.get("package", {})
    deps = workspace.get("dependencies", {})
    
    return {
        "package": package,
        "dependencies": deps,
    }


def resolve_workspace_value(value: Any, key: str, workspace: Dict[str, Any]) -> Any:
    """Resolve a workspace = true value to its concrete value."""
    if isinstance(value, dict) and value.get("workspace") is True:
        # Get from workspace.package
        pkg_value = workspace["package"].get(key)
        if pkg_value is not None:
            # Merge with any other fields in the original value
            result = {k: v for k, v in value.items() if k != "workspace"}
            if isinstance(pkg_value, dict):
                result.update(pkg_value)
            else:
                return pkg_value  # Simple value replacement
            return result if result else pkg_value
        return None  # Will be removed
    return value


def resolve_workspace_dep(dep_name: str, value: Any, workspace: Dict[str, Any]) -> Any:
    """Resolve a workspace dependency to its concrete value."""
    if isinstance(value, dict) and value.get("workspace") is True:
        ws_dep = workspace["dependencies"].get(dep_name, {})
        if isinstance(ws_dep, str):
            # Simple version string in workspace
            result = {"version": ws_dep}
        elif isinstance(ws_dep, dict):
            result = dict(ws_dep)
        else:
            result = {}
        
        # Merge with local overrides (features, optional, etc.)
        for k, v in value.items():
            if k != "workspace":
                result[k] = v
        
        return result
    return value


def patch_crate_toml(
    crate_toml: Path,
    workspace: Dict[str, Any],
    remove_gpu: bool = False,
    lowess_as_path: bool = False,
) -> None:
    """Patch a crate's Cargo.toml to remove workspace inheritance."""
    with open(crate_toml, "rb") as f:
        data = tomllib.load(f)
    
    # Patch [package] section
    package = data.get("package", {})
    keys_to_remove = []
    for key, value in package.items():
        resolved = resolve_workspace_value(value, key, workspace)
        if resolved is None:
            keys_to_remove.append(key)
        else:
            package[key] = resolved
    
    for key in keys_to_remove:
        del package[key]
    
    # Patch [dependencies] section
    if "dependencies" in data:
        deps = data["dependencies"]
        deps_to_remove = []
        
        for dep_name, value in deps.items():
            # Remove GPU deps if requested
            if remove_gpu and dep_name in ("wgpu", "bytemuck", "pollster", "futures-intrusive"):
                deps_to_remove.append(dep_name)
                continue
            
            # Handle lowess -> path dependency
            if lowess_as_path and dep_name == "lowess":
                resolved = resolve_workspace_dep(dep_name, value, workspace)
                if isinstance(resolved, dict):
                    resolved.pop("version", None)
                    resolved["path"] = "../lowess"
                deps[dep_name] = resolved
                continue
            
            deps[dep_name] = resolve_workspace_dep(dep_name, value, workspace)
        
        for dep_name in deps_to_remove:
            del deps[dep_name]
    
    # Patch [dev-dependencies] section
    if "dev-dependencies" in data:
        dev_deps = data["dev-dependencies"]
        for dep_name, value in dev_deps.items():
            dev_deps[dep_name] = resolve_workspace_dep(dep_name, value, workspace)
    
    # Patch [features] section - empty GPU feature if needed
    if remove_gpu and "features" in data:
        features = data["features"]
        if "gpu" in features:
            features["gpu"] = []
    
    # Write back
    with open(crate_toml, "wb") as f:
        tomli_w.dump(data, f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch vendored Cargo.toml files for R package build"
    )
    parser.add_argument(
        "workspace_toml",
        type=Path,
        help="Path to root Cargo.toml with workspace definitions",
    )
    parser.add_argument(
        "vendor_dir",
        type=Path,
        help="Path to vendor directory containing crates to patch",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output",
    )
    
    args = parser.parse_args()
    
    if not args.workspace_toml.exists():
        print(f"Error: Workspace Cargo.toml not found: {args.workspace_toml}", file=sys.stderr)
        return 1
    
    if not args.vendor_dir.exists():
        print(f"Error: Vendor directory not found: {args.vendor_dir}", file=sys.stderr)
        return 1
    
    # Load workspace values
    workspace = load_workspace_values(args.workspace_toml)
    
    # Patch fastLowess
    fast_lowess_toml = args.vendor_dir / "fastLowess" / "Cargo.toml"
    if fast_lowess_toml.exists():
        if not args.quiet:
            print(f"Patching {fast_lowess_toml}")
        patch_crate_toml(
            fast_lowess_toml,
            workspace,
            remove_gpu=True,
            lowess_as_path=True,
        )
    
    # Patch lowess
    lowess_toml = args.vendor_dir / "lowess" / "Cargo.toml"
    if lowess_toml.exists():
        if not args.quiet:
            print(f"Patching {lowess_toml}")
        patch_crate_toml(
            lowess_toml,
            workspace,
            remove_gpu=False,
            lowess_as_path=False,
        )
    
    if not args.quiet:
        print("Patching complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
