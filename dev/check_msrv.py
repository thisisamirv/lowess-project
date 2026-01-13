import json
import subprocess
import sys
import re
import os

def parse_version(v_str):
    """
    Parses a version string (e.g., '1.70.0' or '1.70') into a tuple of integers.
    """
    if not v_str:
        return None
    try:
        parts = v_str.split('.')
        return tuple(map(int, parts))
    except (ValueError, AttributeError):
        return None

def main():
    try:
        # Run cargo metadata
        result = subprocess.run(
            ["cargo", "metadata", "--format-version=1", "--all-features"],
            capture_output=True,
            text=True,
            check=True
        )
        metadata = json.loads(result.stdout)
        
        # Find all Cargo.toml files in crates/ and bindings/
        target_dirs = ["crates", "bindings"]
        # Map dependency name to set of Cargo.toml paths that use it
        direct_deps = {}
        project_rust_version_str = None
        
        # First check root Cargo.toml for workspace rust-version
        try:
            if os.path.exists("Cargo.toml"):
                with open("Cargo.toml", "r") as f:
                    for line in f:
                        if line.strip().startswith("rust-version"):
                            match = re.search(r'rust-version\s*=\s*["\']([^"\']+)["\']', line)
                            if match:
                                project_rust_version_str = match.group(1)
                                break
        except Exception as e:
            print(f"Error reading root Cargo.toml: {e}", file=sys.stderr)

        # Now scan subdirectories for dependencies
        for root_dir in target_dirs:
            if not os.path.isdir(root_dir):
                continue
                
            for root, dirs, files in os.walk(root_dir):
                if "Cargo.toml" in files:
                    toml_path = os.path.join(root, "Cargo.toml")
                    # Skip vendor directories
                    if "vendor" in toml_path.split(os.sep):
                        continue
                        
                    try:
                        with open(toml_path, "r") as f:
                            in_deps = False
                            for line in f:
                                line = line.strip()
                                # Handle dependency sections
                                if line.startswith("[dependencies]") or line.startswith("[dev-dependencies]") or line.startswith("[build-dependencies]"):
                                    in_deps = True
                                    continue
                                elif line.startswith("[") and not (line.startswith("[dependencies.") or line.startswith("[dev-dependencies.") or line.startswith("[build-dependencies.")):
                                    in_deps = False
                                    continue
                                
                                if in_deps and not line.startswith("#") and line:
                                    dep_name = None
                                    # Extract package name: "name = ..." or "name"
                                    if "=" in line:
                                        dep_name = line.split("=")[0].strip()
                                    elif not line.startswith("["): # Simple key in table
                                        dep_name = line.strip()

                                    if dep_name:
                                        if dep_name not in direct_deps:
                                            direct_deps[dep_name] = set()
                                        direct_deps[dep_name].add(toml_path)

                    except Exception as e:
                        print(f"Warning: Failed to read {toml_path}: {e}", file=sys.stderr)

        if not project_rust_version_str:
            print("Warning: Could not find 'rust-version' in [workspace.package]. Skipping validation.", file=sys.stderr)
            sys.exit(0)

        project_rust_ver = parse_version(project_rust_version_str)
        print(f"Project Rust Version: {project_rust_version_str}")
        print("-" * 80)
        print(f"{'Package':<25} {'Version':<15} {'MSRV':<15} {'Status'}")
        print("-" * 80)
        
        violations = []

        for package in metadata["packages"]:
            name = package["name"]
            version = package["version"]
            msrv_str = package.get("rust_version")
            
            # Filter: Only check direct dependencies found in scans
            if name in direct_deps:
                status = "OK"
                msrv_valid = True
                
                if msrv_str:
                    msrv_ver = parse_version(msrv_str)
                    if msrv_ver and project_rust_ver:
                        # Pad versions for comparison (e.g., 1.70 vs 1.70.0)
                        # Normally Rust versions are semver, but MSRV can be partial
                        if msrv_ver > project_rust_ver:
                            status = "FAIL (Too New)"
                            violations.append((name, msrv_str, direct_deps[name]))
                            msrv_valid = False
                
                display_msrv = msrv_str if msrv_str else "N/A"
                print(f"{name:<25} {version:<15} {display_msrv:<15} {status}")

        print("-" * 80)
        if violations:
            print("\nERROR: The following packages have an MSRV higher than the workspace rust-version:", file=sys.stderr)
            for name, ver, sources in violations:
                print(f"  - {name}: requires {ver} (project supports {project_rust_version_str})", file=sys.stderr)
                for source in sources:
                    print(f"    - Imported in: {source}", file=sys.stderr)
            sys.exit(1)
        else:
            print("\nSuccess: All direct dependencies satisfy the project MSRV.")

    except subprocess.CalledProcessError as e:
        print(f"Error running cargo metadata: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing metadata: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
