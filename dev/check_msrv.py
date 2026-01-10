import json
import subprocess
import sys
import re

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
        
        # Read Cargo.toml to get direct workspace dependencies and project rust-version
        direct_deps = set()
        project_rust_version_str = None
        
        try:
            with open("Cargo.toml", "r") as f:
                in_deps = False
                in_package = False
                
                for line in f:
                    line = line.strip()
                    
                    # Track sections
                    if line == "[workspace.dependencies]":
                        in_deps = True
                        in_package = False
                        continue
                    elif line == "[workspace.package]":
                        in_package = True
                        in_deps = False
                        continue
                    elif line.startswith("["):
                        in_deps = False
                        in_package = False
                        continue
                    
                    if not line or line.startswith("#"):
                        continue

                    # Extract rust-version from [workspace.package]
                    if in_package and line.startswith("rust-version"):
                        # Format: rust-version = "1.85.0"
                        match = re.search(r'rust-version\s*=\s*["\']([^"\']+)["\']', line)
                        if match:
                            project_rust_version_str = match.group(1)

                    # Extract direct dependencies
                    if in_deps:
                        if "=" in line:
                            name = line.split("=")[0].strip()
                            direct_deps.add(name)
                        # Handle simple keys (dep = "x.y")
                    
        except Exception as e:
            print(f"Error reading Cargo.toml: {e}", file=sys.stderr)
            sys.exit(1)

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
            
            # Filter: Only check direct dependencies or members
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
                            violations.append((name, msrv_str))
                            msrv_valid = False
                
                display_msrv = msrv_str if msrv_str else "N/A"
                print(f"{name:<25} {version:<15} {display_msrv:<15} {status}")

        print("-" * 80)
        if violations:
            print("\nERROR: The following packages have an MSRV higher than the workspace rust-version:", file=sys.stderr)
            for name, ver in violations:
                print(f"  - {name}: requires {ver} (project supports {project_rust_version_str})", file=sys.stderr)
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
