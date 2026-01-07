import json
import subprocess

try:
    # Run cargo metadata
    result = subprocess.run(
        ["cargo", "metadata", "--format-version=1", "--all-features"],
        capture_output=True,
        text=True,
        check=True
    )
    metadata = json.loads(result.stdout)
    
    # Read Cargo.toml to get direct workspace dependencies
    direct_deps = set()
    try:
        with open("Cargo.toml", "r") as f:
            in_deps = False
            for line in f:
                line = line.strip()
                if line == "[workspace.dependencies]":
                    in_deps = True
                    continue
                if in_deps and line.startswith("["): # End of section
                    break
                if in_deps and line and not line.startswith("#"):
                    # Extract package name (key before =)
                    if "=" in line:
                         name = line.split("=")[0].strip()
                         direct_deps.add(name)
                    # Handle simple keys? usually deps are key = ...
    except Exception as e:
        print("Error reading Cargo.toml:", e)

    print(f"{'Package':<20} {'Version':<10} {'MSRV (rust-version)':<20}")
    print("-" * 50)
    
    for package in metadata["packages"]:
        name = package["name"]
        version = package["version"]
        msrv = package.get("rust_version")
        
        # Filter: Only show if in direct_deps
        if name in direct_deps:
            display_msrv = msrv if msrv else "N/A"
            print(f"{name:<20} {version:<10} {display_msrv:<20}")

except subprocess.CalledProcessError as e:
    print("Error running cargo metadata:", e)
except Exception as e:
    print("Error parsing metadata:", e)
