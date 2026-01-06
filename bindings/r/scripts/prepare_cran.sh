#!/bin/bash
set -e

echo "📦 Preparing package for CRAN submission..."

# 1. Extract vendor archive if needed
echo "   -> Extracting vendored dependencies..."
if [ -f src/vendor.tar.xz ] && [ ! -d src/vendor ]; then
    (cd src && tar --extract --xz -f vendor.tar.xz)
fi

# 2. Ensure cargo config exists
mkdir -p src/cargo
if [ ! -f src/cargo/config.toml ]; then
    echo "   -> Creating cargo config..."
    cat > src/cargo/config.toml << 'EOF'
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "vendor"
EOF
fi

# 3. Generate AUTHORS file
echo "   -> Generating inst/AUTHORS..."
mkdir -p inst
(cd src && cargo metadata --format-version 1 > ../cargo_metadata_temp.json)

python3 -c '
import json
with open("cargo_metadata_temp.json") as f: data = json.load(f)

seen = set()
with open("inst/AUTHORS", "w") as f:
    f.write("Authors and Copyright Holders for Rust Dependencies:\n\n")
    for pkg in data["packages"]:
        name = pkg["name"]
        if name == "fastLowess-R": continue 
        
        # Deduplicate
        if name in seen: continue
        seen.add(name)
        
        version = pkg["version"]
        authors = ", ".join(pkg["authors"])
        license = pkg.get("license", "Unknown")
        
        f.write(f"Package: {name} ({version})\n")
        f.write(f"Authors: {authors}\n")
        f.write(f"License: {license}\n")
        f.write("-" * 40 + "\n")
'
rm cargo_metadata_temp.json

echo "✅ Preparation complete!"
echo "   1. Dependencies are in 'src/vendor/'"
echo "   2. Local config is in 'src/cargo/config.toml'"
echo "   3. Attribution is in 'inst/AUTHORS'"
echo ""
echo "You can now run: make install"
