#!/bin/sh
# Creates vendor.tar.xz from local monorepo crates + crates.io deps.
# Must be run from bindings/r/src/. Leaves vendor/ in place for cargo builds.
set -e

WITH_GPU="${WITH_GPU:-0}"

rm -rf vendor vendor.tar.xz
mkdir -p vendor

# Copy local monorepo crates (paths relative to bindings/r/src/)
cp -rL ../../../crates/fastLowess vendor/fastLowess
cp -rL ../../../crates/lowess vendor/lowess

# Strip build artefacts and noise
rm -rf vendor/fastLowess/target vendor/lowess/target
rm -f vendor/fastLowess/Cargo.lock vendor/lowess/Cargo.lock
for d in tests benches examples doc docs assets .github .config; do
	rm -rf "vendor/fastLowess/$d" "vendor/lowess/$d" 2>/dev/null || true
done
rm -f vendor/fastLowess/README.md vendor/fastLowess/CHANGELOG.md \
	vendor/lowess/README.md vendor/lowess/CHANGELOG.md

# Patch vendored fastLowess: strip GPU deps unless WITH_GPU=1
if [ "$WITH_GPU" != "1" ]; then
	sed -i.bak \
		-e '/^wgpu = /,/^\] }/d' \
		-e '/^bytemuck = /d' \
		-e '/^pollster = /d' \
		-e '/^futures-intrusive = /d' \
		-e 's/^gpu = .*/gpu = []/' \
		-e 's|lowess = { path = "\.\./lowess", version = "[^"]*", |lowess = { path = "../lowess", |' \
		vendor/fastLowess/Cargo.toml
else
	sed -i.bak \
		-e 's|lowess = { path = "\.\./lowess", version = "[^"]*", |lowess = { path = "../lowess", |' \
		vendor/fastLowess/Cargo.toml
fi
rm -f vendor/fastLowess/Cargo.toml.bak

# Checksum placeholders for manually-placed path-dep crates
printf '{"files":{},"package":null}' >vendor/lowess/.cargo-checksum.json
printf '{"files":{},"package":null}' >vendor/fastLowess/.cargo-checksum.json

# Temporarily isolate from the monorepo workspace so cargo vendor scopes
# only to this package, then restore the clean Cargo.toml immediately after
printf '\n\n[patch.crates-io]\nlowess = { path = "vendor/lowess" }\n' >>Cargo.toml
cargo vendor -q --no-delete vendor
sed -i.bak '/^\[patch\.crates-io\]/d;/^lowess = { path = "vendor\/lowess" }/d' \
	Cargo.toml
rm -f Cargo.toml.bak

# Drop directories that bulk up the archive
for d in tests benches examples doc docs assets .github .config; do
	rm -rf "vendor/$d" vendor/*/"$d" 2>/dev/null || true
done
for f in vendor/*/Makefile; do [ -f "$f" ] && rm -f "$f"; done || true
rm -f vendor/*/CITATION.cff vendor/*/CITATION

# Nullify file-level checksums so removed test/bench/doc files don't cause
# "failed to open file" when cargo verifies vendor integrity
for f in vendor/*/.cargo-checksum.json; do
	[ -f "$f" ] || continue
	python3 -c "import json; p='$f'; d=json.load(open(p)); d['files']=[]; json.dump({'files':{},'package':d.get('package')},open(p,'w'))"
done

# Create reproducible archive; include Cargo.lock for reproducible installs
tar --sort=name --mtime='1970-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner \
	--xz --create --file=vendor.tar.xz --exclude='*/Makefile' vendor Cargo.lock \
	2>/dev/null ||
	tar --xz --create --file=vendor.tar.xz --exclude='*/Makefile' vendor Cargo.lock

echo "vendor.tar.xz created"
# vendor/ is left in place for subsequent cargo builds in the same session
