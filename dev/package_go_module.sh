#!/usr/bin/env bash
set -euo pipefail
package="${1:?Go package name is required}"
module_dir="bindings/go/${package}"
source_commit="$(git rev-parse HEAD)"
export GOWORK=off
export CGO_ENABLED=1
unset CGO_CFLAGS CGO_LDFLAGS GOFLAGS

test -n "${TAG:?Release tag is required}"
if git symbolic-ref -q HEAD >/dev/null; then
	echo "Package Go releases from a detached release checkout, not a working branch." >&2
	exit 1
fi
test -f "${module_dir}/include/${package}_go.h"
for platform in linux_amd64 linux_arm64 linux_amd64_musl linux_arm64_musl darwin_amd64 darwin_arm64 windows_amd64 windows_arm64; do
	test -s "${module_dir}/native/${platform}/lib${package}_go.a"
done

cargo about generate --manifest-path bindings/go/Cargo.toml --locked --config dev/about.toml dev/about.hbs \
	--output-file "${module_dir}/native/THIRD_PARTY_LICENSES.html"
jq -n --arg source_commit "$source_commit" --arg version "$TAG" \
	'{source_commit: $source_commit, version: $version}' >"${module_dir}/native/manifest.json"
test "$(du -sb "$module_dir" | cut -f1)" -lt 524288000

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
git add -f "${module_dir}/include" "${module_dir}/native"
: >"${module_dir}/native/SHA256SUMS"
while IFS= read -r -d '' filename; do
	if [ "$filename" != "${module_dir}/native/SHA256SUMS" ]; then
		checksum="$(git show ":${filename}" | sha256sum)"
		printf '%s  %s\n' "${checksum%% *}" "${filename#${module_dir}/}" >>"${module_dir}/native/SHA256SUMS"
	fi
done < <(git ls-files -z -- "${module_dir}/include" "${module_dir}/native")
git add -f "${module_dir}/native/SHA256SUMS"
git commit -m "go: bundle native artifacts for ${TAG}"

module_path="$(cd "$module_dir" && go list -m)"
staging="$(mktemp -d)"
proxy_dir="${staging}/proxy/${module_path}/@v"
mkdir -p "$proxy_dir"
git archive --format=zip --prefix="${module_path}@${TAG}/" \
	--output="${proxy_dir}/${TAG}.zip" "HEAD:${module_dir}"
cp "${module_dir}/go.mod" "${proxy_dir}/${TAG}.mod"
jq -n --arg version "$TAG" --arg time "$(git show -s --format=%cI "$source_commit")" \
	'{Version: $version, Time: $time}' >"${proxy_dir}/${TAG}.info"

GOPROXY="file://${staging}/proxy" GOSUMDB=off GOMODCACHE="${staging}/cache" \
	go mod download -json "${module_path}@${TAG}" >"${staging}/download.json"
downloaded_module="$(jq -r '.Dir' "${staging}/download.json")"
(
	cd "$downloaded_module"
	sha256sum -c native/SHA256SUMS
)
cp -R bindings/go/tests "${staging}/consumer"
go -C "${staging}/consumer" mod edit "-replace=${module_path}=${downloaded_module}"
go -C "${staging}/consumer" test ./...

if [ "${PUBLISH_GO_MODULE:-false}" = "true" ]; then
	module_tag="bindings/go/${package}/${TAG}"
	if git show-ref --verify --quiet "refs/tags/$module_tag"; then
		echo "Refusing to replace existing Go module tag: $module_tag" >&2
		exit 1
	fi
	git tag "$module_tag" HEAD
	git push origin "refs/tags/$module_tag"
fi
