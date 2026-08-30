#!/usr/bin/env bash
# Opens/updates a PR against spack/spack-packages with this repo's Spack
# recipe (bindings/cpp/spack/package.py), for the given package name and
# release tag. Requires GH_TOKEN (a PAT with repo scope) to be set, and the
# gh CLI to be authenticated with it.
#
# Usage: dev/spack_open_pr.sh <package-name> <local-package-file> <tag>
# Example: dev/spack_open_pr.sh fastlowess-cpp bindings/cpp/spack/package.py v3.1.0
set -euo pipefail

PACKAGE_NAME="$1"
LOCAL_PACKAGE_FILE="$2"
TAG="$3"
UPSTREAM_REPO="spack/spack-packages"
SPACK_DIR_NAME="${PACKAGE_NAME//-/_}"
VERSION="${TAG#v}"

LOCAL_PACKAGE_FILE_ABS="$(pwd)/${LOCAL_PACKAGE_FILE}"

FORK_OWNER="$(gh api user --jq .login)"
FORK_REPO="${FORK_OWNER}/spack-packages"

# Create the fork if it doesn't exist yet (idempotent; no-op if already forked)
gh repo fork "$UPSTREAM_REPO" --clone=false --remote=false >/dev/null 2>&1 || true

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

git clone --depth=1 --branch develop "https://x-access-token:${GH_TOKEN}@github.com/${UPSTREAM_REPO}.git" "$WORKDIR"
cd "$WORKDIR"
git remote add fork "https://x-access-token:${GH_TOKEN}@github.com/${FORK_REPO}.git"

BRANCH="${PACKAGE_NAME}-${VERSION}"
git checkout -B "$BRANCH"

mkdir -p "repos/spack_repo/builtin/packages/${SPACK_DIR_NAME}"
cp "$LOCAL_PACKAGE_FILE_ABS" "repos/spack_repo/builtin/packages/${SPACK_DIR_NAME}/package.py"

if git diff --quiet && git diff --cached --quiet; then
	echo "No changes for ${PACKAGE_NAME} ${VERSION}; skipping PR."
	exit 0
fi

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
git add "repos/spack_repo/builtin/packages/${SPACK_DIR_NAME}/package.py"
git commit -m "${PACKAGE_NAME}: add v${VERSION}"
git push --force fork "$BRANCH"

gh pr create \
	--repo "$UPSTREAM_REPO" \
	--head "${FORK_OWNER}:${BRANCH}" \
	--base develop \
	--title "${PACKAGE_NAME}: add v${VERSION}" \
	--body "Adds version ${VERSION} of \`${PACKAGE_NAME}\`. Recipe is maintained upstream at the package's own repository and mirrored here automatically on each release." \
	2>&1 | tee /dev/stderr | grep -q "already exists" && echo "PR already open for this branch." || true
