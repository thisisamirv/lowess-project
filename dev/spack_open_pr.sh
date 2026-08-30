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
BRANCH="${PACKAGE_NAME}-${VERSION}"

LOCAL_PACKAGE_FILE_ABS="$(pwd)/${LOCAL_PACKAGE_FILE}"
FORK_OWNER="$(gh api user --jq .login)"

# Plain `git` commands (push/fetch) don't pick up GH_TOKEN on their own,
# only `gh` subcommands do; this wires gh's credential helper into git.
gh auth setup-git

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR" 2>/dev/null || true' EXIT
cd "$WORKDIR"

# gh forks (idempotent if already forked) and waits for the fork to be
# ready before cloning, avoiding a race with a freshly-created fork.
# Cloning a fork automatically configures "upstream" -> spack/spack-packages.
gh repo fork "$UPSTREAM_REPO" --clone=true
cd spack-packages

git fetch upstream develop
git checkout -B "$BRANCH" upstream/develop

mkdir -p "repos/spack_repo/builtin/packages/${SPACK_DIR_NAME}"
cp "$LOCAL_PACKAGE_FILE_ABS" "repos/spack_repo/builtin/packages/${SPACK_DIR_NAME}/package.py"

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
git add "repos/spack_repo/builtin/packages/${SPACK_DIR_NAME}/package.py"

if git diff --cached --quiet; then
	echo "No changes for ${PACKAGE_NAME} ${VERSION}; skipping PR."
	exit 0
fi

git commit -m "${PACKAGE_NAME}: add v${VERSION}"
git push --force origin "$BRANCH"

gh pr create \
	--repo "$UPSTREAM_REPO" \
	--head "${FORK_OWNER}:${BRANCH}" \
	--base develop \
	--title "${PACKAGE_NAME}: add v${VERSION}" \
	--body "Adds version ${VERSION} of \`${PACKAGE_NAME}\`. Recipe is maintained upstream at the package's own repository and mirrored here automatically on each release." ||
	echo "PR creation failed or a PR already exists for branch ${BRANCH}; check ${UPSTREAM_REPO} pull requests."
