#!/usr/bin/env bash
# Remove LLVM/GCC LTO bitcode sections that newer Go cgo rejects
# ("disallowed LTO section .llvmbc" / ".llvmcmd"). Safe no-op when absent.
set -euo pipefail

archive="${1:?usage: strip-cgo-lto-sections.sh <libfastlowess_go.a>}"
if [ ! -f "$archive" ]; then
	echo "error: archive not found: $archive" >&2
	exit 1
fi

archive="$(cd "$(dirname "$archive")" && pwd)/$(basename "$archive")"

# ELF archives only. Darwin/Windows use different object formats; release-c
# already sets lto=false so residual bitcode is uncommon outside Linux.
case "$(uname -s)" in
Linux) ;;
*)
	echo "Skipping cgo LTO strip on $(uname -s) for $archive"
	exit 0
	;;
esac

if command -v llvm-objcopy >/dev/null 2>&1; then
	OBJCOPY=llvm-objcopy
elif command -v objcopy >/dev/null 2>&1; then
	OBJCOPY=objcopy
else
	host="$(rustc -vV | sed -n 's/^host: //p')"
	candidate="$(rustc --print sysroot)/lib/rustlib/${host}/bin/llvm-objcopy"
	if [ -x "$candidate" ]; then
		OBJCOPY="$candidate"
	else
		echo "error: need objcopy or llvm-objcopy to strip LTO sections" >&2
		exit 1
	fi
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
(
	cd "$tmpdir"
	ar x "$archive"
	for member in *; do
		[ -f "$member" ] || continue
		"$OBJCOPY" \
			--remove-section=.llvmbc \
			--remove-section=.llvmcmd \
			--remove-section=.llvm.lto \
			--remove-section=.gnu.lto \
			"$member" 2>/dev/null || true
	done
	rm -f "$archive"
	ar crs "$archive" ./*
)

echo "Stripped cgo-disallowed LTO sections from $archive"
