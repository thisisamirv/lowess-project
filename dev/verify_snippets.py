"""Extract fenced code blocks from docs/ and run them to verify correctness.

Each snippet is prefixed with language-specific boilerplate that provides common
sample data and imports, so fragment-style doc examples can execute end-to-end.

Usage
-----
    python dev/verify_snippets.py                    # run all supported languages
    python dev/verify_snippets.py --lang python      # Python only
    python dev/verify_snippets.py --lang nodejs      # Node.js only
    python dev/verify_snippets.py --lang julia       # Julia only
    python dev/verify_snippets.py --lang r           # R only
    python dev/verify_snippets.py --lang wasm        # WebAssembly only
    python dev/verify_snippets.py --lang rust        # Rust only
    python dev/verify_snippets.py --lang cpp         # C++ only
    python dev/verify_snippets.py --file docs/api/python.md
    python dev/verify_snippets.py --dry-run          # list snippets, don't run
    python dev/verify_snippets.py --verbose          # show snippet source on failure
    python dev/verify_snippets.py --output out.json  # also write JSON report
    python dev/verify_snippets.py --timeout 60       # per-snippet timeout (seconds)
    python dev/verify_snippets.py --stop-on-fail     # exit after first failure
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import runners.python as _python_runner
from runners import RUNNERS, SKIP_CHECKS
from runners.base import (
    DOCS_DIR,
    JULIA_DOCS_DIR,
    NODEJS_BINDING_DOCS_DIR,
    REPO_ROOT,
    RUST_CRATE_DOCS_DIRS,
    VIGNETTES_DIRS,
    WASM_BINDING_DOCS_DIR,
    RunResult,
    Snippet,
)
from runners.r import skip_reason as _r_skip_reason

# ---------------------------------------------------------------------------
# Terminal colours (disabled on non-TTY or Windows without colour support)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.name != "nt" or os.environ.get("FORCE_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def green(t: str) -> str:
    return _c("32", t)


def red(t: str) -> str:
    return _c("31", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def bold(t: str) -> str:
    return _c("1", t)


# ---------------------------------------------------------------------------
# Python executable detection (prefer venv where fastlowess is installed)
# ---------------------------------------------------------------------------


def _find_python_with_fastlowess() -> str:
    """Return the best Python executable that has fastlowess installed."""
    candidates = [
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",  # Windows root venv
        REPO_ROOT / ".venv" / "bin" / "python",  # Unix root venv
        REPO_ROOT / "bindings" / "python" / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / "bindings" / "python" / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            try:
                r = subprocess.run(
                    [str(c), "-c", "import fastlowess"],
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
                if r.returncode == 0:
                    return str(c)
            except OSError:
                continue
    return sys.executable


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------

_TAB_RE = re.compile(r'^[ \t]*===\s+"([^"]+)"', re.MULTILINE)


def _rmd_chunk_is_runnable(opts_str: str) -> bool:
    """Return False if an Rmd chunk header has eval=FALSE or include=FALSE."""
    opts = opts_str.lower().replace(" ", "")
    return "eval=false" not in opts and "include=false" not in opts


def extract_snippets(md_file: Path) -> list[Snippet]:
    """Extract all fenced code blocks from a markdown file (also handles .Rmd).

    For .Rmd files all runnable R chunks are combined into one snippet, since
    Rmd chunks share state (later chunks depend on variables set by earlier ones).
    """
    text = md_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    result: list[Snippet] = []
    is_rmd = md_file.suffix.lower() == ".rmd"

    current_tab: str | None = None
    rmd_chunks: list[tuple[int, str, str]] = []  # (start_line, chunk_name, code)
    i = 0
    while i < len(lines):
        line = lines[i]

        m = _TAB_RE.match(line)
        if m:
            current_tab = m.group(1)
            i += 1
            continue

        # Rmd-style fence: ```{r} or ```{r chunk-name} or ```{r chunk-name, opts}
        m_rmd = re.match(r"^([ \t]*)```\{r([^}]*)\}\s*$", line)
        if m_rmd:
            fence_indent = m_rmd.group(1)
            raw_opts = m_rmd.group(2)
            parts = raw_opts.split(",", 1)
            chunk_name = parts[0].strip() or ""
            opts_str = parts[1] if len(parts) > 1 else ""
            code_lines_rmd: list[str] = []
            start_line_rmd = i + 1
            i += 1
            while i < len(lines):
                if re.match(r"^" + re.escape(fence_indent) + r"```\s*$", lines[i]):
                    i += 1
                    break
                code_lines_rmd.append(lines[i].removeprefix(fence_indent))
                i += 1
            if _rmd_chunk_is_runnable(opts_str):
                chunk_code = "\n".join(code_lines_rmd)
                # Skip chunks that are individually not runnable (e.g. install.packages).
                # A single bad chunk must not suppress the whole combined vignette snippet.
                dummy = Snippet(
                    file=md_file,
                    line=start_line_rmd,
                    lang_tag="r",
                    tab=None,
                    code=chunk_code,
                )
                if _r_skip_reason(dummy) is None:
                    rmd_chunks.append((start_line_rmd, chunk_name, chunk_code))
            continue

        m = re.match(r"^([ \t]*)```(\w+)\s*$", line)
        if m:
            fence_indent = m.group(1)
            lang_tag = m.group(2)
            start_line = i + 1  # 1-based
            code_lines: list[str] = []
            i += 1
            while i < len(lines):
                close = lines[i]
                if re.match(r"^" + re.escape(fence_indent) + r"```\s*$", close):
                    i += 1
                    break
                code_lines.append(close.removeprefix(fence_indent))
                i += 1
            code = "\n".join(code_lines)
            result.append(
                Snippet(
                    file=md_file,
                    line=start_line,
                    lang_tag=lang_tag,
                    tab=current_tab,
                    code=code,
                )
            )
            current_tab = None
            continue

        if line.startswith("#") or line.strip() == "---":
            current_tab = None

        i += 1

    if is_rmd and rmd_chunks:
        first_line = rmd_chunks[0][0]
        combined_code = "\n\n".join(code for _, _, code in rmd_chunks)
        names = [n for _, n, _ in rmd_chunks if n]
        tab_label = (
            f"{len(rmd_chunks)} chunks: {', '.join(names)}"
            if names
            else f"{len(rmd_chunks)} chunks"
        )
        result.append(
            Snippet(
                file=md_file,
                line=first_line,
                lang_tag="r",
                tab=tab_label,
                code=combined_code,
            )
        )

    return result


def should_skip(snippet: Snippet, runner: str) -> str | None:
    """Return a skip reason string, or None if the snippet should be run."""
    code = snippet.code
    if "--8<--" in code:
        return "file-include (--8<--)"
    if not code.strip():
        return "empty"
    check = SKIP_CHECKS.get(runner)
    return check(snippet) if check else None


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def iter_md_files(root: Path, file_filter: str | None) -> Iterator[Path]:
    if file_filter:
        p = Path(file_filter)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.is_file():
            yield p
            return
        yield from sorted(REPO_ROOT.glob(file_filter))
        return
    yield from sorted(root.rglob("*.md"))
    if JULIA_DOCS_DIR.exists():
        yield from sorted(JULIA_DOCS_DIR.glob("*.md"))
    for vdir in VIGNETTES_DIRS:
        yield from sorted(vdir.glob("*.Rmd"))
    for rust_dir in RUST_CRATE_DOCS_DIRS:
        if rust_dir.exists():
            yield from sorted(rust_dir.glob("*.md"))
    if NODEJS_BINDING_DOCS_DIR.exists():
        yield from sorted(NODEJS_BINDING_DOCS_DIR.glob("*.md"))
    if WASM_BINDING_DOCS_DIR.exists():
        yield from sorted(WASM_BINDING_DOCS_DIR.glob("*.md"))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lang",
        choices=["python", "julia", "nodejs", "r", "wasm", "rust", "cpp", "all"],
        default="all",
        help="Which language runner to use (default: all)",
    )
    parser.add_argument(
        "--file",
        metavar="PATH_OR_GLOB",
        help="Restrict to a specific file or glob (relative to repo root)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List snippets without running them"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print snippet source and full output on failure",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop after the first failing snippet",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-snippet timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--output", metavar="FILE", help="Write JSON report to this file"
    )
    args = parser.parse_args(argv)

    active_runners: set[str] = (
        set(RUNNERS.keys()) if args.lang == "all" else {args.lang}
    )

    _python_runner.PYTHON_BIN = _find_python_with_fastlowess()

    # ---- Collect snippets ---------------------------------------------------
    snippets: list[Snippet] = []
    for md in iter_md_files(DOCS_DIR, args.file):
        snippets.extend(extract_snippets(md))

    total_found = len(snippets)

    runnable: list[tuple[Snippet, str]] = []
    for s in snippets:
        r = s.runner
        if r is None or r not in active_runners:
            continue
        reason = should_skip(s, r)
        if reason:
            continue
        runnable.append((s, r))

    print(bold("\nfastLowess doc snippet verifier"))
    print(f"Docs dir : {DOCS_DIR}")
    print(f"Julia docs: {JULIA_DOCS_DIR}")
    print(f"Runners  : {', '.join(sorted(active_runners))}")
    print(f"Snippets : {len(runnable)} runnable / {total_found} total")
    if args.dry_run:
        print()
        for s, r in runnable:
            print(f"  {cyan(r):20s}  {s.label}")
        print()
        return 0
    print()

    # ---- Run snippets (parallel per language) --------------------------------
    results: list[RunResult] = []
    n_pass = n_fail = n_skip = 0
    _print_lock = threading.Lock()

    _by_runner: dict[str, list[tuple[Snippet, str]]] = defaultdict(list)
    for _s, _r in runnable:
        _by_runner[_r].append((_s, _r))

    def _run_language(lang_items: list[tuple[Snippet, str]]) -> list[RunResult]:
        lang_results: list[RunResult] = []
        for s, runner in lang_items:
            label = s.label
            run_fn = RUNNERS.get(runner)
            if run_fn is None:
                res = RunResult(
                    snippet=s,
                    runner=runner,
                    skipped=True,
                    skip_reason="no runner implementation",
                )
                with _print_lock:
                    print(
                        f"  {cyan(runner):20s}  {label} … {yellow('SKIP (no runner)')}"
                    )
            else:
                res = run_fn(s, args.timeout)
                with _print_lock:
                    sys.stdout.write(f"  {cyan(runner):20s}  {label} … ")
                    if res.skipped:
                        print(yellow(f"SKIP ({res.skip_reason})"))
                    elif res.passed:
                        print(green(f"PASS ({res.duration:.2f}s)"))
                    else:
                        print(red(f"FAIL ({res.duration:.2f}s, exit {res.returncode})"))
                        if args.verbose:
                            _print_failure(s, res)

            lang_results.append(res)
            if args.stop_on_fail and not res.skipped and not res.passed:
                with _print_lock:
                    print(
                        red(
                            f"\n[{runner}] Stopped after first failure (--stop-on-fail)."
                        )
                    )
                break
        return lang_results

    n_workers = len(_by_runner) or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_run_language, items) for items in _by_runner.values()
        ]
        for future in concurrent.futures.as_completed(futures):
            for res in future.result():
                results.append(res)
                if res.skipped:
                    n_skip += 1
                elif res.passed:
                    n_pass += 1
                else:
                    n_fail += 1

    # ---- Summary ------------------------------------------------------------
    print()
    print("-" * 60)
    print(bold("Summary"))
    print(
        f"  {green(f'PASS: {n_pass}'):30s}  {yellow(f'SKIP: {n_skip}'):30s}  {red(f'FAIL: {n_fail}')}"
    )
    print()

    failures = [r for r in results if not r.passed and not r.skipped]
    if failures and not args.verbose:
        print(bold("Failed snippets:"))
        for r in failures:
            print(f"  {red('FAIL')} {r.snippet.label}")
            diag = (r.stderr or r.stdout).strip()
            if diag:
                for line in diag.splitlines()[:5]:
                    print(f"      {line}")
        print()

    # ---- JSON output --------------------------------------------------------
    if args.output:
        report = {
            "summary": {"pass": n_pass, "fail": n_fail, "skip": n_skip},
            "snippets": [
                {
                    "file": str(r.snippet.file.relative_to(REPO_ROOT)),
                    "line": r.snippet.line,
                    "lang": r.snippet.lang_tag,
                    "tab": r.snippet.tab,
                    "runner": r.runner,
                    "status": "skip" if r.skipped else ("pass" if r.passed else "fail"),
                    "skip_reason": r.skip_reason if r.skipped else None,
                    "returncode": r.returncode if not r.skipped else None,
                    "duration": round(r.duration, 3),
                    "stderr": r.stderr[:2000] if r.stderr else "",
                }
                for r in results
            ],
        }
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.output}")

    return 1 if n_fail > 0 else 0


def _print_failure(snippet: Snippet, res: RunResult) -> None:
    sep = "-" * 56
    print()
    print(f"  {sep}")
    print(f"  {bold('File:')} {snippet.label}")
    if snippet.tab:
        print(f"  {bold('Tab:')}  {snippet.tab}")
    print(f"  {bold('Code:')}")
    for line in snippet.code.splitlines()[:20]:
        print(f"    {line}")
    if len(snippet.code.splitlines()) > 20:
        print(f"    ... ({len(snippet.code.splitlines())} lines total)")
    if res.stderr.strip():
        print(f"  {bold('Stderr:')}")
        for line in res.stderr.strip().splitlines()[-20:]:
            print(f"    {line}")
    if res.stdout.strip():
        print(f"  {bold('Stdout:')}")
        for line in res.stdout.strip().splitlines()[-10:]:
            print(f"    {line}")
    print(f"  {sep}")
    print()


if __name__ == "__main__":
    sys.exit(main())
