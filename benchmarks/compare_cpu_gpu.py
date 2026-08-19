"""Analyse rust_benchmark_cpu_vs_gpu.json and print crossover tables.

Usage:
    python compare_cpu_gpu.py [path/to/rust_benchmark_cpu_vs_gpu.json]
"""

import json
import sys
from pathlib import Path

# ─── load ─────────────────────────────────────────────────────────────────────


def load(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["results"]


# ─── group helpers ────────────────────────────────────────────────────────────


def group_by(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


def crossover_n(rows_sorted_by_n: list[dict]) -> int | None:
    """First n where GPU becomes faster (speedup >= 1)."""
    for r in rows_sorted_by_n:
        if r["gpu_speedup"] >= 1.0:
            return r["n"]
    return None


# ─── formatting ───────────────────────────────────────────────────────────────


def fmt_n(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def fmt_ms(v: float) -> str:
    return f"{v:.2f} ms"


def speedup_bar(s: float, width: int = 20) -> str:
    if s >= 1.0:
        filled = min(width, round((s - 1.0) / 4.0 * width))
        return "GPU " + "█" * filled
    else:
        filled = min(width, round((1.0 - s) / 1.0 * width))
        return "CPU " + "▒" * filled


# ─── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("output/rust_benchmark_cpu_vs_gpu.json")
    )
    if not path.exists():
        sys.exit(f"File not found: {path}\nRun:  make bench-cpu-vs-gpu")

    rows = load(path)

    fractions = sorted({r["fraction"] for r in rows})
    iter_counts = sorted({r["robustness_iterations"] for r in rows})

    print("=" * 72)
    print("CPU-parallel vs GPU  —  when does GPU win?")
    print("=" * 72)

    for iters in iter_counts:
        for frac in fractions:
            subset = sorted(
                [
                    r
                    for r in rows
                    if r["fraction"] == frac and r["robustness_iterations"] == iters
                ],
                key=lambda r: r["n"],
            )
            if not subset:
                continue

            xover = crossover_n(subset)
            xover_str = (
                f"n ≥ {fmt_n(xover)}"
                if xover
                else "never (GPU always slower in tested range)"
            )

            print(
                f"\n  fraction={frac}  iterations={iters}  →  GPU wins at {xover_str}"
            )
            print(
                f"  {'n':>8}  {'CPU (med)':>10}  {'GPU (med)':>10}  {'speedup':>8}  chart"
            )
            print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*24}")
            for r in subset:
                marker = "◀ crossover" if r["n"] == xover else ""
                print(
                    f"  {fmt_n(r['n']):>8}  {fmt_ms(r['cpu_median_ms']):>10}"
                    f"  {fmt_ms(r['gpu_median_ms']):>10}  {r['gpu_speedup']:>7.2f}×"
                    f"  {speedup_bar(r['gpu_speedup'])}  {marker}"
                )

    # ── summary table: crossover n for each (fraction × iterations) ──────────
    print("\n" + "=" * 72)
    print("Crossover summary  (first n where GPU median < CPU median)")
    print("=" * 72)
    header = f"  {'fraction':>10}" + "".join(
        f"  {'iter='+str(i):>10}" for i in iter_counts
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for frac in fractions:
        row_str = f"  {frac:>10.2f}"
        for iters in iter_counts:
            subset = sorted(
                [
                    r
                    for r in rows
                    if r["fraction"] == frac and r["robustness_iterations"] == iters
                ],
                key=lambda r: r["n"],
            )
            xover = crossover_n(subset)
            cell = fmt_n(xover) if xover else "never"
            row_str += f"  {cell:>10}"
        print(row_str)


if __name__ == "__main__":
    main()
