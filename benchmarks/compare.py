import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_json(p: Path):
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_time_value(entry: dict):
    """Robustly pick a numeric timing from an entry."""
    for key in ("mean_time_ms", "median_time_ms", "max_time_ms"):
        if key in entry:
            try:
                return float(entry[key]), entry.get("size")
            except Exception:
                pass
    # fallback
    for k, v in entry.items():
        if isinstance(v, (int, float)):
            if k in ("iterations", "size", "runs"):
                continue
            try:
                return float(v), entry.get("size")
            except Exception:
                pass
    return None, entry.get("size")


def pick_fitted(entry: dict) -> "list[float] | None":
    """Return the fitted-values array from a benchmark entry, or None."""
    fitted = entry.get("fitted")
    if isinstance(fitted, list) and len(fitted) > 0:
        return fitted
    return None


def load_all_data(output_dir: Path):
    files = {
        "R": output_dir / "r_benchmark.json",
        "rfastlowess (Serial)": output_dir / "rfastlowess_serial.json",
        "rfastlowess (Parallel)": output_dir / "rfastlowess_parallel.json",
    }

    data = {}
    for label, path in files.items():
        loaded = load_json(path)
        if loaded:
            # Flatten category structure: {category: [entries]} -> {name: entry}
            flat = {}
            for cat, entries in loaded.items():
                for entry in entries:
                    name = entry.get("name")
                    if name:
                        flat[name] = entry
            data[label] = flat
    return data


def _category(name: str) -> str:
    """Return the category prefix of a benchmark name (e.g. 'scale', 'fraction')."""
    return re.split(r"[_\-](\d)", name)[0].rstrip("_")


def plot_benchmarks(data: dict, plots_dir: Path) -> None:
    """Generate a 2-column x 5-row grid of category subplots and save as SVG."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    ROWS, COLS = 5, 2

    all_systems = ["R", "rfastlowess (Serial)", "rfastlowess (Parallel)"]
    systems = [s for s in all_systems if any(data.get(s, {}).values())]

    SYS_COLORS = {
        "R": "#4878d0",
        "rfastlowess (Serial)": "#ee854a",
        "rfastlowess (Parallel)": "#6acc65",
    }

    all_names: set[str] = set()
    for d in data.values():
        all_names.update(d.keys())

    categories: dict[str, list[str]] = {}
    for name in sorted(all_names):
        cat = _category(name)
        categories.setdefault(cat, []).append(name)

    cat_list = sorted(categories.keys(), key=lambda c: -len(categories[c]))
    n_cats = len(cat_list)

    bar_h = 0.25
    group_gap = 0.15

    # Height of each row proportional to the number of benchmarks it contains
    row_item_counts = [
        max(
            (
                len(categories[cat_list[row * COLS + col]])
                if row * COLS + col < n_cats
                else 0
            )
            for col in range(COLS)
        )
        for row in range(ROWS)
    ]
    per_item_h = 0.55  # inches per benchmark group
    fig_h = max(8, sum(per_item_h * n + 0.8 for n in row_item_counts) + 1.5)

    fig, axes = plt.subplots(
        ROWS,
        COLS,
        figsize=(14, fig_h),
        gridspec_kw={"height_ratios": row_item_counts},
    )
    fig.suptitle(
        "Benchmark Comparison - Execution Time (log scale)",
        fontsize=13,
        fontweight="bold",
    )

    for ci, cat in enumerate(cat_list):
        ax = axes[ci // COLS, ci % COLS]
        names = categories[cat]
        n = len(names)

        y_pos: list[float] = []
        y = 0.0
        for _ in names:
            y_pos.append(y)
            y += len(systems) * bar_h + group_gap

        for idx, name in enumerate(names):
            ybase = y_pos[idx]
            for si, sys in enumerate(systems):
                val, _ = pick_time_value(data.get(sys, {}).get(name, {}))
                if val is not None and val > 0:
                    ax.barh(
                        ybase + si * bar_h,
                        val,
                        bar_h * 0.85,
                        color=SYS_COLORS[sys],
                        alpha=0.88,
                        zorder=2,
                        label=sys if idx == 0 else "_nolegend_",
                    )

        ytick_pos = [y_pos[i] + (len(systems) - 1) * bar_h / 2 for i in range(n)]
        ytick_lbl = [name[len(cat) :].lstrip("_") or name for name in names]
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_lbl, fontsize=7.5)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(
            ticker.LogLocator(base=10, subs=(1, 2, 5), numticks=12)
        )
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlabel("ms", fontsize=8)
        ax.set_title(cat.capitalize(), fontsize=10, fontweight="bold", style="italic")
        ax.invert_yaxis()
        ax.grid(axis="x", which="both", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ci == 0:
            ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    for ci in range(n_cats, ROWS * COLS):
        axes[ci // COLS, ci % COLS].set_visible(False)

    fig.tight_layout()
    out = plots_dir / "benchmark_comparison.svg"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out.relative_to(plots_dir.parent)}")


def compare_outputs(data: dict, baseline_label: str, candidates: list[str]) -> None:
    """Print a table of max/mean absolute differences in fitted values vs baseline."""
    base_data = data.get(baseline_label, {})
    all_names = sorted({n for d in data.values() for n in d})

    if not any(pick_fitted(base_data.get(n, {})) is not None for n in all_names):
        return  # no fitted data available

    col_w = 25
    header = f"{'Name':<20} | " + " | ".join(f"{c:^{col_w}}" for c in candidates) + " |"
    sep = "-" * len(header)
    print(f"\nOutput comparison (max absolute difference vs {baseline_label}):")
    print(header)
    print(sep)

    for name in all_names:
        base_fitted = pick_fitted(base_data.get(name, {}))
        row = f"{name:<20} |"
        for cand in candidates:
            cand_fitted = pick_fitted(data.get(cand, {}).get(name, {}))
            if base_fitted is None or cand_fitted is None:
                cell = "-"
            elif len(base_fitted) != len(cand_fitted):
                cell = f"len {len(base_fitted)} vs {len(cand_fitted)}"
            else:
                diff = np.abs(np.array(base_fitted) - np.array(cand_fitted))
                cell = f"max={diff.max():.2e} avg={diff.mean():.2e}"
            row += f" {cell:^{col_w}} |"
        print(row)

    print(sep)


def main():
    out_dir = Path(__file__).resolve().parent / "output"

    data = load_all_data(out_dir)

    # Check what we have
    if not data:
        print("No benchmark results found in benchmarks/output/")
        return

    # Collect all benchmark names
    all_names = set()
    for d in data.values():
        all_names.update(d.keys())

    # Sort names intelligently
    sorted_names = sorted(all_names)

    candidates = ["rfastlowess (Serial)", "rfastlowess (Parallel)"]
    baseline_label = "R"

    # Get baseline data (R) if available
    base_data = data.get(baseline_label, {})

    # Table Header
    print(
        f"{'Name':<20} | {baseline_label:^10} | {'rfastlowess (Serial)':^21} | {'rfastlowess (Parallel)':^21} |"
    )
    print("-" * 82)

    for name in sorted_names:
        # Determine baseline value
        base_entry = base_data.get(name)
        base_val = None
        base_str = "-"

        if base_entry:
            base_val, _ = pick_time_value(base_entry)
            if base_val is not None and base_val > 0:
                base_str = f"{base_val:.2f}ms"
            elif base_val is not None:
                base_str = "<1ms"

        row_str = f"{name:<20} | {base_str:^10} |"

        # Candidates
        for cand in candidates:
            cand_data = data.get(cand, {})
            cand_entry = cand_data.get(name)

            c_val = None
            speedup = None
            disp = "-"

            if cand_entry:
                c_val, _ = pick_time_value(cand_entry)
                if c_val is not None and c_val > 0:
                    if base_val is not None and base_val > 0:
                        speedup = base_val / c_val
                        disp = f"{c_val:.2f}ms ({speedup:.1f}x)"
                    else:
                        disp = f"{c_val:.2f}ms"
                elif c_val is not None:  # entry exists but timed below resolution
                    disp = "<1ms"

            row_str += f" {disp:^21} |"

        print(row_str)

    print("-" * 82)
    print("Format: Time_ms (Speedup vs R). 'x' denotes how many times faster than R.")

    compare_outputs(data, baseline_label, candidates)

    # Generate plots
    print(f"\nGenerating plots -> {out_dir.relative_to(out_dir.parent)}/")
    plot_benchmarks(data, out_dir)


if __name__ == "__main__":
    main()
