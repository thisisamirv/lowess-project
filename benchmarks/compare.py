"""Compare exported benchmark timings across R and Rust backends."""

import json
from pathlib import Path


def load_json(path: Path):
    """Load a JSON file when it exists, otherwise return ``None``."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def pick_time_value(entry: dict):
    """Robustly pick a numeric timing from an entry."""
    for key in ("mean_time_ms", "median_time_ms", "max_time_ms"):
        if key in entry:
            try:
                return float(entry[key]), entry.get("size")
            except (TypeError, ValueError):
                pass

    for k, v in entry.items():
        if isinstance(v, (int, float)):
            if k in ("iterations", "size", "runs"):
                continue
            try:
                return float(v), entry.get("size")
            except (TypeError, ValueError):
                pass
    return None, entry.get("size")


def load_all_data(output_dir: Path):
    """Load all exported benchmark JSON files keyed by backend label."""
    files = {
        "Rust (CPU)": output_dir / "rust_benchmark_cpu.json",
        "Rust (Serial)": output_dir / "rust_benchmark_cpu_serial.json",
        "Rust (GPU)": output_dir / "rust_benchmark_gpu.json",
        "R": output_dir / "r_benchmark.json",
    }

    data = {}
    for label, path in files.items():
        loaded = load_json(path)
        if loaded:
            flat = {}
            for entries in loaded.values():
                for entry in entries:
                    name = entry.get("name")
                    if name:
                        flat[name] = entry
            data[label] = flat
    return data


def find_workspace_output_dir() -> Path:
    """Find the workspace ``output`` directory by walking upward from this script."""
    workspace = Path(__file__).resolve().parent
    for _ in range(6):
        if (workspace / "output").exists():
            break
        if workspace.parent == workspace:
            break
        workspace = workspace.parent
    return workspace / "output"


def collect_benchmark_names(data: dict) -> list[str]:
    """Return all benchmark names seen across backends in sorted order."""
    all_names = set()
    for backend_data in data.values():
        all_names.update(backend_data.keys())
    return sorted(all_names)


def format_baseline(entry: dict | None) -> tuple[float | None, str]:
    """Return the numeric baseline value and its display string."""
    if not entry:
        return None, "-"

    baseline_value, _ = pick_time_value(entry)
    if baseline_value and baseline_value > 0:
        return baseline_value, f"{baseline_value:.2f}ms"
    return None, "-"


def format_candidate_cell(entry: dict | None, baseline_value: float | None) -> str:
    """Format one backend cell, optionally including speedup versus baseline."""
    if not entry:
        return "-"

    candidate_value, _ = pick_time_value(entry)
    if not candidate_value or candidate_value <= 0:
        return "-"

    if baseline_value and baseline_value > 0:
        speedup = baseline_value / candidate_value
        return f"{candidate_value:.2f}ms ({speedup:.1f}x)"
    return f"{candidate_value:.2f}ms"


def print_comparison_table(data: dict) -> None:
    """Print the comparison table for all benchmark names across backends."""
    candidates = ["Rust (Serial)", "Rust (CPU)", "Rust (GPU)"]
    baseline_label = "R"
    base_data = data.get(baseline_label, {})

    print(
        f"{'Name':<20} | {baseline_label:^8} | {'Rust (Serial)':^15} | "
        f"{'Rust (Parallel)':^15} | {'Rust (GPU)':^15} |"
    )
    print("-" * 105)

    for name in collect_benchmark_names(data):
        baseline_value, baseline_display = format_baseline(base_data.get(name))
        row_cells = [f"{name:<20} | {baseline_display:^8} |"]
        for cand in candidates:
            display = format_candidate_cell(
                data.get(cand, {}).get(name), baseline_value
            )
            row_cells.append(f" {display:^15} |")
        print("".join(row_cells))

    print("-" * 105)
    print("Format: Time_ms (Speedup vs R). 'x' denotes how many times faster than R.")


def main():
    """Print a comparison table for benchmark outputs found under ``output``."""
    data = load_all_data(find_workspace_output_dir())

    if not data:
        print("No benchmark results found in benchmarks/output/")
        return

    print_comparison_table(data)


if __name__ == "__main__":
    main()
