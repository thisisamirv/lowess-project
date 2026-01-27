import json
from pathlib import Path
from statistics import mean, median

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

def load_all_data(output_dir: Path):
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
            # Flatten category structure: {category: [entries]} -> {name: entry}
            flat = {}
            for cat, entries in loaded.items():
                for entry in entries:
                    name = entry.get("name")
                    if name:
                        flat[name] = entry
            data[label] = flat
    return data

def main():
    repo_root = Path(__file__).resolve().parent
    # walk up to workspace root
    workspace = repo_root
    for _ in range(6):
        if (workspace / "output").exists():
            break
        if workspace.parent == workspace:
            break
        workspace = workspace.parent
    out_dir = workspace / "output"
    
    data = load_all_data(out_dir)
    
    # Check what we have
    if not data:
        print("No benchmark results found in benchmarks/output/")
        return

    # Collect all benchmark names
    all_names = set()
    for d in data.values():
        all_names.update(d.keys())

    # Sort names intelligently (maybe separate large scale if needed, but for now just sort)
    sorted_names = sorted(all_names)
    
    candidates = ["Rust (Serial)", "Rust (CPU)", "Rust (GPU)"]
    baseline_label = "R"
    
    # Get baseline data (R) if strictly available
    base_data = data.get(baseline_label, {})
    
    # Table Header
    # Name | R (Base) | Rust (Serial) | Rust (CPU) | Rust (GPU) |
    print(f"{'Name':<20} | {baseline_label:^8} | {'Rust (Serial)':^15} | {'Rust (Parallel)':^15} | {'Rust (GPU)':^15} |")
    print("-" * 105)

    for name in sorted_names:
        # Determine baseline value
        base_entry = base_data.get(name)
        base_val = None
        base_str = "-"
        
        if base_entry:
            base_val, _ = pick_time_value(base_entry)
            if base_val and base_val > 0:
                base_str = f"{base_val:.2f}ms"

        row_str = f"{name:<20} | {base_str:^8} |"

        # Candidates
        for cand in candidates:
            cand_data = data.get(cand, {})
            cand_entry = cand_data.get(name)
            
            c_val = None
            speedup = None
            disp = "-"
            
            if cand_entry:
                c_val, _ = pick_time_value(cand_entry)
                if c_val and c_val > 0:
                     if base_val and base_val > 0:
                         speedup = base_val / c_val
                         disp = f"{c_val:.2f}ms ({speedup:.1f}x)"
                     else:
                         disp = f"{c_val:.2f}ms"
            
            row_str += f" {disp:^15} |"

        print(row_str)

    print("-" * 105)
    print("Format: Time_ms (Speedup vs R). 'x' denotes how many times faster than R.")

if __name__ == "__main__":
    main()
