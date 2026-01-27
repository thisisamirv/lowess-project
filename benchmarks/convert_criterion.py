#!/usr/bin/env python3
"""
Convert Criterion benchmark results to JSON format for comparison.

Criterion stores results in:
- target/criterion/<group>/<bench_id>/<param>/new/estimates.json (parameterized)
- target/criterion/<group>/<bench_name>/new/estimates.json (non-parameterized)

This script extracts timing data and writes it to:
- benchmarks/output/rust_benchmark_cpu.json (parallel)
- benchmarks/output/rust_benchmark_cpu_serial.json (serial)

Usage: python3 convert_criterion.py
"""

import json
from pathlib import Path
from typing import Dict, List


def parse_estimates(estimates_file: Path) -> dict | None:
    """Parse a criterion estimates.json file and return timing in ms."""
    try:
        with open(estimates_file) as f:
            estimates = json.load(f)
        
        # Criterion stores in nanoseconds
        mean_ns = estimates.get("mean", {}).get("point_estimate", 0)
        std_ns = estimates.get("std_dev", {}).get("point_estimate", 0)
        median_ns = estimates.get("median", {}).get("point_estimate", 0)
        
        # Try to get min/max from sample data
        sample_file = estimates_file.parent / "sample.json"
        min_ns = mean_ns
        max_ns = mean_ns
        if sample_file.exists():
            try:
                with open(sample_file) as f:
                    sample = json.load(f)
                times = sample.get("times", [])
                if times:
                    min_ns = min(times)
                    max_ns = max(times)
            except Exception:
                pass
        
        return {
            "mean_time_ms": mean_ns / 1_000_000,
            "std_time_ms": std_ns / 1_000_000,
            "median_time_ms": median_ns / 1_000_000,
            "min_time_ms": min_ns / 1_000_000,
            "max_time_ms": max_ns / 1_000_000,
        }
    except Exception as e:
        print(f"Error parsing {estimates_file}: {e}")
        return None


def find_criterion_results(criterion_dir: Path) -> Dict[str, List[dict]]:
    """Find all criterion benchmark results and organize by category."""
    results: Dict[str, List[dict]] = {}
    
    if not criterion_dir.exists():
        print(f"Criterion directory not found: {criterion_dir}")
        return results
    
    # Walk through criterion output structure
    for group_dir in criterion_dir.iterdir():
        if not group_dir.is_dir() or group_dir.name == "report":
            continue
        
        category = group_dir.name
        
        # Strip suffix for logical checking, but keep original category for grouping
        clean_category = category
        if category.endswith("_parallel"):
            clean_category = category[:-9]
        elif category.endswith("_serial"):
            clean_category = category[:-7]
        elif category.endswith("_gpu"):
            clean_category = category[:-4]

        if category not in results:
            results[category] = []
        
        for bench_dir in group_dir.iterdir():
            if not bench_dir.is_dir() or bench_dir.name == "report":
                continue
            
            bench_id = bench_dir.name
            
            # Check if this has 'new/' directly (non-parameterized)
            new_dir = bench_dir / "new"
            if new_dir.exists() and (new_dir / "estimates.json").exists():
                # Non-parameterized
                timing = parse_estimates(new_dir / "estimates.json")
                if timing:
                    result = {
                        "name": bench_id,
                        "size": 5000,
                        "iterations": 10,
                        **timing
                    }
                    results[category].append(result)
            else:
                # Parameterized benchmark
                for param_dir in bench_dir.iterdir():
                    if not param_dir.is_dir() or param_dir.name in ("report", "new", "base", "change"):
                        continue
                    
                    param = param_dir.name
                    estimates_file = param_dir / "new" / "estimates.json"
                    
                    if estimates_file.exists():
                        timing = parse_estimates(estimates_file)
                        if timing:
                            # Try to parse param as size
                            try:
                                size = int(param)
                            except ValueError:
                                size = 0
                            
                            # Create aligned name using clean_category
                            if clean_category == "scalability":
                                name = f"scale_{param}"
                            elif clean_category in ("financial", "scientific", "genomic", "fraction", "iterations"):
                                name = f"{clean_category}_{param}"
                            elif clean_category in ("polynomial_degrees", "distance_metrics"):
                                name = f"{bench_id}_{param}"
                            else:
                                name = f"{bench_id}_{param}"
                            
                            result = {
                                "name": name,
                                "size": size,
                                "iterations": 10,
                                **timing
                            }
                            results[category].append(result)
    
    # Sort results within each category by name
    for category in results:
        results[category].sort(key=lambda x: x["name"])
    
    return results


def main():
    script_dir = Path(__file__).resolve().parent
    criterion_dir = script_dir.parent / "target" / "criterion"
    output_dir = script_dir / "output"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading criterion results from: {criterion_dir}")
    
    results = find_criterion_results(criterion_dir)
    
    if not results:
        print("No criterion results found. Run 'cargo bench' first.")
        return 1
    
    # Separate results into CPU (Parallel), CPU Serial, and GPU
    cpu_results = {}
    cpu_serial_results = {}
    gpu_results = {}
    
    for category, benchmarks in results.items():
        # Determine backend from suffix and map to correct dict
        if category.endswith("_serial"):
            clean_category = category[:-7]
            target_dict = cpu_serial_results
        elif category.endswith("_parallel"):
            clean_category = category[:-9]
            target_dict = cpu_results
        elif category.endswith("_gpu"):
            clean_category = category[:-4]
            target_dict = gpu_results
        else:
            # Fallback for benchmarks without known suffix -> treat as default (parallel/CPU)
            clean_category = category
            target_dict = cpu_results
            
        if clean_category not in target_dict:
            target_dict[clean_category] = []
        target_dict[clean_category].extend(benchmarks)

    # Helper to save results
    def save_results(data, filename):
        if not data:
            return
        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {output_path}")
        
        # Print summary
        total = sum(len(v) for v in data.values())
        print(f"Exported {total} benchmarks across {len(data)} categories:")
        for cat, benches in sorted(data.items()):
            print(f"  {cat}: {len(benches)} benchmarks")
            for b in benches:
                print(f"    - {b['name']}: {b['mean_time_ms']:.3f} ms")

    # Save Results
    if cpu_results:
        print("\n--- CPU (Parallel) Results ---")
        save_results(cpu_results, "rust_benchmark_cpu.json")
        
    if cpu_serial_results:
        print("\n--- CPU Serial Results ---")
        save_results(cpu_serial_results, "rust_benchmark_cpu_serial.json")

    if gpu_results:
        print("\n--- GPU Results ---")
        save_results(gpu_results, "rust_benchmark_gpu.json")
    
    return 0


if __name__ == "__main__":
    exit(main())
