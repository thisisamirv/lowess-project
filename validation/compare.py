#!/usr/bin/env python3
"""
Compare fastLowess validation results against R's lowess implementation.
R is the reference implementation (original Cleveland algorithm).
"""

import json
import numpy as np
from pathlib import Path

def compare_implementations():
    r_dir = Path("output/r")
    fastLowess_dir = Path("output/fastLowess")
    
    if not r_dir.exists():
        print("Error: R output directory not found. Run R/validate.R first.")
        return
    
    if not fastLowess_dir.exists():
        print("Error: fastLowess output directory not found. Run fastLowess/validate first.")
        return
    
    print("=" * 85)
    print("VALIDATION: fastLowess vs R (Reference Implementation)")
    print("=" * 85)
    print()
    print(f"{'SCENARIO':<30} | {'STATUS':<15} | {'MAX DIFF':<15} | {'RMSE':<15}")
    print("-" * 85)
    
    scenarios = sorted([f.stem for f in r_dir.glob("*.json")])
    
    matches = []
    mismatches = []
    
    for scenario in scenarios:
        r_file = r_dir / f"{scenario}.json"
        fastLowess_file = fastLowess_dir / f"{scenario}.json"
        
        if not fastLowess_file.exists():
            print(f"{scenario:<30} | {'MISSING':<15} | {'-':<15} | {'-':<15}")
            continue
        
        # Load data
        with open(r_file) as f:
            r_data = json.load(f)
        with open(fastLowess_file) as f:
            fastLowess_data = json.load(f)
        
        # Compare fitted values
        r_fitted = np.array(r_data['result']['fitted'])
        fastLowess_fitted = np.array(fastLowess_data['result']['fitted'])
        
        diff = np.abs(r_fitted - fastLowess_fitted)
        max_diff = np.max(diff)
        rmse = np.sqrt(np.mean(diff**2))
        
        # Determine status
        if max_diff < 1e-10:
            status = "EXACT MATCH"
            matches.append(scenario)
        elif max_diff < 1e-6:
            status = "MATCH"
            matches.append(scenario)
        elif max_diff < 1e-3:
            status = "CLOSE"
            matches.append(scenario)
        else:
            status = "MISMATCH"
            mismatches.append(scenario)
        
        print(f"{scenario:<30} | {status:<15} | {max_diff:<15.2e} | {rmse:<15.2e}")
    
    print("-" * 85)
    print()
    print(f"Summary: {len(matches)} matches, {len(mismatches)} mismatches")
    
    if mismatches:
        print(f"\nFAILURES ({len(mismatches)}):")
        for scenario in mismatches:
            print(f"  - {scenario}")
    else:
        print("\n✓ All scenarios PASS!")
    
    return len(mismatches) == 0

if __name__ == "__main__":
    success = compare_implementations()
    exit(0 if success else 1)
