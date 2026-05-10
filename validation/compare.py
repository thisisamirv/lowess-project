#!/usr/bin/env python3
"""
Compare fastLowess validation results against R's lowess implementation.
R is the reference implementation (original Cleveland algorithm).
"""

import json
import sys
from pathlib import Path

import numpy as np


def compare_fitted_values(r_data, fast_lowess_data):
    """Return max difference and RMSE for two fitted-value sequences."""
    r_fitted = np.array(r_data["result"]["fitted"])
    fast_lowess_fitted = np.array(fast_lowess_data["result"]["fitted"])

    diff = np.abs(r_fitted - fast_lowess_fitted)
    max_diff = np.max(diff)
    rmse = np.sqrt(np.mean(diff**2))
    return max_diff, rmse


def classify_match(max_diff):
    """Map a maximum absolute difference to a validation status label."""
    if max_diff < 1e-10:
        return "EXACT MATCH"
    if max_diff < 1e-6:
        return "MATCH"
    if max_diff < 1e-3:
        return "CLOSE"
    return "MISMATCH"


def compare_implementations():
    """Compare JSON outputs from R and fastLowess for each validation scenario."""
    r_dir = Path("output/r")
    fast_lowess_dir = Path("output/fastLowess")

    if not r_dir.exists():
        print("Error: R output directory not found. Run R/validate.R first.")
        return False

    if not fast_lowess_dir.exists():
        print(
            "Error: fastLowess output directory not found. Run fastLowess/validate first."
        )
        return False

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
        fast_lowess_file = fast_lowess_dir / f"{scenario}.json"

        if not fast_lowess_file.exists():
            print(f"{scenario:<30} | {'MISSING':<15} | {'-':<15} | {'-':<15}")
            continue

        # Load data
        with open(r_file, encoding="utf-8") as file_handle:
            r_data = json.load(file_handle)
        with open(fast_lowess_file, encoding="utf-8") as file_handle:
            fast_lowess_data = json.load(file_handle)

        max_diff, rmse = compare_fitted_values(r_data, fast_lowess_data)
        status = classify_match(max_diff)

        if status == "MISMATCH":
            mismatches.append(scenario)
        else:
            matches.append(scenario)

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
    sys.exit(0 if compare_implementations() else 1)
