#!/usr/bin/env python3
"""Combined visualization entrypoint for fastLowess examples."""

import os
import sys

try:
    from .plot_common import OUTPUT_DIR
    from .plot_foundation import (
        plot_degree_comparison,
        plot_fast_lowess_concept,
        plot_fraction_comparison,
        plot_intervals_comparison,
        plot_multivariate_fast_lowess,
        plot_robustness_comparison,
    )
    from .plot_modeling import (
        plot_boundary_policy_comparison,
        plot_cv_comparison,
        plot_degree_interpolation_comparison,
        plot_gap_handling,
        plot_higher_degree_comparison,
        plot_kernel_comparison,
        plot_robust_method_comparison,
        plot_scaling_comparison,
        plot_surface_mode_comparison,
        plot_zero_weight_comparison,
    )
    from .plot_streaming import (
        plot_auto_converge_comparison,
        plot_online_comparison,
        plot_streaming_comparison,
    )
except ImportError:
    from plot_common import OUTPUT_DIR  # type: ignore
    from plot_foundation import (  # type: ignore
        plot_degree_comparison,
        plot_fast_lowess_concept,
        plot_fraction_comparison,
        plot_intervals_comparison,
        plot_multivariate_fast_lowess,
        plot_robustness_comparison,
    )
    from plot_modeling import (  # type: ignore
        plot_boundary_policy_comparison,
        plot_cv_comparison,
        plot_degree_interpolation_comparison,
        plot_gap_handling,
        plot_higher_degree_comparison,
        plot_kernel_comparison,
        plot_robust_method_comparison,
        plot_scaling_comparison,
        plot_surface_mode_comparison,
        plot_zero_weight_comparison,
    )
    from plot_streaming import (  # type: ignore
        plot_auto_converge_comparison,
        plot_online_comparison,
        plot_streaming_comparison,
    )


def build_plot_targets():
    """Return the stable CLI mapping from target names to plot functions."""
    return {
        "degree": plot_degree_comparison,
        "fraction": plot_fraction_comparison,
        "intervals": plot_intervals_comparison,
        "robustness": plot_robustness_comparison,
        "concept": plot_fast_lowess_concept,
        "multivariate": plot_multivariate_fast_lowess,
        "kernel": plot_kernel_comparison,
        "robust_method": plot_robust_method_comparison,
        "boundary": plot_boundary_policy_comparison,
        "higher_degree": plot_higher_degree_comparison,
        "gap": plot_gap_handling,
        "cv": plot_cv_comparison,
        "surface": plot_surface_mode_comparison,
        "scaling": plot_scaling_comparison,
        "zero_weight": plot_zero_weight_comparison,
        "degree_interp": plot_degree_interpolation_comparison,
        "streaming": plot_streaming_comparison,
        "online": plot_online_comparison,
        "auto_converge": plot_auto_converge_comparison,
    }


def main():
    """Dispatch to one plot target or generate all figures."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_targets = build_plot_targets()
    target_name = sys.argv[1] if len(sys.argv) >= 2 else "all"

    if target_name == "all":
        for plot_target in plot_targets.values():
            plot_target()
        return 0

    plot_target = plot_targets.get(target_name)
    if plot_target is None:
        print(f"Unknown target: {target_name}")
        print("Usage: python3 plot.py [target] (default: all)")
        print(
            "Targets: all, degree, fraction, intervals, robustness, concept, multivariate, "
            "kernel, robust_method, boundary, higher_degree, gap, cv, surface, scaling, "
            "zero_weight, degree_interp, streaming, online, auto_converge"
        )
        return 1

    plot_target()
    return 0


if __name__ == "__main__":
    sys.exit(main())
