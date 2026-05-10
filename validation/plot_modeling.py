#!/usr/bin/env python3
"""Validation plots for kernels, boundaries, and model-shape comparisons."""

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .plot_common import (
        check_file,
        get_input_path,
        plot_noisy_points,
        plot_true_signal,
        save_figure,
    )
except ImportError:
    from plot_common import (  # type: ignore
        check_file,
        get_input_path,
        plot_noisy_points,
        plot_true_signal,
        save_figure,
    )


def plot_kernel_comparison():
    """Plot the impact of different kernel weight functions."""
    if not check_file("kernel_comparison.csv"):
        return
    print("Plotting Kernel Comparison...")

    df = pd.read_csv(get_input_path("kernel_comparison.csv"))
    figure, ax = plt.subplots(figsize=(10, 6))

    plot_noisy_points(
        ax, df["x"], df["y_noisy"], markersize=3, alpha=0.3, label="Noisy Data"
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1.5, alpha=0.5)

    configs = [
        ("y_tricube", "Tricube", "-", "#3b82f6", 3.0),
        ("y_gaussian", "Gaussian", "--", "#ef4444", 2.0),
        ("y_uniform", "Uniform", ":", "#10b981", 2.0),
        ("y_cosine", "Cosine", "-", "#f59e0b", 2.0),
        ("y_epanechnikov", "Epanechnikov", "--", "#8b5cf6", 2.0),
        ("y_biweight", "Biweight", ":", "#06b6d4", 2.0),
        ("y_triangle", "Triangle", "-", "#ec4899", 2.0),
    ]

    for col, name, style, color, lw in configs:
        if col in df.columns:
            ax.plot(df["x"], df[col], linestyle=style, lw=lw, label=name, color=color)

    ax.set_title(
        "Impact of Different Kernel (Weight) Functions",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    save_figure(figure, "kernel_comparison.svg")


def plot_robust_method_comparison():
    """Compare alternative robust weighting methods."""
    if not check_file("robust_method_comparison.csv"):
        return
    print("Plotting Robust Method Comparison...")

    df = pd.read_csv(get_input_path("robust_method_comparison.csv"))
    figure, ax = plt.subplots(figsize=(12, 7))

    plot_noisy_points(
        ax,
        df["x"],
        df["y_noisy"],
        markersize=4,
        alpha=0.4,
        label="Data with Outliers",
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.5)
    ax.plot(
        df["x"],
        df["y_bisquare"],
        "-",
        lw=2.5,
        label="Bisquare (Aggressive)",
        color="#3b82f6",
    )
    ax.plot(
        df["x"], df["y_huber"], "--", lw=2, label="Huber (Balanced)", color="#ef4444"
    )
    ax.plot(
        df["x"],
        df["y_talwar"],
        ":",
        lw=2,
        label="Talwar (Hard Cutoff)",
        color="#10b981",
    )

    ax.set_title(
        "Outlier Resistance by Robustness Method", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(figure, "robust_method_comparison.svg")


def plot_boundary_policy_comparison():
    """Compare boundary handling policies near the ends of the domain."""
    if not check_file("boundary_comparison.csv"):
        return
    print("Plotting Boundary Policy Comparison...")

    df = pd.read_csv(get_input_path("boundary_comparison.csv"))
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax in [ax1, ax2]:
        plot_noisy_points(
            ax, df["x"], df["y_noisy"], markersize=3, alpha=0.3, label="Noisy Data"
        )
        plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.5)
        ax.plot(df["x"], df["y_none"], "-", lw=2, label="NoBoundary", color="#ef4444")
        ax.plot(df["x"], df["y_extend"], "--", lw=2, label="Extend", color="#3b82f6")
        ax.plot(df["x"], df["y_reflect"], "-.", lw=2, label="Reflect", color="#10b981")
        ax.grid(True, alpha=0.3)

    ax1.set_xlim(-0.05, 0.2)
    ax1.set_ylim(0.8, 2.0)
    ax1.set_title("Left Boundary Impact")
    ax2.set_xlim(0.8, 1.05)
    ax2.set_ylim(10, 22)
    ax2.set_title("Right Boundary Impact")

    figure.suptitle(
        "Mitigating Boundary Bias with Reflection and Extension",
        fontsize=16,
        fontweight="bold",
    )
    ax2.legend(loc="lower right")

    save_figure(figure, "boundary_comparison.svg")


def plot_higher_degree_comparison():
    """Plot increasingly higher-order local polynomial fits."""
    if not check_file("higher_degree_comparison.csv"):
        return
    print("Plotting Higher Degree Comparison...")

    df = pd.read_csv(get_input_path("higher_degree_comparison.csv"))
    figure, ax = plt.subplots(figsize=(10, 6))

    plot_noisy_points(
        ax, df["x"], df["y_noisy"], markersize=3, alpha=0.3, label="Noisy Data"
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.5)
    ax.plot(
        df["x"],
        df["y_quadratic"],
        "-",
        lw=2,
        label="Quadratic (Degree 2)",
        color="#ef4444",
    )
    ax.plot(
        df["x"], df["y_cubic"], "--", lw=2.5, label="Cubic (Degree 3)", color="#3b82f6"
    )
    ax.plot(
        df["x"],
        df["y_quartic"],
        ":",
        lw=2.5,
        label="Quartic (Degree 4)",
        color="#10b981",
    )

    ax.set_title(
        "Handling Complex Curvature with Higher Degrees",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(figure, "higher_degree_comparison.svg")


def plot_gap_handling():
    """Plot smoothing across a region with missing observations."""
    if not check_file("gap_handling.csv"):
        return
    print("Plotting Gap Handling...")

    df = pd.read_csv(get_input_path("gap_handling.csv"))
    figure, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["x"],
        df["y_noisy"],
        "o",
        markersize=4,
        alpha=0.6,
        color="#3b82f6",
        label="Available Data",
    )
    ax.plot(df["x"], df["y_smooth"], "r-", lw=3, label="fastLowess Interpolation")

    ax.set_title(
        "fastLowess Gap Handling (Bridging Missing Regions)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axvspan(4.0, 7.0, color="gray", alpha=0.1, label="Missing Data Region")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(figure, "gap_handling.svg")


def plot_cv_comparison():
    """Compare cross-validation score curves and their resulting fits."""
    if not check_file("cv_scores.csv") or not check_file("cv_fits.csv"):
        return
    print("Plotting Cross-Validation Comparison...")

    df_scores = pd.read_csv(get_input_path("cv_scores.csv"))
    df_fits = pd.read_csv(get_input_path("cv_fits.csv"))
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(
        df_scores["fraction"],
        df_scores["loocv_rmse"],
        "o-",
        label="LOOCV RMSE",
        color="#3b82f6",
    )
    ax1.plot(
        df_scores["fraction"],
        df_scores["kfold_rmse"],
        "s--",
        label="5-Fold RMSE",
        color="#ef4444",
    )

    best_loocv_fraction = df_scores.loc[df_scores["loocv_rmse"].idxmin(), "fraction"]
    best_kfold_fraction = df_scores.loc[df_scores["kfold_rmse"].idxmin(), "fraction"]

    ax1.axvline(best_loocv_fraction, color="#3b82f6", alpha=0.3, linestyle="-")
    ax1.axvline(best_kfold_fraction, color="#ef4444", alpha=0.3, linestyle="--")
    ax1.set_title(
        "Bandwidth Selection: CV Score vs Fraction", fontsize=14, fontweight="bold"
    )
    ax1.set_xlabel("Smoothing Fraction")
    ax1.set_ylabel("RMSE")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plot_noisy_points(
        ax2,
        df_fits["x"],
        df_fits["y_noisy"],
        markersize=3,
        alpha=0.3,
        label="Noisy Data",
    )
    plot_true_signal(ax2, df_fits["x"], df_fits["y_true"], linewidth=1.5, alpha=0.5)
    ax2.plot(
        df_fits["x"],
        df_fits["y_loocv"],
        "-",
        lw=2.5,
        label=f"LOOCV (f={best_loocv_fraction})",
        color="#3b82f6",
    )
    ax2.plot(
        df_fits["x"],
        df_fits["y_kfold"],
        "--",
        lw=2,
        label=f"5-Fold (f={best_kfold_fraction})",
        color="#ef4444",
    )
    ax2.plot(
        df_fits["x"],
        df_fits["y_fixed"],
        ":",
        lw=2,
        label="No CV (f=0.1, Overfit)",
        color="#10b981",
    )

    ax2.set_title(
        "Impact of Bandwidth Selection on Fit", fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    figure.tight_layout()
    save_figure(figure, "cv_comparison.svg")


def plot_surface_mode_comparison():
    """Compare direct and interpolation surface evaluation modes."""
    if not check_file("surface_mode_comparison.csv"):
        return
    print("Plotting Surface Mode Comparison...")

    df = pd.read_csv(get_input_path("surface_mode_comparison.csv"))
    figure, ax = plt.subplots(figsize=(10, 6))

    plot_noisy_points(
        ax, df["x"], df["y_noisy"], markersize=3, alpha=0.3, label="Noisy Data"
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.5)
    ax.plot(df["x"], df["y_direct"], "-", lw=2.5, label="Direct Mode", color="#3b82f6")
    ax.plot(
        df["x"],
        df["y_interpolation"],
        "--",
        lw=2,
        label="Interpolation Mode",
        color="#ef4444",
    )

    ax.set_title("Direct vs Interpolation Evaluation", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(figure, "surface_comparison.svg")


def plot_scaling_comparison():
    """Compare robust residual scaling strategies."""
    if not check_file("scaling_comparison.csv"):
        return
    print("Plotting Scaling Method Comparison...")

    df = pd.read_csv(get_input_path("scaling_comparison.csv"))
    figure, ax = plt.subplots(figsize=(10, 6))

    plot_noisy_points(
        ax,
        df["x"],
        df["y_noisy"],
        markersize=4,
        alpha=0.4,
        label="Data with Outliers",
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.5)
    ax.plot(
        df["x"],
        df["y_mad"],
        "-",
        lw=2.5,
        label="MAD Scaling (Median Abs Dev)",
        color="#3b82f6",
    )
    ax.plot(
        df["x"],
        df["y_mar"],
        "--",
        lw=2.5,
        label="MAR Scaling (Median Abs Residual)",
        color="#ef4444",
    )

    ax.set_title("Robust Residual Scaling: MAD vs MAR", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(figure, "scaling_comparison.svg")


def plot_zero_weight_comparison():
    """Plot fallback behavior when all robustness weights drop to zero."""
    if not check_file("zero_weight_comparison.csv"):
        return
    print("Plotting Zero Weight Fallback Comparison...")

    df = pd.read_csv(get_input_path("zero_weight_comparison.csv"))
    figure, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["x"],
        df["y_noisy"],
        "o",
        markersize=4,
        alpha=0.4,
        color="gray",
        label="Data with Extreme Outlier",
    )
    ax.plot(
        df["x"],
        df["y_mean"],
        "-",
        lw=2.5,
        label="UseLocalMean Fallback",
        color="#3b82f6",
    )
    ax.plot(
        df["x"],
        df["y_original"],
        "--",
        lw=2.5,
        label="ReturnOriginal Fallback",
        color="#ef4444",
    )

    ann_x = df.iloc[50]["x"]
    ann_y = df.iloc[50]["y_noisy"]
    ax.annotate(
        "Extreme Outlier (Zero-Weighted)",
        xy=(ann_x, ann_y),
        xytext=(ann_x + 1, ann_y - 20),
        arrowprops={"facecolor": "black", "shrink": 0.05, "width": 1, "headwidth": 5},
    )
    ax.set_title("Zero-Weight Fallback Policies", fontsize=14, fontweight="bold")
    ax.set_ylim(-2, 110)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(figure, "zero_weight_comparison.svg")


def plot_degree_interpolation_comparison():
    """Compare direct and interpolated evaluation across polynomial degrees."""
    if not check_file("degree_interpolation_comparison.csv"):
        return
    print("Plotting Degree Interpolation Comparison...")

    df = pd.read_csv(get_input_path("degree_interpolation_comparison.csv"))
    figure, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    configs = [
        ("Linear", "y_lin_direct", "y_lin_interp", "#3b82f6", "#93c5fd"),
        ("Quadratic", "y_quad_direct", "y_quad_interp", "#ef4444", "#fca5a5"),
        ("Cubic", "y_cubic_direct", "y_cubic_interp", "#8b5cf6", "#c4b5fd"),
        ("Quartic", "y_quartic_direct", "y_quartic_interp", "#10b981", "#6ee7b7"),
    ]

    for ax, (name, direct_col, interp_col, direct_color, interp_color) in zip(
        axes, configs
    ):
        plot_noisy_points(
            ax, df["x"], df["y_noisy"], markersize=3, alpha=0.3, label=None
        )
        plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.5, label=None)
        ax.plot(
            df["x"],
            df[direct_col],
            "-",
            lw=2.5,
            label=f"{name} Direct",
            color=direct_color,
        )
        ax.plot(
            df["x"],
            df[interp_col],
            "--",
            lw=2.5,
            label=f"{name} Interp",
            color=interp_color,
        )
        ax.set_title(f"{name} Degree", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    figure.suptitle(
        "Surface Evaluation Fidelity across polynomial Degrees: Direct vs Interpolation",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_figure(figure, "degree_interpolation_comparison.svg")
