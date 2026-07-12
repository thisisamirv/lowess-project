#!/usr/bin/env python3
"""Visualization entrypoint for fastLowess validation."""

import os
import sys
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = "output/visual"
OUTPUT_DIR = "output/fig"

plt.rcParams["svg.fonttype"] = "none"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def check_file(filename):
    """Return whether an input CSV exists, printing a helpful error when it does not."""
    filepath = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Run 'cargo run --bin visual' first.")
        return False
    return True


def get_input_path(filename):
    """Return the absolute path for an input CSV under the visual output directory."""
    return os.path.join(INPUT_DIR, filename)


def get_output_path(filename):
    """Return the absolute path for an output figure under the figure directory."""
    return os.path.join(OUTPUT_DIR, filename)


def save_figure(figure, filename, *, dpi=72, bbox_inches=None):
    """Save a Matplotlib figure as SVG and close it afterwards."""
    output_file = get_output_path(filename)
    save_kwargs = {"format": "svg", "dpi": dpi}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    figure.savefig(output_file, **save_kwargs)
    print(f"Saved to {output_file}")
    plt.close(figure)


def column_to_numpy(dataframe, column_name):
    """Convert a DataFrame column to a NumPy array for plotting helpers."""
    return np.asarray(dataframe[column_name].to_numpy())


def add_rasterized_colorbar(figure, surface, axis):
    """Attach a colorbar and rasterize its solids when the backend exposes them."""
    colorbar = figure.colorbar(surface, ax=axis, shrink=0.5)
    solids = getattr(colorbar, "solids", None)
    if solids is not None:
        solids.set_rasterized(True)


def plot_noisy_points(axis, x_values, y_values, **plot_kwargs):
    """Plot noisy sample points with the shared marker style used across figures."""
    plot_kwargs.setdefault("color", "gray")
    axis.plot(x_values, y_values, "o", **plot_kwargs)


def plot_true_signal(axis, x_values, y_values, **plot_kwargs):
    """Plot the shared dashed true-signal reference line."""
    plot_kwargs.setdefault("label", "True Signal")
    axis.plot(x_values, y_values, "k--", **plot_kwargs)


# ---------------------------------------------------------------------------
# Foundation plots
# ---------------------------------------------------------------------------


def plot_fraction_comparison():
    """Plot the effect of different smoothing fractions."""
    if not check_file("fraction_comparison.csv"):
        return
    print("Plotting Fraction Comparison...")

    df = pd.read_csv(get_input_path("fraction_comparison.csv"))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))

    fractions = [0.2, 0.5, 0.9]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    titles = [
        "Small Fraction (0.2)\nCaptures Details, May Overfit",
        "Medium Fraction (0.5)\nBalanced Smoothing",
        "Large Fraction (0.9)\nVery Smooth, May Underfit",
    ]

    for ax, frac, color, title in zip(axes, fractions, colors, titles):
        ax.plot(
            df["x"],
            df["y_noisy"],
            "o",
            markersize=3.5,
            alpha=0.3,
            color="gray",
            markeredgewidth=0,
            label="Noisy data",
            zorder=1,
            rasterized=True,
        )
        ax.plot(
            df["x"],
            df["y_true"],
            "k--",
            lw=1.5,
            label="True signal",
            alpha=0.6,
            zorder=2,
        )
        ax.plot(
            df["x"],
            df[f"y_frac_{frac}"],
            color=color,
            lw=2.5,
            label=f"fastLowess (fraction={frac})",
            zorder=3,
        )

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

        rmse = ((df[f"y_frac_{frac}"] - df["y_true"]) ** 2).mean() ** 0.5
        ax.text(
            0.98,
            0.02,
            f"RMSE: {rmse:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    figure.suptitle(
        "Effect of Fraction Parameter on fastLowess Smoothing",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    figure.tight_layout()
    save_figure(figure, "fraction_comparison.svg")


def plot_intervals_comparison():
    """Plot confidence and prediction interval bands around the smooth."""
    if not check_file("intervals_comparison.csv"):
        return
    print("Plotting Intervals Comparison...")

    df = pd.read_csv(get_input_path("intervals_comparison.csv"))
    figure, ax = plt.subplots(figsize=(10, 8))
    x_values = column_to_numpy(df, "x")
    pred_lower = column_to_numpy(df, "pred_lower")
    pred_upper = column_to_numpy(df, "pred_upper")
    conf_lower = column_to_numpy(df, "conf_lower")
    conf_upper = column_to_numpy(df, "conf_upper")

    ax.plot(
        df["x"],
        df["y_noisy"],
        "o",
        markersize=4.5,
        alpha=0.6,
        color="gray",
        markeredgewidth=0,
        label="Noisy Data",
        zorder=1,
    )
    ax.plot(
        df["x"], df["y_true"], "k--", lw=1.5, label="True Signal", alpha=0.7, zorder=2
    )
    ax.plot(df["x"], df["y_smooth"], "k-", lw=2.5, label="fastLowess Fit", zorder=5)

    ax.fill_between(
        x_values,
        cast(Any, pred_lower),
        cast(Any, pred_upper),
        alpha=0.2,
        color="#3b82f6",
        label="95% Prediction Interval",
        zorder=3,
    )
    ax.plot(
        df["x"], df["pred_lower"], linestyle="--", color="#1d4ed8", lw=1.5, alpha=0.6
    )
    ax.plot(
        df["x"], df["pred_upper"], linestyle="--", color="#1d4ed8", lw=1.5, alpha=0.6
    )

    ax.fill_between(
        x_values,
        cast(Any, conf_lower),
        cast(Any, conf_upper),
        alpha=0.4,
        color="#22c55e",
        label="95% Confidence Interval",
        zorder=4,
    )
    ax.plot(
        df["x"], df["conf_lower"], linestyle="--", color="#15803d", lw=1.5, alpha=0.8
    )
    ax.plot(
        df["x"], df["conf_upper"], linestyle="--", color="#15803d", lw=1.5, alpha=0.8
    )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        "Uncertainty Decomposition: Confidence vs Prediction Intervals",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    avg_pred_width = (df["pred_upper"] - df["pred_lower"]).mean()
    avg_conf_width = (df["conf_upper"] - df["conf_lower"]).mean()
    footnote_text = (
        f"Avg Width Ratio (Pred/Conf): {avg_pred_width / avg_conf_width:.2f}x\n"
        "Confidence Interval: Uncertainty in mean curve (Green)\n"
        "Prediction Interval: Uncertainty for new observations (Blue)"
    )
    figure.subplots_adjust(bottom=0.15)
    figure.text(
        0.05,
        0.02,
        footnote_text,
        fontsize=11,
        family="monospace",
        va="bottom",
        ha="left",
    )

    save_figure(figure, "intervals_comparison.svg")


def plot_robust_iter_comparison():
    """Plot the effect of robust reweighting in the presence of outliers."""
    if not check_file("robust_iter_comparison.csv"):
        return
    print("Plotting Robustness Comparison...")

    df = pd.read_csv(get_input_path("robust_iter_comparison.csv"))
    figure, ax = plt.subplots(figsize=(12, 7))
    x_values = df["x"]
    noisy_values = df["y_noisy"]
    true_values = df["y_true"]

    plot_noisy_points(
        ax,
        x_values,
        noisy_values,
        markersize=5.5,
        alpha=0.6,
        label="Noisy Data",
        markeredgewidth=0,
        zorder=1,
    )
    plot_true_signal(
        ax,
        x_values,
        true_values,
        linewidth=1.5,
        alpha=0.6,
        zorder=2,
    )
    ax.plot(
        df["x"],
        df["y_non_robust"],
        color="#ef4444",
        lw=2.0,
        alpha=0.9,
        label="Non-Robust (0 iter)",
    )
    ax.plot(
        df["x"],
        df["y_robust"],
        color="#10b981",
        lw=3.0,
        alpha=0.95,
        label="Robust (6 iter)",
    )

    outlier_mask = np.abs(noisy_values - true_values) > 2.0
    ax.scatter(
        df.loc[outlier_mask, "x"],
        df.loc[outlier_mask, "y_noisy"],
        s=80,
        facecolors="none",
        edgecolors="#ef4444",
        lw=1.5,
        label="Outliers",
        zorder=5,
    )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_ylim(-10, 25)
    ax.set_title("Impact of Robustness Iterations", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    save_figure(figure, "robust_iter_comparison.svg")


def plot_lowess_concept():
    """Visualize the tricube-weighted local linear fit used at one focal point."""
    if not check_file("lowess_concept.csv"):
        return
    print("Plotting Lowess Concept...")

    df = pd.read_csv(get_input_path("lowess_concept.csv"))
    focus_row = df[df["is_focus"] == 1].iloc[0]
    x0 = focus_row["x"]
    y0_fit = focus_row["y_smooth"]
    neighborhood = df[df["weight"] > 0]

    figure, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(
        df["x"], df["y_noisy"], c="lightgray", s=30, alpha=0.5, label="Other Data"
    )
    ax.plot(
        df["x"], df["y_smooth"], color="black", lw=2, alpha=0.3, label="Global Curve"
    )

    scatter = ax.scatter(
        neighborhood["x"],
        neighborhood["y_noisy"],
        c=neighborhood["weight"],
        cmap="Blues",
        s=60,
        edgecolor="k",
        linewidth=0.5,
        label="Tricube-Weighted Points",
    )

    ax.plot(
        neighborhood["x"],
        neighborhood["y_local_fit_x0"],
        color="#d97706",
        lw=3,
        label="Weighted Linear Fit",
    )

    ax.scatter(
        [x0],
        [y0_fit],
        s=150,
        facecolor="#d97706",
        edgecolor="white",
        lw=2,
        zorder=10,
        label="Fitted Value",
    )

    ax.set_title(f"How LOWESS Works (Focus x={x0:.2f})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    colorbar = figure.colorbar(scatter, ax=ax)
    colorbar.set_label("Weight", fontsize=10)

    save_figure(figure, "lowess_concept.svg")


# ---------------------------------------------------------------------------
# Modeling plots
# ---------------------------------------------------------------------------


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
        ("y_epanechnikov", "Epanechnikov", "--", "#ef4444", 2.5),
        ("y_gaussian", "Gaussian", "-.", "#10b981", 2.5),
        ("y_uniform", "Uniform", ":", "#f59e0b", 2.5),
        ("y_biweight", "Biweight", "-", "#8b5cf6", 2.0),
        ("y_triangle", "Triangle", "--", "#ec4899", 2.0),
        ("y_cosine", "Cosine", "-.", "#06b6d4", 2.0),
    ]

    for col, label, ls, color, lw in configs:
        ax.plot(df["x"], df[col], ls, lw=lw, label=label, color=color)

    ax.set_title("Kernel Weight Function Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    save_figure(figure, "kernel_comparison.svg")


def plot_robust_method_comparison():
    """Plot how different robustness methods handle outliers."""
    if not check_file("robust_method_comparison.csv"):
        return
    print("Plotting Robust Method Comparison...")

    df = pd.read_csv(get_input_path("robust_method_comparison.csv"))
    figure, ax = plt.subplots(figsize=(10, 6))

    plot_noisy_points(
        ax, df["x"], df["y_noisy"], markersize=4, alpha=0.4, label="Data with Outliers"
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.5)
    ax.plot(
        df["x"],
        df["y_bisquare"],
        "-",
        lw=2.5,
        label="Bisquare (default)",
        color="#3b82f6",
    )
    ax.plot(df["x"], df["y_huber"], "--", lw=2.5, label="Huber", color="#ef4444")
    ax.plot(df["x"], df["y_talwar"], ":", lw=2.5, label="Talwar", color="#10b981")

    ax.set_title(
        "Robust Weight Functions: Bisquare vs Huber vs Talwar",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(figure, "robust_method_comparison.svg")


def plot_boundary_policy_comparison():
    """Plot four boundary handling modes on the same noisy signal."""
    if not check_file("boundary_comparison.csv"):
        return
    print("Plotting Boundary Policy Comparison...")

    df = pd.read_csv(get_input_path("boundary_comparison.csv"))
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax in (ax1, ax2):
        plot_noisy_points(
            ax, df["x"], df["y_noisy"], markersize=3, alpha=0.3, label="Noisy Data"
        )
        plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.5)

    ax1.plot(
        df["x"], df["y_extend"], "-", lw=2.5, label="Extend (default)", color="#3b82f6"
    )
    ax1.plot(df["x"], df["y_reflect"], "--", lw=2.5, label="Reflect", color="#ef4444")
    ax1.set_title("Extend vs Reflect", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["x"], df["y_zero"], "-", lw=2.5, label="Zero", color="#10b981")
    ax2.plot(
        df["x"], df["y_noboundary"], "--", lw=2.5, label="NoBoundary", color="#f59e0b"
    )
    ax2.set_title("Zero vs NoBoundary", fontsize=14, fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    figure.suptitle(
        "Mitigating Boundary Bias with Reflection and Extension",
        fontsize=16,
        fontweight="bold",
    )

    save_figure(figure, "boundary_comparison.svg")


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
        alpha=0.3,
        label="Data (20% moderate +1.5, 20% extreme +6)",
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.5)
    ax.plot(
        df["x"],
        df["y_none"],
        ":",
        lw=2,
        label="No Robustness \u2014 both tiers bias the fit",
        color="#10b981",
    )
    ax.plot(
        df["x"],
        df["y_mean"],
        "--",
        lw=2,
        label="Mean (MAE) \u2014 extreme outliers inflate scale \u2192 moderate outliers slip through",
        color="#ef4444",
    )
    ax.plot(
        df["x"],
        df["y_mad"],
        "-.",
        lw=2,
        label="MAD \u2014 centers on median residual first",
        color="#f59e0b",
    )
    ax.plot(
        df["x"],
        df["y_mar"],
        "-",
        lw=2.5,
        label="MAR \u2014 scale anchored to clean noise level \u2192 rejects all outliers",
        color="#3b82f6",
    )

    ax.set_title(
        "Robust Scaling: MAR vs MAD vs Mean (MAE) vs None\n"
        "Extreme outliers inflate Mean scale \u2192 looser threshold \u2192 moderate outliers leak in",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    save_figure(figure, "scaling_comparison.svg")


def plot_zero_weight_comparison():
    """Compare ZeroWeightFallback policies: UseLocalMean, ReturnOriginal, ReturnNone."""
    if not check_file("zero_weight_comparison.csv"):
        return
    print("Plotting Zero Weight Fallback Comparison...")

    df = pd.read_csv(get_input_path("zero_weight_comparison.csv"))
    figure, ax = plt.subplots(figsize=(11, 6))

    ax.axvspan(
        4.0,
        6.0,
        alpha=0.07,
        color="red",
        zorder=0,
        label="Anomalous zone",
    )

    plot_noisy_points(
        ax,
        df["x"],
        df["y_noisy"],
        markersize=3,
        alpha=0.25,
        label="Data",
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1.2, alpha=0.5)

    ax.plot(
        df["x"],
        df["y_local_mean"],
        "-",
        lw=2.5,
        label="UseLocalMean",
        color="#3b82f6",
        zorder=3,
    )
    ax.plot(
        df["x"],
        df["y_return_original"],
        "--",
        lw=2,
        label="ReturnOriginal",
        color="#ef4444",
        zorder=3,
    )
    ax.plot(
        df["x"],
        df["y_return_none"],
        ":",
        lw=2,
        label="ReturnNone",
        color="#f59e0b",
        zorder=3,
    )

    ax.set_title(
        "Zero-Weight Fallback Policies: UseLocalMean / ReturnOriginal / ReturnNone",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    save_figure(figure, "zero_weight_comparison.svg")


# ---------------------------------------------------------------------------
# Streaming / online / convergence plots
# ---------------------------------------------------------------------------


def plot_merge_comparison():
    """Compare streaming merge strategies."""
    if not check_file("merge_comparison.csv"):
        return
    print("Plotting Streaming Comparison...")

    df = pd.read_csv(get_input_path("merge_comparison.csv"))
    figure, ax = plt.subplots(figsize=(12, 7))

    plot_noisy_points(
        ax, df["x"], df["y_noisy"], markersize=3, alpha=0.2, label="Noisy Data"
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1.5, alpha=0.6)
    ax.plot(
        df["x"],
        df["y_weighted"],
        "-",
        lw=3,
        label="Streaming (WeightedAverage)",
        color="#3b82f6",
    )
    ax.plot(
        df["x"],
        df["y_average"],
        "--",
        lw=2,
        label="Streaming (Average)",
        color="#ef4444",
    )
    ax.plot(
        df["x"],
        df["y_first"],
        ":",
        lw=2,
        label="Streaming (TakeFirst)",
        color="#10b981",
    )

    ax.set_title(
        "Streaming fastLowess: Comparison of Merge Strategies",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Time (x)")
    ax.set_ylabel("Value (y)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(figure, "merge_comparison.svg")


def plot_online_comparison():
    """Compare online smoothing settings with different window sizes."""
    if not check_file("online_comparison.csv"):
        return
    print("Plotting Online Comparison...")

    df = pd.read_csv(get_input_path("online_comparison.csv"))
    figure, ax = plt.subplots(figsize=(12, 7))

    plot_noisy_points(
        ax, df["x"], df["y_noisy"], markersize=3, alpha=0.3, label="Streaming Data"
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1.5, alpha=0.6)
    ax.plot(
        df["x"],
        df["y_small_window"],
        "-",
        lw=2.5,
        label="Online (Window=50, Incremental)",
        color="#ef4444",
    )
    ax.plot(
        df["x"],
        df["y_large_window"],
        "-",
        lw=3,
        label="Online (Window=200, Full + Robust)",
        color="#3b82f6",
    )

    ax.set_title(
        "Online fastLowess: Incremental Smoothing with Sliding Window",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Point Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(250, color="gray", linestyle=":", alpha=0.5)
    ax.text(255, ax.get_ylim()[1] * 0.9, "Signal Shift", color="gray", fontsize=10)

    save_figure(figure, "online_comparison.svg")


def plot_adapter_comparison():
    """Plot the effect of automatic convergence across adapter styles."""
    if not check_file("adapter_comparison.csv"):
        return
    print("Plotting Auto-Convergence Comparison...")

    df = pd.read_csv(get_input_path("adapter_comparison.csv"))
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    adapters = [
        ("Batch", "batch", ax1, "#3b82f6"),
        ("Streaming", "stream", ax2, "#10b981"),
        ("Online", "online", ax3, "#ef4444"),
    ]

    for name, key, ax, base_color in adapters:
        plot_noisy_points(
            ax, df["x"], df["y_noisy"], markersize=2, alpha=0.15, label="Noisy Data"
        )
        plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.4)
        ax.plot(
            df["x"],
            df[f"y_{key}_off"],
            "--",
            lw=3,
            label=f"{name} (Standard)",
            color="gray",
        )
        ax.plot(
            df["x"],
            df[f"y_{key}_on"],
            "-",
            lw=2,
            label=f"{name} (Auto-Converge)",
            color=base_color,
        )

        saved = df[f"iter_{key}_off"] - df[f"iter_{key}_on"]
        avg_saved = saved.mean()
        total_saved = saved.sum()

        ax.set_title(
            f"{name} Adapter\nSaved: {total_saved:.0f} iters (Avg {avg_saved:.1f}/pt)",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X")
        if ax == ax1:
            ax.set_ylabel("Y")

    figure.tight_layout()
    save_figure(figure, "adapter_comparison.svg")


# ---------------------------------------------------------------------------
# CLI dispatcher
# ---------------------------------------------------------------------------


def build_plot_targets():
    """Return the stable CLI mapping from target names to plot functions."""
    return {
        "fraction": plot_fraction_comparison,
        "intervals": plot_intervals_comparison,
        "robustness": plot_robust_iter_comparison,
        "concept": plot_lowess_concept,
        "kernel": plot_kernel_comparison,
        "robust_method": plot_robust_method_comparison,
        "boundary": plot_boundary_policy_comparison,
        "gap": plot_gap_handling,
        "cv": plot_cv_comparison,
        "surface": plot_surface_mode_comparison,
        "scaling": plot_scaling_comparison,
        "zero_weight": plot_zero_weight_comparison,
        "streaming": plot_merge_comparison,
        "online": plot_online_comparison,
        "auto_converge": plot_adapter_comparison,
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
