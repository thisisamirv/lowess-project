#!/usr/bin/env python3
"""Core validation plots for fit quality and uncertainty."""

from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    from .plot_common import (
        add_rasterized_colorbar,
        check_file,
        column_to_numpy,
        get_input_path,
        plot_noisy_points,
        plot_true_signal,
        save_figure,
    )
except ImportError:
    from plot_common import (  # type: ignore
        add_rasterized_colorbar,
        check_file,
        column_to_numpy,
        get_input_path,
        plot_noisy_points,
        plot_true_signal,
        save_figure,
    )


def plot_degree_comparison():
    """Plot the linear-versus-quadratic smoothing comparison."""
    if not check_file("degree_comparison.csv"):
        return
    print("Plotting Degree Comparison...")

    df = pd.read_csv(get_input_path("degree_comparison.csv"))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax1 = axes[0]
    ax1.plot(
        df["x"],
        df["y_noisy"],
        "o",
        markersize=3,
        alpha=0.4,
        color="gray",
        markeredgewidth=0,
        label="Noisy data",
        rasterized=True,
    )
    ax1.plot(df["x"], df["y_true"], "k-", lw=1.5, label="True signal", alpha=0.7)
    ax1.plot(df["x"], df["y_lowess"], "b-", lw=2, label="LOWESS (Linear)")
    ax1.plot(df["x"], df["y_fastLowess"], "r-", lw=2, label="fastLowess (Quadratic)")
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("LOWESS vs fastLowess: Full View", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    mask = (df["x"] > -0.6) & (df["x"] < 0.6)
    ax2.plot(
        df.loc[mask, "x"],
        df.loc[mask, "y_noisy"],
        "o",
        markersize=4,
        alpha=0.5,
        color="gray",
        markeredgewidth=0,
        label="Noisy data",
        rasterized=True,
    )
    ax2.plot(df.loc[mask, "x"], df.loc[mask, "y_true"], "k-", lw=2, label="True signal")
    ax2.plot(
        df.loc[mask, "x"],
        df.loc[mask, "y_lowess"],
        "b-",
        lw=2.5,
        label="LOWESS (Linear)",
    )
    ax2.plot(
        df.loc[mask, "x"],
        df.loc[mask, "y_fastLowess"],
        "r-",
        lw=2.5,
        label="fastLowess (Quadratic)",
    )
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.set_title("Peak Region: Linear Flattens, Quadratic Captures", fontsize=14)
    ax2.legend(loc="lower center")
    ax2.grid(True, alpha=0.3)

    peak_x = df.loc[df["y_true"].idxmax(), "x"]
    peak_true = df["y_true"].max()
    peak_lowess = df.loc[df["y_true"].idxmax(), "y_lowess"]

    ax2.annotate(
        f"Gap = {peak_true - peak_lowess:.3f}",
        xy=(peak_x, (peak_lowess + peak_true) / 2),
        xytext=(0.3, 0.85),
        fontsize=10,
        color="blue",
        arrowprops={"arrowstyle": "->", "color": "blue", "lw": 1.5},
    )

    figure.tight_layout()
    save_figure(figure, "degree_comparison.svg", dpi=72, bbox_inches=None)


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


def plot_robustness_comparison():
    """Plot the effect of robust reweighting in the presence of outliers."""
    if not check_file("robustness_comparison.csv"):
        return
    print("Plotting Robustness Comparison...")

    df = pd.read_csv(get_input_path("robustness_comparison.csv"))
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

    save_figure(figure, "robustness_comparison.svg")


def plot_fast_lowess_concept():
    """Visualize the local neighborhood and fit used at one focal point."""
    if not check_file("fastLowess_concept.csv"):
        return
    print("Plotting fastLowess Concept...")

    df = pd.read_csv(get_input_path("fastLowess_concept.csv"))
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
        label="Neighborhood",
    )

    ax.plot(
        neighborhood["x"],
        neighborhood["y_local_fit_x0"],
        color="#d97706",
        lw=3,
        label="Local Polynomial",
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

    ax.set_title(
        f"How fastLowess Works (Focus x={x0:.2f})", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    colorbar = figure.colorbar(scatter, ax=ax)
    colorbar.set_label("Weight", fontsize=10)

    save_figure(figure, "fastLowess_concept.svg")


def plot_multivariate_fast_lowess():
    """Plot the true and smoothed multivariate surfaces side by side."""
    if not check_file("multivariate_fastLowess.csv"):
        return
    print("Plotting Multivariate fastLowess...")

    df = pd.read_csv(get_input_path("multivariate_fastLowess.csv"))
    n_x = len(df["x"].unique())
    n_y = len(df["y"].unique())
    x_grid = column_to_numpy(df, "x").reshape(n_x, n_y)
    y_grid = column_to_numpy(df, "y").reshape(n_x, n_y)
    z_true_grid = column_to_numpy(df, "z_true").reshape(n_x, n_y)
    z_smooth_grid = column_to_numpy(df, "z_smooth").reshape(n_x, n_y)

    figure = plt.figure(figsize=(10, 5))

    ax1 = figure.add_subplot(1, 2, 1, projection="3d")
    surf1 = ax1.plot_surface(
        x_grid,
        y_grid,
        z_true_grid,
        cmap="viridis",
        alpha=0.9,
        edgecolor="none",
        rasterized=True,
    )
    ax1.set_title("True Surface")
    ax1.locator_params(nbins=4)
    add_rasterized_colorbar(figure, surf1, ax1)

    ax2 = figure.add_subplot(1, 2, 2, projection="3d")
    surf2 = ax2.plot_surface(
        x_grid,
        y_grid,
        z_smooth_grid,
        cmap="magma",
        alpha=0.9,
        edgecolor="none",
        rasterized=True,
    )
    ax2.set_title("fastLowess Smoothed")
    ax2.locator_params(nbins=4)
    add_rasterized_colorbar(figure, surf2, ax2)

    figure.tight_layout()
    save_figure(figure, "multivariate_fastLowess.svg", dpi=100)
