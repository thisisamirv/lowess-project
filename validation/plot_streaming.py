#!/usr/bin/env python3
"""Validation plots for streaming, online, and convergence workflows."""

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


def plot_streaming_comparison():
    """Compare streaming merge strategies."""
    if not check_file("streaming_comparison.csv"):
        return
    print("Plotting Streaming Comparison...")

    df = pd.read_csv(get_input_path("streaming_comparison.csv"))
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

    save_figure(figure, "streaming_comparison.svg")


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


def plot_auto_converge_comparison():
    """Plot the effect of automatic convergence across adapter styles."""
    if not check_file("auto_converge_comparison.csv"):
        return
    print("Plotting Auto-Convergence Comparison...")

    df = pd.read_csv(get_input_path("auto_converge_comparison.csv"))
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
    save_figure(figure, "auto_converge_comparison.svg")
