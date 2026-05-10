#!/usr/bin/env python3
"""Shared plotting helpers for validation visualizations."""

import os

import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = "output/visual"
OUTPUT_DIR = "output/fig"

# Optimized SVG settings
plt.rcParams["svg.fonttype"] = "none"


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
