#!/usr/bin/env python3
"""
fastlowess Streaming Smoothing Example

This example demonstrates streaming LOWESS smoothing for large datasets:
- Basic chunked processing
- Handling datasets that don't fit in memory
- Parallel execution for extreme speed
"""

import time
import os

import matplotlib.pyplot as plt
import numpy as np

from fastlowess import Lowess, StreamingLowess

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")


def main():
    """Run the streaming smoothing example and compare it with batch mode."""
    print("=== fastlowess Streaming Mode Example ===")
    n_points = 100_000
    print(f"Generating large dataset: {n_points} points...")
    x = np.linspace(0, 100, n_points)
    y = np.cos(x * 0.1) + np.random.normal(0, 0.5, n_points)

    res_batch, batch_time = run_batch_smoothing(x, y)
    full_stream_y, stream_time = run_streaming_smoothing(x, y, n_points)
    compare_results(res_batch.y, full_stream_y)
    save_streaming_plot(x, y, res_batch.y, full_stream_y, n_points)
    print(f"Batch took: {batch_time:.4f} seconds")
    print(f"Streaming took: {stream_time:.4f} seconds")


def run_batch_smoothing(x, y):
    """Run the baseline batch smoothing pass and return the result and timing."""
    start = time.time()
    print("Running Batch LOWESS (Parallel)...")
    res_batch = Lowess(fraction=0.01).fit(x, y)
    return res_batch, time.time() - start


def run_streaming_smoothing(x, y, n_points):
    """Run the streaming smoother in chunks and return output and timing."""
    start = time.time()
    print("Running Streaming LOWESS (Chunked)...")
    chunk_size = 10_000
    model = StreamingLowess(fraction=0.01, chunk_size=2000, overlap=200, parallel=True)
    stream_y = []

    for i in range(0, n_points, chunk_size):
        cx = x[i : i + chunk_size]
        cy = y[i : i + chunk_size]
        res = model.process_chunk(cx, cy)
        if hasattr(res, "y") and len(res.y) > 0:
            stream_y.append(res.y)

    res_final = model.finalize()
    if hasattr(res_final, "y") and len(res_final.y) > 0:
        stream_y.append(res_final.y)

    full_stream_y = np.concatenate(stream_y)
    print(f"Stream output length: {len(full_stream_y)}")
    return full_stream_y, time.time() - start


def compare_results(batch_y, full_stream_y):
    """Report the numerical difference between batch and streaming outputs."""
    if len(full_stream_y) == len(batch_y):
        mse = np.mean((batch_y - full_stream_y) ** 2)
        print(f"Mean Squared Difference (Batch vs Stream): {mse:.2e}")
        return

    print(
        f"Warning: Length mismatch. Batch={len(batch_y)}, Stream={len(full_stream_y)}"
    )
    min_len = min(len(batch_y), len(full_stream_y))
    mse = np.mean((batch_y[:min_len] - full_stream_y[:min_len]) ** 2)
    print(f"Mean Squared Difference (First {min_len} points): {mse:.2e}")


def save_streaming_plot(x, y, batch_y, full_stream_y, n_points):
    """Save a zoomed plot comparing batch and streaming smoothing outputs."""
    zoom_range = (40, 60)
    zoom_mask = (x >= zoom_range[0]) & (x <= zoom_range[1])
    min_len = min(len(x), len(full_stream_y))

    plt.figure(figsize=(12, 8))
    display_mask = np.random.choice([False, True], size=n_points, p=[0.99, 0.01])
    plt.scatter(
        x[display_mask & zoom_mask],
        y[display_mask & zoom_mask],
        alpha=0.3,
        color="gray",
        s=10,
        label="Raw Data (sampled)",
    )
    plt.plot(x[zoom_mask], batch_y[zoom_mask], "r-", linewidth=3, label="Batch Result")
    plt.plot(
        x[:min_len][zoom_mask[:min_len]],
        full_stream_y[:min_len][zoom_mask[:min_len]],
        "b--",
        linewidth=2,
        label="Streaming Result",
    )

    plt.title(f"fastlowess: Streaming Smoothing on {n_points} points")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.xlim(zoom_range)
    plt.ylim(-2.5, 2.5)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "streaming_smoothing.png"))
    print(f"\nPlot saved to {PLOTS_DIR}/streaming_smoothing.png")


if __name__ == "__main__":
    main()
