#!/usr/bin/env python3
"""
fastlowess Streaming Smoothing Example

This example demonstrates streaming LOWESS smoothing for large datasets:
- Basic chunked processing
- Handling datasets that don't fit in memory
- Parallel execution for extreme speed
"""

import numpy as np
import matplotlib.pyplot as plt
from fastlowess import StreamingLowess, Lowess
import time
import os

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")


def main():
    print("=== fastlowess Streaming Mode Example ===")

    # 1. Generate Very Large Dataset
    # 100,000 points
    n_points = 100_000
    print(f"Generating large dataset: {n_points} points...")
    x = np.linspace(0, 100, n_points)
    y = np.cos(x * 0.1) + np.random.normal(0, 0.5, n_points)

    # 2. Regular Batch Smoothing (for comparison)
    start = time.time()
    print("Running Batch LOWESS (Parallel)...")
    res_batch = Lowess(fraction=0.01).fit(x, y)
    batch_time = time.time() - start
    print(f"Batch took: {batch_time:.4f} seconds")

    # 3. Streaming Mode
    # Divide the data into chunks of 10,000 for low memory usage
    start = time.time()
    print("Running Streaming LOWESS (Chunked)...")
    chunk_size = 10_000
    model = StreamingLowess(fraction=0.01, chunk_size=2000, overlap=200, parallel=True)

    stream_y = []

    # Process in chunks
    for i in range(0, n_points, chunk_size):
        cx = x[i : i + chunk_size]
        cy = y[i : i + chunk_size]
        res = model.process_chunk(cx, cy)
        # Check if result has data (it might be empty if accumulating for overlap)
        if hasattr(res, "y") and len(res.y) > 0:
            stream_y.append(res.y)

    # Finalize
    res_final = model.finalize()
    if hasattr(res_final, "y") and len(res_final.y) > 0:
        stream_y.append(res_final.y)

    # Concatenate all results
    full_stream_y = np.concatenate(stream_y)

    stream_time = time.time() - start
    print(f"Streaming took: {stream_time:.4f} seconds")
    print(f"Stream output length: {len(full_stream_y)}")

    # 4. Verify Accuracy
    # Note: Streamed output might be slightly shorter or different length depending on boundary logic,
    # but with extend policy it should match input length ideally.
    if len(full_stream_y) == len(res_batch.y):
        mse = np.mean((res_batch.y - full_stream_y) ** 2)
        print(f"Mean Squared Difference (Batch vs Stream): {mse:.2e}")
    else:
        print(
            f"Warning: Length mismatch. Batch={len(res_batch.y)}, Stream={len(full_stream_y)}"
        )
        # Check intersection for MSE
        min_len = min(len(res_batch.y), len(full_stream_y))
        mse = np.mean((res_batch.y[:min_len] - full_stream_y[:min_len]) ** 2)
        print(f"Mean Squared Difference (First {min_len} points): {mse:.2e}")

    # Plotting Results
    zoom_range = (40, 60)
    zoom_mask = (x >= zoom_range[0]) & (x <= zoom_range[1])

    plt.figure(figsize=(12, 8))

    # Raw Data (downsampled for performance)
    display_mask = np.random.choice([False, True], size=n_points, p=[0.99, 0.01])
    plt.scatter(
        x[display_mask & zoom_mask],
        y[display_mask & zoom_mask],
        alpha=0.3,
        color="gray",
        s=10,
        label="Raw Data (sampled)",
    )

    # Smooth Curves
    plt.plot(
        x[zoom_mask], res_batch.y[zoom_mask], "r-", linewidth=3, label="Batch Result"
    )

    # Ensure stream_y matches x dimensions for plotting (basic trim/pad logic for display)
    min_len = min(len(x), len(full_stream_y))

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
