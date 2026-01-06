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
import fastlowess
from fastlowess import smooth_streaming, smooth
import time
import os

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
    res_batch = smooth(x, y, fraction=0.01) 
    batch_time = time.time() - start
    print(f"Batch took: {batch_time:.4f} seconds")

    # 3. Streaming Mode
    # Divide the data into chunks of 10,000 for low memory usage
    start = time.time()
    print("Running Streaming LOWESS (Chunked)...")
    res_stream = smooth_streaming(
        x, y, 
        fraction=0.01, 
        chunk_size=2000, 
        overlap=200,
        parallel=True
    )
    stream_time = time.time() - start
    print(f"Streaming took: {stream_time:.4f} seconds")

    # 4. Verify Accuracy
    mse = np.mean((res_batch.y - res_stream.y)**2)
    print(f"Mean Squared Difference (Batch vs Stream): {mse:.2e}")

    # Plotting Results
    zoom_range = (40, 60)
    zoom_mask = (x >= zoom_range[0]) & (x <= zoom_range[1])
    
    plt.figure(figsize=(12, 8))
    
    # Raw Data (downsampled for performance)
    display_mask = np.random.choice([False, True], size=n_points, p=[0.99, 0.01])
    plt.scatter(x[display_mask & zoom_mask], y[display_mask & zoom_mask], alpha=0.3, color='gray', s=10, label='Raw Data (sampled)')
    
    # Smooth Curves
    plt.plot(x[zoom_mask], res_batch.y[zoom_mask], 'r-', linewidth=3, label='Batch Result')
    plt.plot(x[zoom_mask], res_stream.y[zoom_mask], 'b--', linewidth=2, label='Streaming Result')
    
    plt.title(f"fastlowess: Streaming Smoothing on {n_points} points")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.xlim(zoom_range)
    plt.ylim(-2.5, 2.5)
    
    plt.tight_layout()
    os.makedirs("examples/plots", exist_ok=True)
    plt.savefig("examples/plots/streaming_smoothing.png")
    print("\nPlot saved to examples/plots/streaming_smoothing.png")

if __name__ == "__main__":
    main()
