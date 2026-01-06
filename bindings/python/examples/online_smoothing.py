#!/usr/bin/env python3
"""
fastlowess Online Smoothing Example

This example demonstrates online LOWESS smoothing for real-time data:
- Basic incremental processing with streaming data
- Real-time sensor data smoothing
- Different update modes (Full vs Incremental)
- Memory-bounded processing with sliding window

The online adapter (smooth_online function) is designed for:
- Real-time data streams
- Sensors and monitoring
- Low-latency applications
"""

import numpy as np
import matplotlib.pyplot as plt
import fastlowess
from fastlowess import smooth_online
import os

def main():
    print("=== fastlowess Online Smoothing Example ===")
    
    # 1. Simulate a real-time signal
    # A sine wave with changing frequency and random noise
    n_points = 1000
    x = np.arange(n_points, dtype=float)
    y_true = 20.0 + 5.0 * np.sin(x * 0.1) + 2.0 * np.sin(x * 0.02)
    y = y_true + np.random.normal(0, 1.2, n_points)
    
    # Add some sudden spikes (sensor glitches)
    y[200:205] += 15.0
    y[600:610] -= 10.0
    
    print(f"Simulating {n_points} real-time data points...")

    # 2. Sequential Online Processing
    # Full Update Mode (higher accuracy)
    print("Processing with 'full' update mode...")
    res_full = smooth_online(
        x, y, 
        fraction=0.3, 
        window_capacity=50, 
        iterations=3,
        update_mode="full"
    )
    
    # Incremental Update Mode (faster for large windows)
    print("Processing with 'incremental' update mode...")
    res_inc = smooth_online(
        x, y, 
        fraction=0.3, 
        window_capacity=50, 
        iterations=3,
        update_mode="incremental"
    )

    # Plotting
    os.makedirs("examples/plots", exist_ok=True)
    
    fig1 = plt.figure(figsize=(12, 7))
    
    # Original Data
    plt.scatter(x, y, s=5, alpha=0.3, color='gray', label='Raw Sensor Stream')
    plt.plot(x, y_true, 'k--', alpha=0.6, label='True Signal')
    
    # Online Results
    plt.plot(x, res_full.y, 'r-', linewidth=2, label='Online LOWESS (Full)')
    plt.plot(x, res_inc.y, 'b-', linewidth=1.5, alpha=0.7, label='Online LOWESS (Incremental)')
    
    # Highlight a zoom area to show the windowing effect
    plt.axvspan(400, 500, color='yellow', alpha=0.1, label='Zoom Area')

    plt.title("fastlowess: Real-time Online Smoothing (Sliding Window)")
    plt.xlabel("Time / Sequence Index")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom In
    fig2 = plt.figure(figsize=(12, 4))
    mask = (x >= 400) & (x <= 500)
    plt.scatter(x[mask], y[mask], s=20, alpha=0.4, color='gray')
    plt.plot(x[mask], y_true[mask], 'k--')
    plt.plot(x[mask], res_full.y[mask], 'r-', linewidth=3, label='Full Update')
    plt.plot(x[mask], res_inc.y[mask], 'b-', linewidth=2, label='Incremental')
    plt.title("Detailed View (Time 400-500)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    print("\nSaving plots to examples/plots/...")
    fig1.savefig("examples/plots/online_main.png")
    fig2.savefig("examples/plots/online_zoom.png")
    print("Done!")

if __name__ == "__main__":
    main()
