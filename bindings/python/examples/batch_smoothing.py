#!/usr/bin/env python3
"""
fastlowess Batch Smoothing Example

This example demonstrates batch LOWESS smoothing features:
- Basic smoothing with different parameters
- Robustness iterations for outlier handling
- Confidence and prediction intervals
- Diagnostics and cross-validation

The batch adapter (smooth function) is the primary interface for
processing complete datasets that fit in memory.
"""

import numpy as np
import matplotlib.pyplot as plt
import fastlowess
from fastlowess import smooth
import os

def generate_sample_data(n_points=1000):
    """
    Generate complex sample data with a trend, seasonality, and outliers.
    """
    np.random.seed(42)
    x = np.linspace(0, 50, n_points)
    
    # Trend + Seasonality
    y_true = 0.5 * x + 5 * np.sin(x * 0.5)
    
    # Gaussian noise
    y = y_true + np.random.normal(0, 1.5, n_points)
    
    # Add significant outliers (10% of data)
    n_outliers = int(n_points * 0.1)
    outlier_indices = np.random.choice(n_points, size=n_outliers, replace=False)
    y[outlier_indices] += np.random.uniform(10, 20, n_outliers) * np.random.choice([-1, 1], n_outliers)
    
    return x, y, y_true

def main():
    print("=== fastlowess Batch Smoothing Example ===")
    
    # 1. Generate Data
    x, y, y_true = generate_sample_data(1000)
    print(f"Generated {len(x)} data points with outliers.")

    # 2. Basic Smoothing (Default parameters)
    print("Running basic smoothing...")
    # Use a smaller fraction (0.05) to capture the sine wave seasonality
    res_basic = smooth(x, y, iterations=0, fraction=0.05)
    
    # 3. Robust Smoothing (IRLS)
    print("Running robust smoothing (3 iterations)...")
    res_robust = smooth(
        x, y, 
        fraction=0.05, 
        iterations=3, 
        robustness_method="bisquare",
        return_robustness_weights=True
    )
    
    # 4. Uncertainty Quantification
    print("Computing confidence and prediction intervals...")
    res_intervals = smooth(
        x, y, 
        fraction=0.05, 
        confidence_intervals=0.95, 
        prediction_intervals=0.95,
        return_diagnostics=True
    )
    
    # 5. Cross-Validation for optimal fraction
    print("Running cross-validation to find optimal fraction...")
    cv_fractions = [0.05, 0.1, 0.2, 0.4]
    res_cv = smooth(x, y, cv_fractions=cv_fractions, cv_method="kfold", cv_k=5)
    print(f"Optimal fraction found: {res_cv.fraction_used}")

    # Plotting Results
    os.makedirs("examples/plots", exist_ok=True)
    
    fig1 = plt.figure(figsize=(12, 8))
    
    # Original Data
    plt.scatter(x, y, alpha=0.3, color='gray', s=10, label='Noisy Data (w/ Outliers)')
    plt.plot(x, y_true, 'k--', alpha=0.8, label='True Signal')
    
    # Basic Smoothing
    plt.plot(x, res_basic.y, 'r-', linewidth=2, label=f'Basic LOWESS (Non-robust)')
    
    # Robust Smoothing
    plt.plot(x, res_robust.y, 'g-', linewidth=2.5, label='Robust LOWESS (3 iters)')
    
    # Confidence Intervals
    plt.fill_between(
        x, 
        res_intervals.confidence_lower, 
        res_intervals.confidence_upper, 
        color='blue', 
        alpha=0.2, 
        label='95% Confidence Interval'
    )
    
    # Diagnostics Printout
    diag = res_intervals.diagnostics
    print("\nFit Statistics (Intervals Model):")
    print(f" - R²:   {diag.r_squared:.4f}")
    print(f" - RMSE: {diag.rmse:.4f}")
    print(f" - MAE:  {diag.mae:.4f}")

    plt.title("fastlowess: Robust Batch Smoothing with Intervals")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot for robustness weights
    fig2 = plt.figure(figsize=(12, 3))
    plt.scatter(x, res_robust.robustness_weights, c=res_robust.robustness_weights, cmap='viridis', s=10)
    plt.title("Robustness Weights (Low weight = Outlier suspected)")
    plt.colorbar(label='Weight')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    # 6. Boundary Policy Comparison
    print("\nDemonstrating boundary policy effects on linear data...")
    xl = np.linspace(0, 10, 50)
    yl = 2 * xl + 1
    
    # Compare policies
    r_ext = smooth(xl, yl, fraction=0.6, boundary_policy="extend")
    r_ref = smooth(xl, yl, fraction=0.6, boundary_policy="reflect")
    r_zr = smooth(xl, yl, fraction=0.6, boundary_policy="zero")

    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(xl, yl, 'k--', label='True Linear Trend')
    plt.plot(xl, r_ext.y, 'r-', label='Extend (Default) - constant padding')
    plt.plot(xl, r_ref.y, 'g-', label='Reflect - mirrored padding')
    plt.plot(xl, r_zr.y, 'b-', label='Zero - zero padding')
    
    plt.title("Effect of Boundary Policies on Linear Data (q=0.6)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    print("\nSaving plots to examples/plots/...")
    fig1.savefig("examples/plots/batch_main.png")
    fig2.savefig("examples/plots/batch_weights.png")
    fig3.savefig("examples/plots/batch_boundary.png")
    print("Done!")

if __name__ == "__main__":
    main()
