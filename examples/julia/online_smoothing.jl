#!/usr/bin/env julia
"""
FastLOWESS Online Smoothing Example

This example demonstrates online LOWESS smoothing for real-time data:
- Basic incremental processing with streaming data
- Real-time sensor data smoothing
- Different update modes (Full vs Incremental)
- Memory-bounded processing with sliding window

The OnlineLowess class is designed for:
- Real-time data streams
- Sensors and monitoring
- Low-latency applications
"""

using Random
using Printf

# Handle package loading - check if we're already in the FastLOWESS project
using Pkg
project_name = Pkg.project().name
if project_name != "FastLOWESS"
    # Not in the FastLOWESS project, need to develop it
    script_dir = @__DIR__
    julia_pkg_dir = joinpath(dirname(script_dir), "julia")
    if !haskey(Pkg.project().dependencies, "FastLOWESS")
        Pkg.develop(path = julia_pkg_dir)
    end
end

using FastLOWESS

function main()
    println("=== FastLOWESS Online Smoothing Example ===")

    # 1. Simulate a real-time signal
    # A sine wave with changing frequency and random noise
    n_points = 1000
    Random.seed!(42)
    x = collect(Float64, 0:(n_points-1))
    y_true = 20.0 .+ 5.0 .* sin.(x .* 0.1) .+ 2.0 .* sin.(x .* 0.02)
    y = y_true .+ randn(n_points) .* 1.2

    # Add some sudden spikes (sensor glitches)
    y[201:205] .+= 15.0
    y[601:610] .-= 10.0

    println("Simulating $n_points real-time data points...")

    # 2. Sequential Online Processing
    # Full Update Mode (higher accuracy)
    println("Processing with 'full' update mode...")
    model_full = OnlineLowess(
        fraction = 0.3,
        window_capacity = 50,
        iterations = 3,
        update_mode = "full",
    )
    res_full = add_points(model_full, x, y)

    # Incremental Update Mode (faster for large windows)
    println("Processing with 'incremental' update mode...")
    model_inc = OnlineLowess(
        fraction = 0.3,
        window_capacity = 50,
        iterations = 3,
        update_mode = "incremental",
    )
    res_inc = add_points(model_inc, x, y)

    # Compare results
    println("\nResults Comparison:")

    # Show sample around spike area
    println("\nSample around spike (indices 198-208):")
    println("Index\tRaw\t\tTrue\t\tFull\t\tIncremental")
    for i = 198:208
        @printf(
            "%d\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\n",
            i,
            y[i],
            y_true[i],
            res_full.y[i],
            res_inc.y[i]
        )
    end

    # Calculate overall statistics
    mse_full = sum((res_full.y .- y_true) .^ 2) / n_points
    mse_inc = sum((res_inc.y .- y_true) .^ 2) / n_points

    println("\nMean Squared Error vs True Signal:")
    @printf(" - Full Update:        %.4f\n", mse_full)
    @printf(" - Incremental Update: %.4f\n", mse_inc)

    println("\n=== Online Smoothing Example Complete ===")
end

main()
