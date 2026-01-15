#!/usr/bin/env julia
"""
fastlowess Streaming Smoothing Example

This example demonstrates streaming LOWESS smoothing for large datasets:
- Basic chunked processing
- Handling datasets that don't fit in memory
- Parallel execution for extreme speed
"""

using Random
using Printf

# Handle package loading - check if we're already in the fastlowess project
using Pkg
project_name = Pkg.project().name
if project_name != "fastlowess"
    # Not in the fastlowess project, need to develop it
    script_dir = @__DIR__
    julia_pkg_dir = joinpath(dirname(script_dir), "julia")
    if !haskey(Pkg.project().dependencies, "fastlowess")
        Pkg.develop(path = julia_pkg_dir)
    end
end

using fastlowess

function main()
    println("=== fastlowess Streaming Mode Example ===")

    # 1. Generate Very Large Dataset
    # 100,000 points
    n_points = 100_000
    println("Generating large dataset: $n_points points...")
    Random.seed!(42)
    x = collect(range(0, 100, length = n_points))
    y = cos.(x .* 0.1) .+ randn(n_points) .* 0.5

    # 2. Regular Batch Smoothing (for comparison)
    println("Running Batch LOWESS (Parallel)...")
    batch_start = time()
    res_batch = fit(Lowess(fraction = 0.01), x, y)
    batch_time = time() - batch_start
    @printf("Batch took: %.4f seconds\n", batch_time)

    # 3. Streaming Mode
    # Divide the data into chunks of 2,000 for low memory usage
    println("Running Streaming LOWESS (Chunked)...")
    stream_start = time()
    model =
        StreamingLowess(fraction = 0.01, chunk_size = 2000, overlap = 200, parallel = true)
    res_stream = process_chunk(model, x, y)
    append!(res_stream, finalize(model))
    stream_time = time() - stream_start
    @printf("Streaming took: %.4f seconds\n", stream_time)

    # 4. Verify Accuracy
    mse = sum((res_batch.y .- res_stream.y) .^ 2) / length(res_batch.y)
    @printf("Mean Squared Difference (Batch vs Stream): %.2e\n", mse)

    # Show sample of results
    println("\nSample comparison (indices 1000-1005):")
    println("Index\tBatch\t\tStreaming\tDiff")
    for i = 1000:1005
        diff = abs(res_batch.y[i] - res_stream.y[i])
        @printf("%d\t%.6f\t%.6f\t%.6f\n", i, res_batch.y[i], res_stream.y[i], diff)
    end

    println("\n=== Streaming Smoothing Example Complete ===")
end

main()
