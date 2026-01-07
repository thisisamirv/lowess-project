#!/usr/bin/env Rscript
# =============================================================================
# fastLowess Streaming Smoothing Examples
#
# This example demonstrates streaming LOWESS smoothing for large datasets:
# - Basic chunked processing
# - Different chunk sizes and overlap strategies
# - Processing very large datasets
# - Parallel vs sequential execution
#
# The streaming adapter (smooth_streaming function) is designed for:
# - Large datasets (>100K points) that don't fit in memory
# - Batch processing pipelines
# - File-based data processing
# - ETL (Extract, Transform, Load) workflows
# =============================================================================

library(rfastlowess)

main <- function() {
    cat(strrep("=", 80), "\n")
    cat("fastLowess Streaming Smoothing Examples\n")
    cat(strrep("=", 80), "\n\n")

    example_1_basic_streaming()
    example_2_chunk_comparison()
    example_3_large_dataset()
    example_4_parallel_comparison()
}

# =============================================================================
# Example 1: Basic Streaming Processing
# Demonstrates the fundamental streaming workflow
# =============================================================================
example_1_basic_streaming <- function() {
    cat("Example 1: Basic Streaming Processing\n")
    cat(strrep("-", 80), "\n")

    # Generate test data: y = 2x + 1 with noise
    n <- 100
    x <- as.numeric(0:(n - 1))
    y <- 2.0 * x + 1.0 + sin(x * 0.3) * 2.0

    # Process with streaming adapter
    result <- fastlowess_streaming(
        x, y,
        fraction = 0.5,
        chunk_size = 30L,
        overlap = 10L,
        iterations = 2L,
        weight_function = "tricube",
        robustness_method = "bisquare"
    )

    cat(sprintf("Dataset: %d points\n", n))
    cat("Chunk size: 30, Overlap: 10\n")
    cat(sprintf("Output points: %d\n", length(result$y)))
    cat(sprintf("All points processed: %s\n", length(result$y) == n))
    cat(
        "First 5 smoothed values:",
        paste(round(result$y[1:5], 4), collapse = ", "), "\n\n"
    )
}

# =============================================================================
# Example 2: Chunk Size Comparison
# Shows how different chunk sizes affect processing
# =============================================================================
example_2_chunk_comparison <- function() {
    cat("Example 2: Chunk Size Comparison\n")
    cat(strrep("-", 80), "\n")

    # Generate test data
    n <- 500
    x <- as.numeric(0:(n - 1))
    y <- 2.0 * x + 1.0

    chunk_configs <- list(
        list(chunk_size = 50L, overlap = 10L, description = "Small chunks"),
        list(chunk_size = 100L, overlap = 20L, description = "Medium chunks"),
        list(chunk_size = 200L, overlap = 40L, description = "Large chunks")
    )

    for (config in chunk_configs) {
        start_time <- Sys.time()
        result <- fastlowess_streaming(
            x, y,
            fraction = 0.5,
            chunk_size = config$chunk_size,
            overlap = config$overlap,
            iterations = 2L
        )
        duration <- as.numeric(Sys.time() - start_time, units = "secs")

        cat(sprintf(
            "%s (size: %d, overlap: %d)\n",
            config$description, config$chunk_size, config$overlap
        ))
        cat(sprintf("  Output points: %d\n", length(result$y)))
        cat(sprintf("  Time: %.4fs\n", duration))
    }
    cat("\n")
}

# =============================================================================
# Example 3: Large Dataset Processing
# Demonstrates processing a very large dataset
# =============================================================================
example_3_large_dataset <- function() {
    cat("Example 3: Large Dataset Processing\n")
    cat(strrep("-", 80), "\n")

    n <- 50000 # 50K points
    cat(sprintf("Processing %d data points in streaming mode...\n", n))

    x <- as.numeric(0:(n - 1))
    y <- 2.0 * x + 1.0 + sin(x * 0.01) * 10.0

    start_time <- Sys.time()
    result <- fastlowess_streaming(
        x, y,
        fraction = 0.3,
        chunk_size = 5000L,
        overlap = 500L,
        iterations = 2L,
        parallel = TRUE # Enable parallel execution
    )
    duration <- as.numeric(Sys.time() - start_time, units = "secs")

    cat(sprintf("Processed %d points in %.4fs\n", length(result$y), duration))
    cat("Memory efficiency: Constant (chunk size = 5000 points)\n\n")
}

# =============================================================================
# Example 4: Parallel vs Sequential Comparison
# Compares execution time with and without parallelism
# =============================================================================
example_4_parallel_comparison <- function() {
    cat("Example 4: Parallel vs Sequential Comparison\n")
    cat(strrep("-", 80), "\n")

    n <- 10000
    x <- as.numeric(0:(n - 1))
    y <- sin(x * 0.1) + cos(x * 0.01)

    # Parallel execution
    start_time <- Sys.time()
    result_parallel <- fastlowess_streaming(
        x, y,
        fraction = 0.5,
        chunk_size = 1000L,
        overlap = 100L,
        iterations = 3L,
        parallel = TRUE
    )
    parallel_time <- as.numeric(Sys.time() - start_time, units = "secs")

    # Sequential execution
    start_time <- Sys.time()
    result_sequential <- fastlowess_streaming(
        x, y,
        fraction = 0.5,
        chunk_size = 1000L,
        overlap = 100L,
        iterations = 3L,
        parallel = FALSE
    )
    sequential_time <- as.numeric(Sys.time() - start_time, units = "secs")

    cat(sprintf(
        "Parallel:   %.4fs (%d points)\n",
        parallel_time, length(result_parallel$y)
    ))
    cat(sprintf(
        "Sequential: %.4fs (%d points)\n",
        sequential_time, length(result_sequential$y)
    ))
    if (sequential_time > 0) {
        cat(sprintf("Speedup: %.2fx\n", sequential_time / parallel_time))
    }
    cat("\n")
}

# Run if called directly
if (sys.nframe() == 0) {
    main()
}
