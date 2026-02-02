#!/usr/bin/env Rscript
# =============================================================================
# rfastlowess Online Smoothing Examples
#
# This example demonstrates online LOWESS smoothing for real-time data:
# - Basic incremental processing with streaming data
# - Real-time sensor data smoothing
# - Different window sizes and their effects
# - Memory-bounded processing
#
# The online adapter (smooth_online function) is designed for:
# - Real-time data streams
# - Memory-constrained environments
# - Sensor data processing
# - Incremental updates without reprocessing entire dataset
# =============================================================================

library(rfastlowess)

main <- function() {
    cat(strrep("=", 80), "\n")
    cat("rfastlowess Online Smoothing Examples\n")
    cat(strrep("=", 80), "\n\n")

    example_1_basic_online()
    example_2_sensor_simulation()
    example_3_window_comparison()
    example_4_memory_bounded()
}

# =============================================================================
# Example 1: Basic Online Processing
# Demonstrates incremental data processing
# =============================================================================
example_1_basic_online <- function() {
    cat("Example 1: Basic Online Processing\n")
    cat(strrep("-", 80), "\n")

    # Simulate streaming data: y = 2x + 1 with small noise
    x <- c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
    y <- c(3.1, 5.0, 7.2, 8.9, 11.1, 13.0, 15.2, 16.8, 19.1, 21.0)

    # Process all at once with online adapter
    model <- OnlineLowess(
        fraction = 0.5,
        window_capacity = 5L,
        min_points = 3L,
        iterations = 2L,
        weight_function = "tricube",
        robustness_method = "bisquare"
    )
    print(model)
    result <- model$add_points(x, y)

    cat("Processing data points with sliding window...\n")
    cat("Window capacity: 5\n")
    cat(sprintf("Output points: %d\n", length(result$y)))
    cat("Smoothed values:", paste(round(result$y, 4), collapse = ", "), "\n\n")
}

# =============================================================================
# Example 2: Real-Time Sensor Data Simulation
# Simulates processing temperature sensor readings
# =============================================================================
example_2_sensor_simulation <- function() {
    cat("Example 2: Real-Time Sensor Data Simulation\n")
    cat(strrep("-", 80), "\n")
    cat("Simulating temperature sensor readings with noise...\n\n")

    # Simulate temperature sensor: base temp 20.0 degC with daily cycle + noise
    n <- 24 # 24 hours
    x <- as.numeric(0:(n - 1))
    base_temp <- 20.0
    daily_cycle <- 5.0 * sin(x * pi / 12.0)
    noise <- ((0:(n - 1)) * 7) %% 11 * 0.3 - 1.5
    y <- base_temp + daily_cycle + noise

    model <- OnlineLowess(
        fraction = 0.4,
        window_capacity = 12L, # Half-day window
        min_points = 3L,
        iterations = 2L
    )
    print(model)
    result <- model$add_points(x, y)

    cat(sprintf("%6s %12s %12s\n", "Hour", "Raw Temp", "Smoothed"))
    cat(strrep("-", 35), "\n")

    # Show first 10 values
    for (i in seq_len(min(10, length(result$y)))) {
        cat(sprintf("%6.0f %10.2f degC %10.2f degC\n", x[i], y[i], result$y[i]))
    }

    if (length(result$y) > 10) {
        cat(sprintf("  ... (%d more rows)\n", length(result$y) - 10))
    }
    cat("\n")
}

# =============================================================================
# Example 3: Window Size Comparison
# Shows how different window sizes affect smoothing behavior
# =============================================================================
example_3_window_comparison <- function() {
    cat("Example 3: Window Size Comparison\n")
    cat(strrep("-", 80), "\n")

    # Generate test data with some variation
    x <- as.numeric(1:50)
    y <- 2.0 * x + sin(x * 0.5) * 3.0

    window_sizes <- c(5L, 10L, 20L)

    for (window_size in window_sizes) {
        model <- OnlineLowess(
            fraction = 0.5,
            window_capacity = window_size,
            min_points = 3L,
            iterations = 2L
        )
        result <- model$add_points(x, y)

        cat(sprintf("Window capacity: %d\n", window_size))
        cat(sprintf("  Output points: %d\n", length(result$y)))
        if (length(result$y) >= 5) {
            last_5 <- tail(result$y, 5)
            cat(
                "  Last 5 smoothed:",
                paste(round(last_5, 4), collapse = ", "), "\n"
            )
        } else {
            cat(
                "  Smoothed values:",
                paste(round(result$y, 4), collapse = ", "), "\n"
            )
        }
    }
    cat("\n")
}

# =============================================================================
# Example 4: Memory-Bounded Processing
# Demonstrates efficient processing for resource-constrained systems
# =============================================================================
example_4_memory_bounded <- function() {
    cat("Example 4: Memory-Bounded Processing\n")
    cat(strrep("-", 80), "\n")

    # Simulate a long data stream
    total_points <- 10000
    cat(sprintf(
        "Processing %d points with minimal memory footprint...\n",
        total_points
    ))

    x <- as.numeric(0:(total_points - 1))
    y <- 2.0 * x + sin(x * 0.1) * 5.0 +
        ((0:(total_points - 1)) %% 7 - 3.0) * 0.5

    start_time <- Sys.time()
    model <- OnlineLowess(
        fraction = 0.3,
        window_capacity = 20L, # Small window = low memory usage
        min_points = 3L,
        iterations = 1L,
        parallel = FALSE # Sequential for low latency
    )
    result <- model$add_points(x, y)
    duration <- as.numeric(Sys.time() - start_time, units = "secs")

    cat(sprintf("\nProcessed %d points in %.4fs\n", length(result$y), duration))
    if (length(result$y) > 0) {
        cat(sprintf("Final smoothed value: %.2f\n", tail(result$y, 1)))
    }
    cat("Memory usage: Constant (window size = 20 points)\n\n")
}

# Run if called directly
if (sys.nframe() == 0) {
    main()
}
