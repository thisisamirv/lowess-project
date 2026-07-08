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
    # Helper: process all points one at a time
    online_smooth <- function(model, x, y) {
        sapply(seq_along(x), function(i) {
            r <- model$add_point(x[[i]], y[[i]])
            if (is.null(r)) y[[i]] else r
        })
    }

    print(model)
    smoothed <- online_smooth(model, x, y)

    cat("Processing data points with sliding window...\n")
    cat("Window capacity: 5\n")
    cat(sprintf("Output points: %d\n", length(smoothed)))
    cat("Smoothed values:", paste(round(smoothed, 4), collapse = ", "), "\n\n")
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
    smoothed <- sapply(seq_along(x), function(i) {
        r <- model$add_point(x[[i]], y[[i]])
        if (is.null(r)) y[[i]] else r
    })

    cat(sprintf("%6s %12s %12s\n", "Hour", "Raw Temp", "Smoothed"))
    cat(strrep("-", 35), "\n")

    # Show first 10 values
    for (i in seq_len(min(10, length(smoothed)))) {
        cat(sprintf("%6.0f %10.2f degC %10.2f degC\n", x[i], y[i], smoothed[i]))
    }

    if (length(smoothed) > 10) {
        cat(sprintf("  ... (%d more rows)\n", length(smoothed) - 10))
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
        result <- model$add_point(x[[length(x)]], y[[length(x)]])
        final_val <- if (is.null(result)) tail(y, 1) else result

        cat(sprintf("Window capacity: %d\n", window_size))
        cat(sprintf("  Final smoothed value: %.4f\n", final_val))
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
    last_smoothed <- NA_real_
    for (i in seq_along(x)) {
        r <- model$add_point(x[[i]], y[[i]])
        if (!is.null(r)) last_smoothed <- r
    }
    duration <- as.numeric(Sys.time() - start_time, units = "secs")

    cat(sprintf("\nProcessed %d points in %.4fs\n", total_points, duration))
    if (!is.na(last_smoothed)) {
        cat(sprintf("Final smoothed value: %.2f\n", last_smoothed))
    }
    cat("Memory usage: Constant (window size = 20 points)\n\n")
}

# Run if called directly
if (sys.nframe() == 0) {
    main()
}
