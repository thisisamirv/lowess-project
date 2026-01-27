#' Industry-level LOWESS benchmarks for R with JSON output for comparison.
#'
#' Benchmarks are aligned with the Rust benchmarks to enable direct comparison.
#' Results are written to benchmarks/output/r_benchmark.json.
#'
#' Run with: Rscript benchmark.R

library(jsonlite)
library(stats)

# ============================================================================
# Benchmark Result Storage
# ============================================================================

run_benchmark <- function(name, size, func, iterations = 10, warmup = 2) {
    cat(sprintf("Running benchmark: %s (size: %d)\n", name, size))

    # Warmup runs
    for (i in seq_len(warmup)) {
        tryCatch(
            {
                func()
            },
            error = function(e) {
                cat(
                    sprintf(
                        "Benchmark %s failed during warmup: %s\n",
                        name,
                        e$message
                    )
                )
            }
        )
    }

    # Timed runs
    times <- numeric(iterations)
    for (i in seq_len(iterations)) {
        start <- Sys.time()
        tryCatch(
            {
                func()
                end <- Sys.time()
                elapsed <- as.numeric(difftime(end, start, units = "secs"))
                times[i] <- elapsed * 1000 # convert to ms
            },
            error = function(e) {
                cat(sprintf("Benchmark %s failed: %s\n", name, e$message))
            }
        )
    }

    list(
        name = name,
        size = size,
        iterations = iterations,
        mean_time_ms = mean(times),
        std_time_ms = sd(times),
        median_time_ms = median(times),
        min_time_ms = min(times),
        max_time_ms = max(times)
    )
}

# ============================================================================
# Data Generation (Aligned with Rust/Python)
# ============================================================================

generate_sine_data <- function(size, seed = 42) {
    set.seed(seed)
    x <- seq(0, 10, length.out = size)
    y <- sin(x) + rnorm(size, mean = 0, sd = 0.2)
    list(x = x, y = y)
}

generate_outlier_data <- function(size, seed = 42) {
    set.seed(seed)
    x <- seq(0, 10, length.out = size)
    y <- sin(x) + rnorm(size, mean = 0, sd = 0.2)

    n_outliers <- floor(size / 20)
    indices <- sample(seq_len(size), n_outliers)
    y[indices] <- y[indices] + runif(n_outliers, -5, 5)
    list(x = x, y = y)
}

generate_financial_data <- function(size, seed = 42) {
    set.seed(seed)
    x <- seq(0, size - 1)
    y <- numeric(size)
    y[1] <- 100.0
    returns <- rnorm(size - 1, mean = 0.0005, sd = 0.02)
    for (i in 2:size) {
        y[i] <- y[i - 1] * (1 + returns[i - 1])
    }
    list(x = x, y = y)
}

generate_scientific_data <- function(size, seed = 42) {
    set.seed(seed)
    x <- seq(0, 10, length.out = size)
    signal <- exp(-x * 0.3) * cos(x * 2 * pi)
    noise <- rnorm(size, mean = 0, sd = 0.05)
    list(x = x, y = signal + noise)
}

generate_genomic_data <- function(size, seed = 42) {
    set.seed(seed)
    x <- seq(0, size - 1) * 1000.0
    base <- 0.5 + sin(x / 50000.0) * 0.3
    noise <- rnorm(size, mean = 0, sd = 0.1)
    y <- pmin(pmax(base + noise, 0.0), 1.0)
    list(x = x, y = y)
}

generate_clustered_data <- function(size, seed = 42) {
    set.seed(seed)
    i <- seq(0, size - 1)
    x <- (i %/% 100) + (i %% 100) * 1e-6
    y <- sin(x) + rnorm(size, mean = 0, sd = 0.1)
    list(x = x, y = y)
}

generate_high_noise_data <- function(size, seed = 42) {
    set.seed(seed)
    x <- seq(0, 10, length.out = size)
    signal <- sin(x) * 0.5
    noise <- rnorm(size, mean = 0, sd = 2.0)
    list(x = x, y = signal + noise)
}

# ============================================================================
# Benchmark Categories
# ============================================================================

benchmark_scalability <- function(iterations = 10) {
    results <- list()
    sizes <- c(1000, 5000, 10000)

    for (size in sizes) {
        data <- generate_sine_data(size)
        run <- function() {
            lowess(x = data$x, y = data$y, f = 0.1, iter = 3)
        }
        results[[paste0("scale_", size)]] <- run_benchmark(
            paste0("scale_", size), size, run, iterations
        )
    }
    results
}

benchmark_fraction <- function(iterations = 10) {
    results <- list()
    size <- 5000
    fractions <- c(0.05, 0.1, 0.2, 0.3, 0.5, 0.67)
    data <- generate_sine_data(size)

    for (frac in fractions) {
        run <- function() {
            lowess(x = data$x, y = data$y, f = frac, iter = 3)
        }
        results[[paste0("fraction_", frac)]] <- run_benchmark(
            paste0("fraction_", frac), size, run, iterations
        )
    }
    results
}

benchmark_iterations <- function(iterations = 10) {
    results <- list()
    size <- 5000
    iter_values <- c(0, 1, 2, 3, 5, 10)
    data <- generate_outlier_data(size)

    for (it in iter_values) {
        run <- function() {
            lowess(x = data$x, y = data$y, f = 0.2, iter = it)
        }
        results[[paste0("iterations_", it)]] <- run_benchmark(
            paste0("iterations_", it), size, run, iterations
        )
    }
    results
}

benchmark_financial <- function(iterations = 10) {
    results <- list()
    sizes <- c(500, 1000, 5000)

    for (size in sizes) {
        data <- generate_financial_data(size)
        run <- function() {
            lowess(x = data$x, y = data$y, f = 0.1, iter = 2)
        }
        results[[paste0("financial_", size)]] <- run_benchmark(
            paste0("financial_", size), size, run, iterations
        )
    }
    results
}

benchmark_scientific <- function(iterations = 10) {
    results <- list()
    sizes <- c(500, 1000, 5000)

    for (size in sizes) {
        data <- generate_scientific_data(size)
        run <- function() {
            lowess(x = data$x, y = data$y, f = 0.15, iter = 3)
        }
        results[[paste0("scientific_", size)]] <- run_benchmark(
            paste0("scientific_", size), size, run, iterations
        )
    }
    results
}

benchmark_genomic <- function(iterations = 10) {
    results <- list()
    sizes <- c(1000, 5000, 100000)

    for (size in sizes) {
        data <- generate_genomic_data(size)
        run <- function() {
            lowess(x = data$x, y = data$y, f = 0.1, iter = 3)
        }
        results[[paste0("genomic_", size)]] <- run_benchmark(
            paste0("genomic_", size), size, run, iterations
        )
    }
    results
}

benchmark_pathological <- function(iterations = 10) {
    results <- list()
    size <- 5000

    # Clustered
    data_clustered <- generate_clustered_data(size)
    run_clustered <- function() {
        lowess(x = data_clustered$x, y = data_clustered$y, f = 0.3, iter = 2)
    }
    results$clustered <- run_benchmark(
        "clustered", size, run_clustered, iterations
    )

    # High noise
    data_noisy <- generate_high_noise_data(size)
    run_noise <- function() {
        lowess(x = data_noisy$x, y = data_noisy$y, f = 0.5, iter = 5)
    }
    results$high_noise <- run_benchmark(
        "high_noise", size, run_noise, iterations
    )

    # Extreme outliers
    data_outlier <- generate_outlier_data(size)
    run_outliers <- function() {
        lowess(x = data_outlier$x, y = data_outlier$y, f = 0.2, iter = 10)
    }
    results$extreme_outliers <- run_benchmark(
        "extreme_outliers", size, run_outliers, iterations
    )

    # Constant y
    x_const <- seq(size)
    y_const <- rep(5.0, size)
    data_const <- list(x = x_const, y = y_const)
    run_const <- function() {
        lowess(x = data_const$x, y = data_const$y, f = 0.2, iter = 2)
    }
    results$constant_y <- run_benchmark(
        "constant_y", size, run_const, iterations
    )

    results
}

# ============================================================================
# Main Execution
# ============================================================================

main <- function() {
    cat("=============================================================\n")
    cat("R LOWESS BENCHMARK SUITE (Aligned with Python/Rust)\n")
    cat("=============================================================\n")

    iterations <- 10
    all_results <- list()

    all_results$scalability <- unname(benchmark_scalability(iterations))
    all_results$fraction <- unname(benchmark_fraction(iterations))
    all_results$iterations <- unname(benchmark_iterations(iterations))
    all_results$financial <- unname(benchmark_financial(iterations))
    all_results$scientific <- unname(benchmark_scientific(iterations))
    all_results$genomic <- unname(benchmark_genomic(iterations))
    all_results$pathological <- unname(benchmark_pathological(iterations))

    # Move to output directory
    out_dir <- "output"
    if (!dir.exists(out_dir)) {
        dir.create(out_dir, recursive = TRUE)
    }

    out_path <- file.path(out_dir, "r_benchmark.json")
    write_json(all_results, out_path, auto_unbox = TRUE, pretty = TRUE)

    cat("\n============================================================\n")
    cat(sprintf("Results saved to %s\n", out_path))
    cat("============================================================\n")
}

if (interactive() == FALSE) {
    main()
}
