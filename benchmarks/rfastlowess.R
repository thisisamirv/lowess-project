#' Benchmarks for the rfastlowess R package with JSON output.
#'
#' Scenarios mirror stats::lowess so results can be compared
#' directly via benchmarks/compare.py.
#' Results are written to benchmarks/output/rfastlowess_serial.json or
#' benchmarks/output/rfastlowess_parallel.json depending on the mode flag.
#'
#' Run with:
#'   Rscript benchmarks/rfastlowess.R --serial
#'   Rscript benchmarks/rfastlowess.R --parallel

library(jsonlite)
library(rfastlowess)

# ============================================================================
# Benchmark Runner
# ============================================================================

run_benchmark <- function(name, size, func, n_iter = 10, warmup = 2) {
    cat(sprintf("Running benchmark: %s (size: %d)\n", name, size))

    for (i in seq_len(warmup)) {
        tryCatch(func(), error = function(e) {
            cat(sprintf("  Warmup failed for %s: %s\n", name, e$message))
        })
    }

    times <- numeric(n_iter)
    last_output <- NULL
    for (i in seq_len(n_iter)) {
        start <- Sys.time()
        tryCatch(
            {
                val <- func()
                times[i] <- as.numeric(difftime(Sys.time(), start, units = "secs")) * 1000
                last_output <- val
            },
            error = function(e) {
                cat(sprintf("  Run %d failed for %s: %s\n", i, name, e$message))
            }
        )
    }

    list(
        name            = name,
        size            = size,
        iterations      = n_iter,
        mean_time_ms    = mean(times),
        std_time_ms     = sd(times),
        median_time_ms  = median(times),
        min_time_ms     = min(times),
        max_time_ms     = max(times),
        fitted          = if (!is.null(last_output)) as.numeric(last_output$y) else NULL
    )
}

# ============================================================================
# Data Generation (aligned with stats_lowess.R)
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
    idx <- seq(0, size - 1)
    x <- (idx %/% 100) + (idx %% 100) * 1e-6
    y <- sin(x) + rnorm(size, mean = 0, sd = 0.1)
    list(x = x, y = y)
}

generate_high_noise_data <- function(size, seed = 42) {
    set.seed(seed)
    x <- seq(0, 10, length.out = size)
    y <- sin(x) * 0.5 + rnorm(size, mean = 0, sd = 2.0)
    list(x = x, y = y)
}

# ============================================================================
# Benchmark Categories
# ============================================================================

make_model <- function(fraction, iterations) {
    Lowess(
        fraction        = fraction,
        iterations      = iterations,
        parallel        = isTRUE(getOption("rfastlowess.parallel")),
        boundary_policy = "noboundary",
        scaling_method  = "mar"
    )
}

benchmark_scalability <- function(n_iter = 10) {
    results <- list()
    for (size in c(1000, 5000, 10000)) {
        d <- generate_sine_data(size)
        model <- make_model(0.1, 3)
        results[[paste0("scale_", size)]] <- run_benchmark(
            paste0("scale_", size), size,
            function() fit(model, d$x, d$y),
            n_iter
        )
    }
    results
}

benchmark_fraction <- function(n_iter = 10) {
    size <- 5000
    d <- generate_sine_data(size)
    fracs <- c(0.05, 0.1, 0.2, 0.3, 0.5, 0.67)
    results <- lapply(fracs, function(frac) {
        model <- make_model(frac, 3)
        run_benchmark(
            paste0("fraction_", frac), size,
            function() fit(model, d$x, d$y),
            n_iter
        )
    })
    names(results) <- paste0("fraction_", fracs)
    results
}

benchmark_iterations <- function(n_iter = 10) {
    size <- 5000
    d <- generate_outlier_data(size)
    iter_values <- c(0, 1, 2, 3, 5, 10)
    results <- lapply(iter_values, function(it) {
        model <- make_model(0.2, it)
        run_benchmark(
            paste0("iterations_", it), size,
            function() fit(model, d$x, d$y),
            n_iter
        )
    })
    names(results) <- paste0("iterations_", iter_values)
    results
}

benchmark_financial <- function(n_iter = 10) {
    results <- list()
    for (size in c(500, 1000, 5000)) {
        d <- generate_financial_data(size)
        model <- make_model(0.1, 2)
        results[[paste0("financial_", size)]] <- run_benchmark(
            paste0("financial_", size), size,
            function() fit(model, d$x, d$y),
            n_iter
        )
    }
    results
}

benchmark_scientific <- function(n_iter = 10) {
    results <- list()
    for (size in c(500, 1000, 5000)) {
        d <- generate_scientific_data(size)
        model <- make_model(0.15, 3)
        results[[paste0("scientific_", size)]] <- run_benchmark(
            paste0("scientific_", size), size,
            function() fit(model, d$x, d$y),
            n_iter
        )
    }
    results
}

benchmark_genomic <- function(n_iter = 10) {
    results <- list()
    for (size in c(1000, 5000, 100000)) {
        d <- generate_genomic_data(size)
        model <- make_model(0.1, 3)
        size_str <- format(size, scientific = FALSE, trim = TRUE)
        results[[paste0("genomic_", size_str)]] <- run_benchmark(
            paste0("genomic_", size_str), size,
            function() fit(model, d$x, d$y),
            n_iter
        )
    }
    results
}

benchmark_pathological <- function(n_iter = 10) {
    size <- 5000
    results <- list()

    d <- generate_clustered_data(size)
    model <- make_model(0.3, 2)
    results$clustered <- run_benchmark(
        "clustered", size, function() fit(model, d$x, d$y), n_iter
    )

    d <- generate_high_noise_data(size)
    model <- make_model(0.5, 5)
    results$high_noise <- run_benchmark(
        "high_noise", size, function() fit(model, d$x, d$y), n_iter
    )

    d <- generate_outlier_data(size)
    model <- make_model(0.2, 10)
    results$extreme_outliers <- run_benchmark(
        "extreme_outliers", size, function() fit(model, d$x, d$y), n_iter
    )

    xk <- as.numeric(seq_len(size))
    yk <- rep(5.0, size)
    model <- make_model(0.2, 2)
    results$constant_y <- run_benchmark(
        "constant_y", size, function() fit(model, xk, yk), n_iter
    )

    results
}

benchmark_large <- function(n_iter = 3) {
    # delta = 0 matches stats_lowess.R's large benchmark (which
    # disables stats::lowess's interpolation shortcut); at n = 50000 this
    # reliably takes >= 5s for stats::lowess and gives a fair comparison.
    size <- 50000
    d <- generate_sine_data(size)
    model <- Lowess(
        fraction        = 0.1,
        iterations      = 3,
        delta           = 0,
        parallel        = isTRUE(getOption("rfastlowess.parallel")),
        boundary_policy = "noboundary",
        scaling_method  = "mar"
    )
    results <- list(large_delta_0 = run_benchmark(
        "large_delta_0", size, function() fit(model, d$x, d$y), n_iter,
        warmup = 1
    ))

    # Same workload as above but with delta left at its default (auto =
    # 1/100th of the x range = 0.1 here), showing how much the
    # interpolation shortcut speeds things up at scale.
    model_delta_default <- Lowess(
        fraction        = 0.1,
        iterations      = 3,
        parallel        = isTRUE(getOption("rfastlowess.parallel")),
        boundary_policy = "noboundary",
        scaling_method  = "mar"
    )
    results$large_delta_0.1 <- run_benchmark(
        "large_delta_0.1", size,
        function() fit(model_delta_default, d$x, d$y), n_iter,
        warmup = 1
    )

    # More robustness iterations at the same scale (still delta = 0).
    model_high_iter <- Lowess(
        fraction        = 0.1,
        iterations      = 10,
        delta           = 0,
        parallel        = isTRUE(getOption("rfastlowess.parallel")),
        boundary_policy = "noboundary",
        scaling_method  = "mar"
    )
    results$large_high_iter <- run_benchmark(
        "large_high_iter", size,
        function() fit(model_high_iter, d$x, d$y), n_iter,
        warmup = 1
    )

    # Larger fraction (wider local window) at a smaller size, since the
    # per-point cost grows with fraction * n and 0.67 * 50000 is too slow.
    size_frac <- 20000
    d_frac <- generate_sine_data(size_frac)
    model_high_frac <- Lowess(
        fraction        = 0.67,
        iterations      = 3,
        delta           = 0,
        parallel        = isTRUE(getOption("rfastlowess.parallel")),
        boundary_policy = "noboundary",
        scaling_method  = "mar"
    )
    results$large_high_fraction <- run_benchmark(
        "large_high_fraction", size_frac,
        function() fit(model_high_frac, d_frac$x, d_frac$y), n_iter,
        warmup = 1
    )

    results
}

# ============================================================================
# Main
# ============================================================================

main <- function() {
    trail_args <- commandArgs(trailingOnly = TRUE)
    if ("--serial" %in% trail_args) {
        use_parallel <- FALSE
        mode_label <- "serial"
    } else if ("--parallel" %in% trail_args) {
        use_parallel <- TRUE
        mode_label <- "parallel"
    } else {
        stop("Usage: Rscript rfastlowess.R --serial | --parallel\n", call. = FALSE)
    }
    options(rfastlowess.parallel = use_parallel)

    cat("=============================================================\n")
    cat(sprintf("rfastlowess BENCHMARK SUITE [%s]\n", toupper(mode_label)))
    cat("=============================================================\n")

    n_iter <- 10
    all_results <- list()

    all_results$scalability <- unname(benchmark_scalability(n_iter))
    all_results$fraction <- unname(benchmark_fraction(n_iter))
    all_results$iterations <- unname(benchmark_iterations(n_iter))
    all_results$financial <- unname(benchmark_financial(n_iter))
    all_results$scientific <- unname(benchmark_scientific(n_iter))
    all_results$genomic <- unname(benchmark_genomic(n_iter))
    all_results$pathological <- unname(benchmark_pathological(n_iter))
    all_results$large <- unname(benchmark_large())

    all_args <- commandArgs(trailingOnly = FALSE)
    file_flag <- grep("--file=", all_args, value = TRUE)
    if (length(file_flag) > 0) {
        script_dir <- dirname(normalizePath(sub("--file=", "", file_flag)))
    } else {
        script_dir <- getwd()
    }
    out_dir <- file.path(script_dir, "output")
    if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

    out_path <- file.path(out_dir, sprintf("rfastlowess_%s.json", mode_label))
    write_json(all_results, out_path, auto_unbox = TRUE, pretty = TRUE)

    cat("\n============================================================\n")
    cat(sprintf("Results saved to %s\n", out_path))
    cat("============================================================\n")
}

if (!interactive()) {
    main()
}
