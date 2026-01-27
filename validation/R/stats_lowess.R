#!/usr/bin/env Rscript
# R LOWESS Validation Script
# Generates reference outputs for validation using R's lowess() function
# (which implements Cleveland's original Fortran algorithm).
#
# NOTE: R's lowess() is strictly Univariate and mostly Linear (degree 1).
# It uses 'f' for fraction/span and 'iter' for robustness iterations.

library(jsonlite)

OUTPUT_DIR <- "output/r/"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

run_scenario <- function(
    name,
    x,
    y,
    frac,
    iter,
    delta = NULL,
    notes = "",
    ...
) {
    cat(sprintf("Running scenario: %s\n", name))

    # R's lowess default delta is 0.01 * diff(range(x)).
    # If we want "direct" computation (exact), we set delta = 0.
    
    args <- list(...)
    is_direct <- FALSE
    if (!is.null(args$surface) && args$surface == "direct") {
        is_direct <- TRUE
    }
    
    # If delta is not provided explicitly, check if we want direct
    val_delta <- if (!is.null(delta)) {
        delta 
    } else if (is_direct) {
        0.0
    } else {
        # Use R's default (NULL lets lowess decide, typically 0.01 * range)
        NULL
    }

    # Run lowess
    # lowess(x, y, f, iter, delta)
    # Returns list(x, y) where y is smoothed values.
    # Note: lowess sorts x. Our generators produce sorted x, so output order matches.
    if (is.null(val_delta)) {
        fit <- lowess(x, y, f = frac, iter = iter)
    } else {
        fit <- lowess(x, y, f = frac, iter = iter, delta = val_delta)
    }
    
    fitted <- fit$y # These are the smoothed values

    # Create output structure
    data <- list(
        name = name,
        notes = notes,
        input = list(
            x = x,
            y = y
        ),
        params = list(
            fraction = frac,
            degree = 1, # lowess is effectively degree 1
            iterations = iter,
            delta = val_delta,
            extra = list(...)
        ),
        result = list(
            fitted = fitted
        )
    )

    # Save to JSON
    path <- file.path(OUTPUT_DIR, paste0(name, ".json"))
    write_json(data, path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

generate_data <- function(
    n = 100,
    kind = "linear",
    noise = 0.0,
    range_min = 0.0,
    range_max = 1.0,
    outlier_ratio = 0.0
) {
    # Fixed seed for reproducibility
    set.seed(42)

    x <- seq(range_min, range_max, length.out = n)

    if (kind == "linear") {
        y <- 2 * x + 1
    } else if (kind == "quadratic") {
        y <- x^2
    } else if (kind == "sine") {
        y <- sin(4 * x)
    } else if (kind == "step") {
        y <- ifelse(x < (range_min + range_max) / 2, 0.0, 1.0)
    } else if (kind == "constant") {
        y <- rep(5.0, n)
    } else {
        y <- x
    }

    # Add noise
    if (noise > 0) {
        y <- y + rnorm(n, 0, noise)
    }

    # Add outliers
    if (outlier_ratio > 0) {
        n_out <- as.integer(n * outlier_ratio)
        indices <- sample(n, n_out, replace = FALSE)
        y[indices] <- y[indices] + 10.0 # Significant outlier
    }

    list(x = x, y = y)
}

main <- function() {
    # 1. Tiny Linear
    data <- generate_data(n = 10, kind = "linear")
    run_scenario("01_tiny_linear", data$x, data$y, frac = 0.8, iter = 0)

    # 3. Sine Standard
    data <- generate_data(n = 100, kind = "sine", noise = 0.1)
    run_scenario("03_sine_standard", data$x, data$y, frac = 0.3, iter = 0)

    # 4. Sine Robust
    data <- generate_data(n = 100, kind = "sine", outlier_ratio = 0.05)
    run_scenario("04_sine_robust", data$x, data$y, frac = 0.3, iter = 4)

    # 6. Large scale
    data <- generate_data(n = 500, kind = "sine")
    run_scenario("06_large_scale", data$x, data$y, frac = 0.1, iter = 0)

    # 7. High Smoothness
    data <- generate_data(n = 100, kind = "linear", noise = 0.5)
    run_scenario("07_high_smoothness", data$x, data$y, frac = 0.9, iter = 0)

    # 8. Low Smoothness
    data <- generate_data(n = 100, kind = "sine")
    run_scenario("08_low_smoothness", data$x, data$y,
        frac = 0.05, iter = 0,
        surface = "direct"
    )

    # 10. Constant Function
    data <- generate_data(n = 50, kind = "constant")
    run_scenario("10_constant", data$x, data$y, frac = 0.5, iter = 0)

    # 11. Step Function
    data <- generate_data(n = 100, kind = "step")
    run_scenario("11_step_func", data$x, data$y, frac = 0.4, iter = 0)

    # 12. End-effects Left
    data <- generate_data(n = 50, kind = "linear", noise = 0.1)
    run_scenario("12_end_effects_left", data$x, data$y,
        frac = 0.3, iter = 0,
        notes = "Check left boundary"
    )

    # 13. End-effects Right (same data, just naming)
    run_scenario("13_end_effects_right", data$x, data$y,
        frac = 0.3, iter = 0,
        notes = "Check right boundary"
    )

    # 14. Sparse Data
    data <- generate_data(
        n = 20,
        range_max = 100.0,
        kind = "linear",
        noise = 1.0
    )
    run_scenario("14_sparse_data", data$x, data$y, frac = 0.6, iter = 0)

    # 15. Dense Data
    data <- generate_data(n = 1000, kind = "sine", noise = 0.1)
    run_scenario("15_dense_data", data$x, data$y,
        frac = 0.01, iter = 0,
        surface = "direct"
    )

    # 18. Iter 2 Check
    data <- generate_data(n = 100, kind = "sine", outlier_ratio = 0.05)
    run_scenario("18_iter_2", data$x, data$y, frac = 0.4, iter = 2)

    # 19. Interpolate Exact
    data <- generate_data(n = 50, kind = "linear")
    run_scenario("19_interpolate_exact", data$x, data$y, frac = 0.5, iter = 0)

    # 20. Zero Variance
    data <- generate_data(n = 10, kind = "constant") # all 5.0
    run_scenario("20_zero_variance", data$x, data$y, frac = 0.5, iter = 0)

    cat("\nAll supported lowess scenarios completed successfully!\n")
}

# Run main function
main()
