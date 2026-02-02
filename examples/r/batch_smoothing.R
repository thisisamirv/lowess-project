#!/usr/bin/env Rscript
# =============================================================================
# rfastlowess Batch Smoothing Example
#
# This example demonstrates batch LOWESS smoothing features:
# - Basic smoothing with different parameters
# - Robustness iterations for outlier handling
# - Confidence and prediction intervals
# - Diagnostics and cross-validation
# - S3 methods for print and plot
#
# The Lowess class is the primary interface for
# processing complete datasets that fit in memory.
# =============================================================================

library(rfastlowess)

generate_sample_data <- function(n_points = 1000) {
    # Generate complex sample data with a trend, seasonality, and outliers.
    set.seed(42)
    x <- seq(0, 50, length.out = n_points)

    # Trend plus Seasonality
    y_true <- 0.5 * x + 5 * sin(x * 0.5)

    # Gaussian noise
    y <- y_true + rnorm(n_points, mean = 0, sd = 1.5)

    # Add significant outliers (10% of data)
    n_outliers <- round(n_points * 0.1)
    outlier_indices <- sample(seq_len(n_points), n_outliers)
    # Random magnitude 10-20, random sign
    dataset_outliers <- runif(n_outliers, 10, 20) *
        sample(c(-1, 1), n_outliers, replace = TRUE)
    y[outlier_indices] <- y[outlier_indices] + dataset_outliers

    list(x = x, y = y, y_true = y_true)
}

main <- function() {
    cat(strrep("=", 80), "\n")
    cat("rfastlowess Batch Smoothing Example\n")
    cat(strrep("=", 80), "\n\n")

    # 1. Generate Data
    data <- generate_sample_data(1000)
    x <- data$x
    y <- data$y
    cat(sprintf("Generated %d data points with outliers.\n", length(x)))

    # 2. Basic Smoothing (Default parameters)
    cat("Running basic smoothing...\n")
    # Use a smaller fraction (0.05) to capture the sine wave seasonality
    basic_model <- Lowess(iterations = 0L, fraction = 0.05)
    print(basic_model)
    basic_res <- basic_model$fit(x, y)
    print(basic_res)

    # 3. Robust Smoothing (IRLS)
    cat("Running robust smoothing (3 iterations)...\n")
    robust_model <- Lowess(
        fraction = 0.05,
        iterations = 3L,
        robustness_method = "bisquare",
        return_robustness_weights = TRUE
    )
    print(robust_model)
    robust_res <- robust_model$fit(x, y)
    print(robust_res)

    # 4. Uncertainty Quantification
    cat("Computing confidence and prediction intervals...\n")
    res_intervals <- Lowess(
        fraction = 0.05,
        confidence_intervals = 0.95,
        prediction_intervals = 0.95,
        return_diagnostics = TRUE
    )$fit(x, y)

    # 5. Cross-Validation for optimal fraction
    cat("Running cross-validation to find optimal fraction...\n")
    cv_fractions <- c(0.05, 0.1, 0.2, 0.4)
    res_cv <- Lowess(
        cv_fractions = cv_fractions,
        cv_method = "kfold",
        cv_k = 5L
    )$fit(x, y)

    if (!is.null(res_cv$fraction_used)) {
        cat(sprintf("Optimal fraction found: %.2f\n", res_cv$fraction_used))
    }

    # Diagnostics Printout
    if (!is.null(res_intervals$diagnostics)) {
        diag <- res_intervals$diagnostics
        cat("\nFit Statistics (Intervals Model):\n")
        # Handle potential list or S3 object return structure
        r2 <- if (!is.null(diag$r_squared)) diag$r_squared else NA
        rmse <- if (!is.null(diag$rmse)) diag$rmse else NA
        mae <- if (!is.null(diag$mae)) diag$mae else NA

        cat(sprintf(" - R^2:   %.4f\n", r2))
        cat(sprintf(" - RMSE: %.4f\n", rmse))
        cat(sprintf(" - MAE:  %.4f\n", mae))
    }

    # 6. Boundary Policy Comparison
    cat("\nDemonstrating boundary policy effects on linear data...\n")
    xl <- seq(0, 10, length.out = 50)
    yl <- 2 * xl + 1

    # Compare policies
    r_ext <- Lowess(fraction = 0.6, boundary_policy = "extend")$fit(xl, yl)
    r_ref <- Lowess(fraction = 0.6, boundary_policy = "reflect")$fit(xl, yl)
    r_zr <- Lowess(fraction = 0.6, boundary_policy = "zero")$fit(xl, yl)

    cat("Boundary policy comparison:\n")
    cat(sprintf(
        " - Extend (Default): first=%.2f, last=%.2f\n",
        r_ext$y[1], r_ext$y[length(r_ext$y)]
    ))
    cat(sprintf(
        " - Reflect:          first=%.2f, last=%.2f\n",
        r_ref$y[1], r_ref$y[length(r_ref$y)]
    ))
    cat(sprintf(
        " - Zero:             first=%.2f, last=%.2f\n",
        r_zr$y[1], r_zr$y[length(r_zr$y)]
    ))

    cat("\n=== Batch Smoothing Example Complete ===\n")
}

# Run if called directly
if (sys.nframe() == 0) {
    main()
}
