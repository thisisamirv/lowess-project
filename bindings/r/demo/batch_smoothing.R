#!/usr/bin/env Rscript
# =============================================================================
# fastLowess Batch Smoothing Examples
#
# This example demonstrates batch LOWESS smoothing features:
# - Basic smoothing with different parameters
# - Robustness iterations for outlier handling
# - Confidence and prediction intervals
# - Diagnostics and cross-validation
#
# The batch adapter (smooth function) is the primary interface for
# processing complete datasets that fit in memory.
# =============================================================================

library(rfastlowess)

main <- function() {
  cat(strrep("=", 80), "\n")
  cat("fastLowess Batch Smoothing Examples\n")
  cat(strrep("=", 80), "\n\n")

  example_1_basic_smoothing()
  example_2_with_intervals()
  example_3_robust_smoothing()
  example_4_cross_validation()
}

# =============================================================================
# Example 1: Basic Smoothing
# Demonstrates the fundamental smoothing workflow
# =============================================================================
example_1_basic_smoothing <- function() {
  cat("Example 1: Basic Smoothing\n")
  cat(strrep("-", 80), "\n")

  # Generate synthetic dataset
  n <- 10000
  x <- as.numeric(0:(n - 1))
  y <- sin(x * 0.1) + cos(x * 0.01)

  start_time <- Sys.time()
  result <- fastlowess(
    x, y,
    fraction = 0.5, # Use 50% of data for each local fit
    iterations = 3L, # 3 robustness iterations
    parallel = TRUE # Enable parallel processing
  )
  duration <- as.numeric(Sys.time() - start_time, units = "secs")

  cat(sprintf("Processed %d points in %.4fs\n", n, duration))
  cat(
    "First 5 smoothed values:",
    paste(round(result$y[1:5], 6), collapse = ", "), "\n\n"
  )
}

# =============================================================================
# Example 2: Confidence and Prediction Intervals
# Demonstrates computing uncertainty intervals
# =============================================================================
example_2_with_intervals <- function() {
  cat("Example 2: Confidence and Prediction Intervals\n")
  cat(strrep("-", 80), "\n")

  x <- c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
  y <- c(2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7)

  result <- fastlowess(
    x, y,
    fraction = 0.5,
    confidence_intervals = 0.95, # 95% confidence intervals
    prediction_intervals = 0.95, # 95% prediction intervals
    parallel = TRUE
  )

  cat(sprintf(
    "%8s %12s %12s %12s\n", "X", "Y_smooth", "CI_Lower", "CI_Upper"
  ))
  cat(strrep("-", 50), "\n")
  for (i in seq_along(x)) {
    ci_lower <- if (!is.null(result$confidence_lower)) {
      result$confidence_lower[i]
    } else {
      0
    }
    ci_upper <- if (!is.null(result$confidence_upper)) {
      result$confidence_upper[i]
    } else {
      0
    }
    cat(sprintf(
      "%8.2f %12.4f %12.4f %12.4f\n",
      x[i], result$y[i], ci_lower, ci_upper
    ))
  }
  cat("\n")
}

# =============================================================================
# Example 3: Robust Smoothing with Outliers
# Demonstrates outlier handling with robustness iterations
# =============================================================================
example_3_robust_smoothing <- function() {
  cat("Example 3: Robust Smoothing with Outliers\n")
  cat(strrep("-", 80), "\n")

  # Data with outliers
  n <- 1000
  x <- as.numeric(0:(n - 1)) * 0.1
  y <- sin(x)
  # Add periodic outliers (every 100th point)
  outlier_indices <- seq(1, n, by = 100)
  y[outlier_indices] <- x[outlier_indices] + 10.0

  methods <- c("bisquare", "huber", "talwar")

  for (method in methods) {
    result <- fastlowess(
      x, y,
      fraction = 0.1,
      iterations = 3L,
      robustness_method = method,
      return_robustness_weights = TRUE,
      parallel = TRUE
    )

    if (!is.null(result$robustness_weights)) {
      outliers <- sum(result$robustness_weights < 0.1)
      cat(sprintf(
        "%s: Identified %d potential outliers (weight < 0.1)\n",
        tools::toTitleCase(method), outliers
      ))
    } else {
      cat(sprintf(
        "%s: Completed (weights not available)\n",
        tools::toTitleCase(method)
      ))
    }
    cat("\n")
  } # nolint: indentation_linter. # nolint
}

# =============================================================================
# Example 4: Cross-Validation for Fraction Selection
# Demonstrates automatic parameter selection
# =============================================================================
example_4_cross_validation <- function() {
  cat(
    "Example 4: Cross-Validation for Fraction Selection\n"
  ) # nolint: indentation_linter.
  cat(strrep("-", 80), "\n")

  # Generate test data
  set.seed(42)
  x <- as.numeric(0:99)
  y <- 2 * x + 1 + rnorm(100) * 5

  fractions <- c(0.2, 0.3, 0.5, 0.7)

  # Run smooth with CV for each fraction and find the best
  cv_scores <- numeric(length(fractions))

  for (i in seq_along(fractions)) {
    result <- fastlowess(
      x, y,
      fraction = fractions[i],
      cv_fractions = fractions,
      cv_method = "kfold",
      cv_k = 5L,
      return_diagnostics = TRUE,
      parallel = TRUE
    )

    # Use the result from the first iteration to get CV scores
    if (!is.null(result$cv_scores)) {
      cv_scores <- result$cv_scores
      break
    }
  }

  # Find optimal fraction (lowest CV score)
  if (any(cv_scores != 0)) {
    best_idx <- which.min(cv_scores)
    optimal_fraction <- fractions[best_idx]
    cat(sprintf("Selected fraction: %.2f\n", optimal_fraction))
    cat("CV scores:", paste(round(cv_scores, 4), collapse = ", "), "\n")
  } else {
    cat("CV scores not available\n")
  }
  cat("\n")
}

# Run if called directly
if (sys.nframe() == 0) {
  main()
}
