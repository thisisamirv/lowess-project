#' @srrstats {G5.0} Tests use standard data patterns (sin, linear, constant).
#' @srrstats {G5.1} Test datasets are reproducible via set.seed().
#' @srrstats {G5.3} Tests verify no NA/NaN in outputs.
#' @srrstats {G5.4, G5.4a} Correctness tests against expected behavior.
#' @srrstats {G5.5} Fixed random seeds in all tests.
#' @srrstats {G5.6, G5.6a} Parameter recovery within tolerance.
#' @srrstats {G5.7} Performance tests (parallel vs serial).
#' @srrstats {G5.8c, G5.8d} Edge cases: minimum points, constant values.
#' @srrstats {G5.9a} Noise susceptibility via rnorm() variations.
#' @srrstats {RE4.10} Residuals tested in Lowess residuals test.
#' @srrstats {RE4.11} Goodness-of-fit tested in diagnostics test.
#' @srrstats {RE5.0} Confidence/prediction intervals tested.
#' @srrstats {RE6.0} Cross-validation tested.
test_that("Lowess basic functionality works", {
    x <- c(1, 2, 3, 4, 5)
    y <- c(2, 4, 6, 8, 10)

    result <- fit(Lowess(fraction = 0.67), as.double(x), as.double(y))

    expect_type(result, "list")
    expect_named(result, c("x", "y", "fraction_used"))
    expect_length(result$x, length(x))
    expect_length(result$y, length(y))
    expect_type(result$x, "double")
    expect_type(result$y, "double")
})

test_that("Lowess handles different fractions", {
    set.seed(42)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, sd = 0.1)

    result_low <- fit(Lowess(fraction = 0.2), as.double(x), as.double(y))
    result_high <- fit(Lowess(fraction = 0.8), as.double(x), as.double(y))

    expect_length(result_low$y, length(y))
    expect_length(result_high$y, length(y))

    # Higher fraction should produce smoother results
    expect_lt(sd(diff(result_high$y)), sd(diff(result_low$y)))
})

test_that("Lowess robustness iterations work", {
    set.seed(123)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, sd = 0.1)

    # Add outliers
    y[c(10, 25, 40)] <- y[c(10, 25, 40)] + 5

    model_no_robust <- Lowess(fraction = 0.3, iterations = 0)
    result_no_robust <- fit(model_no_robust, as.double(x), as.double(y))
    model_robust <- Lowess(fraction = 0.3, iterations = 5)
    result_robust <- fit(model_robust, as.double(x), as.double(y))

    expect_length(result_no_robust$y, length(y))
    expect_length(result_robust$y, length(y))
})

test_that("Lowess confidence intervals work", {
    set.seed(42)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, sd = 0.2)

    model <- Lowess(fraction = 0.5, confidence_intervals = 0.95)
    result <- fit(model, as.double(x), as.double(y))

    expect_true("confidence_lower" %in% names(result))
    expect_true("confidence_upper" %in% names(result))
    expect_length(result$confidence_lower, length(y))
    expect_length(result$confidence_upper, length(y))

    # CI bounds should bracket the fitted values
    expect_lte(max(result$confidence_lower - result$y), 0)
    expect_gte(min(result$confidence_upper - result$y), 0)
})

test_that("Lowess prediction intervals work", {
    set.seed(42)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, sd = 0.2)

    model <- Lowess(fraction = 0.5, prediction_intervals = 0.95)
    result <- fit(model, as.double(x), as.double(y))

    expect_true("prediction_lower" %in% names(result))
    expect_true("prediction_upper" %in% names(result))
    expect_length(result$prediction_lower, length(y))
    expect_length(result$prediction_upper, length(y))

    # PI should be wider than CI
    model_ci <- Lowess(fraction = 0.5, confidence_intervals = 0.95)
    result_ci <- fit(model_ci, as.double(x), as.double(y))
    expect_gt(
        mean(result$prediction_upper - result$prediction_lower),
        mean(result_ci$confidence_upper - result_ci$confidence_lower)
    )
})

test_that("Lowess diagnostics work", {
    set.seed(42)
    x <- seq(0, 10, length.out = 50)
    y <- 2 * x + rnorm(50, sd = 0.5)

    model <- Lowess(fraction = 0.5, return_diagnostics = TRUE)
    result <- fit(model, as.double(x), as.double(y))

    expect_true("diagnostics" %in% names(result))
    expect_type(result$diagnostics, "list")
    expect_true("rmse" %in% names(result$diagnostics))
    expect_true("mae" %in% names(result$diagnostics))
    expect_true("r_squared" %in% names(result$diagnostics))

    #  R2 should be between 0 and 1
    expect_gte(result$diagnostics$r_squared, 0)
    expect_lte(result$diagnostics$r_squared, 1)
})

test_that("Lowess residuals work", {
    set.seed(42)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, sd = 0.1)

    model <- Lowess(fraction = 0.5, return_residuals = TRUE)
    result <- fit(model, as.double(x), as.double(y))

    expect_true("residuals" %in% names(result))
    expect_length(result$residuals, length(y))
    expect_type(result$residuals, "double")
})

test_that("Lowess robustness weights work", {
    set.seed(42)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, sd = 0.1)
    y[25] <- y[25] + 5 # Add outlier

    result <- fit(
        Lowess(
            fraction = 0.5,
            iterations = 3,
            return_robustness_weights = TRUE
        ),
        as.double(x),
        as.double(y)
    )

    expect_true("robustness_weights" %in% names(result))
    expect_length(result$robustness_weights, length(y))

    # Outlier should have lower weight
    expect_lt(result$robustness_weights[25], median(result$robustness_weights))
})

test_that("Lowess return_sorted defaults to original input order", {
    x <- c(3.0, 1.0, 5.0, 2.0, 4.0)
    y <- c(6.0, 2.0, 10.0, 4.0, 8.0)

    result <- fit(Lowess(fraction = 0.7), x, y)

    expect_identical(result$x, x)
})

test_that("Lowess return_sorted = TRUE returns results sorted ascending by x", {
    x <- c(3.0, 1.0, 5.0, 2.0, 4.0)
    y <- c(6.0, 2.0, 10.0, 4.0, 8.0)

    result <- fit(
        Lowess(
            fraction = 0.7,
            return_residuals = TRUE,
            return_robustness_weights = TRUE,
            return_sorted = TRUE
        ),
        x,
        y
    )

    # x must be strictly ascending, and differ from the unsorted input order.
    expect_true(all(diff(result$x) >= 0))
    expect_false(isTRUE(all.equal(result$x, x)))

    # Same (x, y) pairs as the unsorted-order fit, just reordered.
    unsorted_result <- fit(
        Lowess(
            fraction = 0.7,
            return_residuals = TRUE,
            return_robustness_weights = TRUE
        ),
        x,
        y
    )

    sorted_order <- order(result$x)
    unsorted_order <- order(unsorted_result$x)
    # Two independent fits with default parallel = TRUE; Rayon's reduction
    # order can differ minutely, so compare with tolerance, not identity.
    expect_equal(result$y[sorted_order], unsorted_result$y[unsorted_order]) # nolint: expect_identical_linter.

    expect_length(result$residuals, length(x))
    expect_length(result$robustness_weights, length(x))
})

test_that("Lowess cross-validation works", {
    set.seed(42)
    x <- seq(0, 10, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.2)

    result <- fit(
        Lowess(
            cv_fractions = c(0.2, 0.3, 0.5, 0.7),
            cv_method = "kfold",
            cv_k = 5
        ),
        as.double(x),
        as.double(y)
    )

    expect_true("cv_scores" %in% names(result))
    expect_length(result$cv_scores, 4)
    expect_true(result$fraction_used %in% c(0.2, 0.3, 0.5, 0.7))
})

test_that("Lowess handles edge cases", {
    # Minimum data points
    x <- c(1, 2, 3)
    y <- c(1, 2, 3)
    result <- fit(Lowess(fraction = 0.67), as.double(x), as.double(y))
    expect_length(result$y, 3)

    # Constant y values
    x <- 1:10
    y <- rep(5, 10)
    result <- fit(Lowess(fraction = 0.5), as.double(x), as.double(y))
    expect_lt(max(abs(result$y - 5)), 1e-10)
})

test_that("Lowess parallel execution works", {
    set.seed(42)
    x <- seq(0, 10, length.out = 1000)
    y <- sin(x) + rnorm(1000, sd = 0.1)

    model_serial <- Lowess(fraction = 0.3, parallel = FALSE)
    result_serial <- fit(model_serial, as.double(x), as.double(y))
    model_parallel <- Lowess(fraction = 0.3, parallel = TRUE)
    result_parallel <- fit(model_parallel, as.double(x), as.double(y))

    # Results should be nearly identical
    expect_equal(result_serial$y, result_parallel$y, tolerance = 1e-10)
})

# ── custom_weights ────────────────────────────────────────────────────────────

test_that("custom_weights: uniform weights produce same result as no weights", {
    x <- as.double(seq(0, 5, length.out = 20))
    y <- sin(x)
    weights <- rep(1.0, 20)

    result_no_w <- fit(Lowess(fraction = 0.4, iterations = 2), x, y)
    result_unit_w <- fit(
        Lowess(fraction = 0.4, iterations = 2),
        x,
        y,
        custom_weights = weights
    )

    expect_equal(result_no_w$y, result_unit_w$y, tolerance = 1e-10)
})

test_that("custom_weights: zero weight on outlier reduces its influence", {
    x <- as.double(0:9)
    y <- x * 2.0
    y[6] <- 100.0 # outlier at index 6 (1-based)

    result_no_w <- fit(Lowess(fraction = 0.5, iterations = 0), x, y)

    weights <- rep(1.0, 10)
    weights[6] <- 0.0
    result_zero_w <- fit(
        Lowess(fraction = 0.5, iterations = 0),
        x,
        y,
        custom_weights = weights
    )

    true_val <- 5.0 * 2.0 # true value at x=5 (index 6 in 1-based)
    err_no_w <- abs(result_no_w$y[6] - true_val)
    err_zero_w <- abs(result_zero_w$y[6] - true_val)

    expect_lt(err_zero_w, err_no_w)
})

test_that("custom_weights: high weight pulls fit toward spike", {
    x <- as.double(0:14)
    y <- rep(0.0, 15)
    y[8] <- 10.0 # spike at index 8 (1-based)

    weights_high <- rep(1.0, 15)
    weights_high[8] <- 100.0

    result_high <- fit(
        Lowess(fraction = 0.6, iterations = 0),
        x,
        y,
        custom_weights = weights_high
    )
    result_equal <- fit(Lowess(fraction = 0.6, iterations = 0), x, y)

    expect_gt(result_high$y[8], result_equal$y[8])
})

test_that("custom_weights: wrong length raises error", {
    x <- as.double(1:10)
    y <- as.double(1:10)

    expect_error(
        fit(Lowess(fraction = 0.5), x, y, custom_weights = rep(1.0, 7))
    )
})

test_that("custom_weights: negative value raises error", {
    x <- as.double(1:5)
    y <- as.double(1:5)

    expect_error(
        fit(
            Lowess(fraction = 0.5),
            x,
            y,
            custom_weights = c(1.0, -1.0, 1.0, 1.0, 1.0)
        )
    )
})

test_that("missing default (\"error\") raises on NaN", {
    x <- c(1.0, 2.0, 3.0, 4.0, 5.0)
    y <- c(2.0, NaN, 6.0, 8.0, 10.0)

    expect_error(fit(Lowess(fraction = 0.5), x, y))
})

test_that("missing = \"drop\" removes non-finite rows", {
    x <- c(1.0, 2.0, 3.0, 4.0, 5.0)
    y <- c(2.0, NaN, 6.0, 8.0, 10.0)

    result <- fit(Lowess(fraction = 0.5, missing = "drop"), x, y)

    expect_length(result$y, length(x) - 1)
})

test_that("missing: invalid policy raises error", {
    expect_error(
        fit(
            Lowess(fraction = 0.5, missing = "invalid"),
            c(1.0, 2.0),
            c(1.0, 2.0)
        )
    )
})
