# Shared helper for comparing this package's output against base R's
# `stats::lowess`. Used by both the fixed scenarios in test-validation.R and
# the randomized property-based fuzzing in test-property-lowess.R.

#' Assert that this package reproduces `stats::lowess` on a fixed dataset.
#'
#' Pins R's defaults: no boundary padding (`"noboundary"`) and MAR residual
#' scaling (`"mar"`). `x` is sorted ascending first, as `stats::lowess`
#' assumes.
#'
#' @param direct If `TRUE`, request an exact (non-interpolated) surface by
#'   setting `delta = 0` on both sides; otherwise each side uses its default
#'   (1% of the x-range).
#' @param tolerance Relative tolerance; observed agreement is ~1e-15.
#' @noRd
expect_matches_stats_lowess <- function(
    x,
    y,
    fraction,
    iterations,
    direct = FALSE,
    tolerance = 1e-10
) {
    ord <- order(x)
    x <- as.double(x[ord])
    y <- as.double(y[ord])

    delta <- if (direct) 0.0 else NULL

    reference <- if (is.null(delta)) {
        stats::lowess(x, y, f = fraction, iter = iterations)
    } else {
        stats::lowess(x, y, f = fraction, iter = iterations, delta = delta)
    }

    model <- Lowess(
        fraction = fraction,
        iterations = as.integer(iterations),
        delta = delta,
        boundary_policy = "noboundary",
        scaling_method = "mar",
        zero_weight_fallback = "return_original"
    )
    result <- fit(model, x, y)

    expect_equal(result$y, reference$y, tolerance = tolerance)
    expect_identical(result$fraction_used, fraction)

    invisible(result)
}

#' Assert that `outputs = "sorted"` reproduces `stats::lowess` output order.
#'
#' Unlike `expect_matches_stats_lowess()`, this intentionally passes unsorted
#' inputs through to `Lowess()` and checks both `x` and `y` against
#' `stats::lowess()`'s always-sorted return value.
#' @noRd
expect_stats_lowess_sorted <- function(
    x,
    y,
    fraction,
    iterations,
    direct = FALSE,
    tolerance = 1e-10
) {
    x <- as.double(x)
    y <- as.double(y)

    delta <- if (direct) 0.0 else NULL

    reference <- if (is.null(delta)) {
        stats::lowess(x, y, f = fraction, iter = iterations)
    } else {
        stats::lowess(x, y, f = fraction, iter = iterations, delta = delta)
    }

    model <- Lowess(
        fraction = fraction,
        iterations = as.integer(iterations),
        delta = delta,
        boundary_policy = "noboundary",
        scaling_method = "mar",
        zero_weight_fallback = "return_original",
        outputs = "sorted"
    )
    result <- fit(model, x, y)

    expect_equal(result$x, reference$x, tolerance = tolerance)
    expect_equal(result$y, reference$y, tolerance = tolerance)
    expect_identical(result$fraction_used, fraction)

    invisible(result)
}

#' Check `Lowess()` against `stats::lowess()` without testthat expectations.
#'
#' `quickcheck` treats signalled `testthat` success conditions as falsifying in
#' some shrink paths, so property tests use this base-R checker instead.
#' @noRd
check_stats_lowess <- function(
    x,
    y,
    fraction,
    iterations,
    sorted = FALSE,
    tolerance = 1e-10
) {
    if (sorted) {
        x_fit <- as.double(x)
        y_fit <- as.double(y)
    } else {
        ord <- order(x)
        x_fit <- as.double(x[ord])
        y_fit <- as.double(y[ord])
    }

    reference <- stats::lowess(x_fit, y_fit, f = fraction, iter = iterations)
    model <- Lowess(
        fraction = fraction,
        iterations = as.integer(iterations),
        boundary_policy = "noboundary",
        scaling_method = "mar",
        zero_weight_fallback = "return_original",
        outputs = if (sorted) "sorted" else NULL
    )
    result <- fit(model, x_fit, y_fit)

    if (
        sorted &&
            !isTRUE(all.equal(result$x, reference$x, tolerance = tolerance))
    ) {
        stop("x does not match stats::lowess sorted output", call. = FALSE)
    }
    if (!isTRUE(all.equal(result$y, reference$y, tolerance = tolerance))) {
        max_diff <- max(abs(result$y - reference$y))
        stop(
            sprintf(
                "y does not match stats::lowess output (max abs diff: %.17g)",
                max_diff
            ),
            call. = FALSE
        )
    }
    if (!identical(result$fraction_used, fraction)) {
        stop("fraction_used does not match requested fraction", call. = FALSE)
    }
    TRUE
}
