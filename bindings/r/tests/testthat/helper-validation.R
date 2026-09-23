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
        scaling_method = "mar"
    )
    result <- fit(model, x, y)

    expect_equal(result$y, reference$y, tolerance = tolerance)
    expect_identical(result$fraction_used, fraction)

    invisible(result)
}
