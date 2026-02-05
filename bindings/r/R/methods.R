#' Print Lowess Model
#'
#' @srrstats {G1.3} S3 print methods for model objects.
#'
#' @param x A Lowess object.
#' @param ... Additional arguments (ignored).
#' @export
print.Lowess <- function(x, ...) {
    cat("<Lowess Model>\n")
    cat("  Fraction:         ", x$params$fraction, "\n")
    cat("  Iterations:       ", x$params$iterations, "\n")
    cat("  Weight Function:  ", x$params$weight_function, "\n")
    cat("  Parallel:         ", x$params$parallel, "\n")
    invisible(x)
}

#' Print Lowess Result
#'
#' @param x A LowessResult object.
#' @param ... Additional arguments (ignored).
#' @export
print.LowessResult <- function(x, ...) {
    cat("<LowessResult>\n")
    cat("  Points:           ", length(x$x), "\n")
    cat("  Fraction Used:    ", x$fraction_used, "\n")
    if (!is.null(x$iterations_used)) {
        cat("  Iterations Used:  ", x$iterations_used, "\n")
    }
    if (!is.null(x$cv_scores)) {
        cat("  CV Scores:        ", length(x$cv_scores), "folds\n")
    }
    invisible(x)
}

#' Plot Lowess Result
#'
#' @param x A LowessResult object.
#' @param main Plot title.
#' @param ... Additional arguments passed to plot() and lines().
#'
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- Lowess(fraction = 0.2)
#' res <- model$fit(x, y)
#' plot(res)
#' @export
plot.LowessResult <- function(x, main = "LOWESS Fit", ...) {
    # Plot the smoothed curve
    plot(
        x$x, x$y,
        type = "l", col = "blue", lwd = 2,
        xlab = "x", ylab = "Fitted",
        main = main, ...
    )

    # If confidence intervals exist, plot them
    if (!is.null(x$confidence_lower)) {
        lines(x$x, x$confidence_lower, lty = 2, col = "gray")
        lines(x$x, x$confidence_upper, lty = 2, col = "gray")
    }
}

#' Print StreamingLowess Model
#'
#' @param x A StreamingLowess object.
#' @param ... Additional arguments.
#' @export
print.StreamingLowess <- function(x, ...) {
    cat("<StreamingLowess Model>\n")
    cat("  Fraction:         ", x$params$fraction, "\n")
    cat("  Chunk Size:       ", x$params$chunk_size, "\n")
    cat("  Parallel:         ", x$params$parallel, "\n")
    invisible(x)
}

#' Print OnlineLowess Model
#'
#' @param x A OnlineLowess object.
#' @param ... Additional arguments.
#' @export
print.OnlineLowess <- function(x, ...) {
    cat("<OnlineLowess Model>\n")
    cat("  Fraction:         ", x$params$fraction, "\n")
    cat("  Window Capacity:  ", x$params$window_capacity, "\n")
    cat("  Min Points:       ", x$params$min_points, "\n")
    cat("  Update Mode:      ", x$params$update_mode, "\n")
    invisible(x)
}
