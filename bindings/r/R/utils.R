#' Common argument validation and coercion
#'
#' @description
#' Internal helper to validate x, y, fraction, and iterations inputs and
#' force them to the correct types for Rust FFI.
#'
#' @param x Numeric vector
#' @param y Numeric vector
#' @param fraction Numeric
#' @param iterations Integer
#'
#' @return A list containing the coerced x, y, fraction, and iterations.
#' @noRd
validate_common_args <- function(x, y, fraction, iterations) {
    if (length(x) != length(y)) {
        stop("x and y must have the same length")
    }
    if (length(x) < 3) {
        stop("At least 3 data points are required")
    }
    if (!is.numeric(fraction) || length(fraction) != 1) {
        stop("fraction must be a single numeric value")
    }
    if (fraction <= 0 || fraction > 1) {
        stop("fraction must be between 0 and 1")
    }
    if (!is.numeric(iterations) || length(iterations) != 1 || iterations < 0) {
        stop("iterations must be a non-negative integer")
    }

    list(
        x = as.double(x),
        y = as.double(y),
        fraction = as.double(fraction),
        iterations = as.integer(iterations)
    )
}
