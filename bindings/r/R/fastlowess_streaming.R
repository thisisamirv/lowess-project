#' Streaming LOWESS for Large Datasets
#'
#' @description
#' Chunked LOWESS for large datasets. Processes data in chunks to maintain
#' constant memory usage.
#'
#' For full documentation, see: \url{https://lowess.readthedocs.io/}
#'
#' @param x Numeric vector of independent variable values.
#' @param y Numeric vector of dependent variable values (same length as x).
#' @param fraction Smoothing fraction. Default: 0.3.
#' @param chunk_size Points per chunk. Default: 5000.
#' @param overlap Points overlapping between chunks. NULL = 10 percent of chunk.
#' @param iterations Robustness iterations. Default: 3.
#' @param delta Interpolation threshold. NULL = auto.
#' @param weight_function Kernel function. Default: "tricube".
#' @param robustness_method Robustness method. Default: "bisquare".
#' @param scaling_method Scale estimation. Default: "mad".
#' @param boundary_policy Edge handling. Default: "extend".
#' @param auto_converge Convergence tolerance. NULL disables.
#' @param return_diagnostics Return cumulative metrics. Default: FALSE.
#' @param return_robustness_weights Return weights. Default: FALSE.
#' @param parallel Enable parallel processing. Default: TRUE.
#'
#' @return A list with x, y (smoothed), fraction_used, and optional fields.
#'
#' @examples
#' n <- 20000
#' x <- seq(0, 100, length.out = n)
#' y <- sin(x / 10) + rnorm(n, sd = 0.5)
#' result <- fastlowess_streaming(x, y, chunk_size = 5000L)
#' plot(x, y, pch = ".")
#' lines(result$x, result$y, col = "red", lwd = 2)
#'
#' @seealso \code{\link{fastlowess}}, \code{\link{fastlowess_online}}
#' @family fastlowess
#' @export
fastlowess_streaming <- function(
    x,
    y,
    fraction = 0.3,
    chunk_size = 5000L,
    overlap = NULL,
    iterations = 3L,
    delta = NULL,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    auto_converge = NULL,
    return_diagnostics = FALSE,
    return_robustness_weights = FALSE,
    parallel = TRUE
) {
    args <- validate_common_args(x, y, fraction, iterations)
    x <- args$x
    y <- args$y
    fraction <- args$fraction
    iterations <- args$iterations

    if (!is.numeric(chunk_size) || length(chunk_size) != 1 || chunk_size <= 0) {
        stop("chunk_size must be a positive integer")
    }

    chunk_size <- as.integer(chunk_size)

    if (!is.null(overlap)) {
        overlap <- as.integer(overlap)
    }

    # Call the Rust function
    .Call("wrap__fastlowess_streaming", x, y, fraction, chunk_size, overlap,
        iterations, delta, weight_function, robustness_method, scaling_method,
        boundary_policy, auto_converge, return_diagnostics,
        return_robustness_weights, parallel,
        PACKAGE = "rfastlowess"
    )
}
