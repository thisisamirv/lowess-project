#' Online LOWESS with Sliding Window
#'
#' @description
#' Sliding window LOWESS for real-time data streams. Maintains a window of
#' recent points for incremental updates.
#'
#' For full documentation, see: \url{https://lowess.readthedocs.io/}
#'
#' @param x Numeric vector of independent variable values.
#' @param y Numeric vector of dependent variable values (same length as x).
#' @param fraction Smoothing fraction. Default: 0.2.
#' @param window_capacity Max points in sliding window. Default: 100.
#' @param min_points Min points before smoothing starts. Default: 3.
#' @param iterations Robustness iterations. Default: 3.
#' @param delta Interpolation threshold. NULL = auto.
#' @param weight_function Kernel function. Default: "tricube".
#' @param robustness_method Robustness method. Default: "bisquare".
#' @param scaling_method Scale estimation. Default: "mad".
#' @param boundary_policy Edge handling. Default: "extend".
#' @param update_mode "full" or "incremental". Default: "full".
#' @param auto_converge Convergence tolerance. NULL disables.
#' @param return_robustness_weights Return weights. Default: FALSE.
#' @param parallel Enable parallel processing. Default: FALSE.
#'
#' @return A list with x, y (smoothed), fraction_used, and optional fields.
#'
#' @examples
#' x <- 1:100
#' y <- sin(x / 10) + rnorm(100, sd = 0.3)
#' result <- fastlowess_online(x, y, window_capacity = 25L)
#' plot(x, y)
#' lines(result$x, result$y, col = "red", lwd = 2)
#'
#' @seealso \code{\link{fastlowess}}, \code{\link{fastlowess_streaming}}
#' @family fastlowess
#' @export
fastlowess_online <- function(
    x,
    y,
    fraction = 0.2,
    window_capacity = 100L,
    min_points = 3L,
    iterations = 3L,
    delta = NULL,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    update_mode = "full",
    auto_converge = NULL,
    return_robustness_weights = FALSE,
    parallel = FALSE
) {
    args <- validate_common_args(x, y, fraction, iterations)
    x <- args$x
    y <- args$y
    fraction <- args$fraction
    iterations <- args$iterations

    # Validate window_capacity
    is_invalid_wc <- !is.numeric(window_capacity)
    is_wrong_length_wc <- length(window_capacity) != 1
    is_non_positive_wc <- window_capacity <= 0
    if (is_invalid_wc || is_wrong_length_wc || is_non_positive_wc) {
        stop("window_capacity must be a positive integer")
    }

    # Validate min_points
    is_invalid_mp <- !is.numeric(min_points)
    is_wrong_length_mp <- length(min_points) != 1
    is_negative_mp <- min_points < 0
    if (is_invalid_mp || is_wrong_length_mp || is_negative_mp) {
        stop("min_points must be a non-negative integer")
    }

    window_capacity <- as.integer(window_capacity)
    min_points <- as.integer(min_points)

    # Call the Rust function
    .Call("wrap__fastlowess_online", x, y, fraction, window_capacity,
        min_points, iterations, delta, weight_function, robustness_method,
        scaling_method, boundary_policy, update_mode, auto_converge,
        return_robustness_weights, parallel,
        PACKAGE = "rfastlowess"
    )
}
