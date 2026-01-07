#' Online LOWESS with Sliding Window
#'
#' @description
#' Perform LOWESS smoothing using an online/sliding window approach for
#' real-time data streams. Maintains a sliding window of recent points for
#' incremental updates without reprocessing the entire dataset.
#'
#' ## When to use online smoothing:
#' \itemize{
#'   \item Data arrives incrementally (e.g., sensor readings, live feeds).
#'   \item Need real-time updates for each new observation.
#'   \item Maintaining a sliding window of the most recent history.
#'   \item Performance/latency is critical for individually arriving points.
#' }
#'
#' @param x Numeric vector of independent variable values.
#' @param y Numeric vector of dependent variable values (same length as x).
#' @param fraction Smoothing fraction (default: 0.2). Lower values (0.1-0.3)
#'   are recommended for online processing with small windows to ensure
#'   responsiveness to local changes.
#' @param window_capacity Maximum number of points to retain in the sliding
#'   window (default: 100). When capacity is reached, the oldest points are
#'   removed as new ones arrive.
#' @param min_points Minimum number of points required before smoothing starts
#'   (default: 2). Points before this threshold return their original y values.
#' @param iterations Number of robustness iterations (default: 3).
#'   \itemize{
#'     \item **0**: Fastest; no outlier weighting.
#'     \item **1-2**: Recommended balance for real-time applications.
#'   }
#' @param delta Interpolation optimization threshold. NULL (default)
#'   auto-calculates. Note: Online mode usually uses small deltas (or 0) for
#'   maximum responsive precision.
#' @param weight_function Kernel function for distance weighting. Options:
#'   "tricube" (default), "epanechnikov", "gaussian", "uniform", "biweight",
#'   "triangle", "cosine".
#' @param robustness_method Method for computing robustness weights. Options:
#'   "bisquare" (default), "huber", "talwar".
#' @param scaling_method Scaling method for robustness weight calculation.
#'   Options: "mad" (default), "mar" (Median Absolute Residual).
#' @param boundary_policy Handling of edge effects. Options: "extend" (default),
#'   "reflect", "zero", "noboundary".
#' @param update_mode Update strategy. Options: "full" (default) or
#'   "incremental".
#'   \itemize{
#'     \item **"full"**: Re-smooths all points in the window for each update.
#'     \item **"incremental"**: Only computes the estimate for the latest point.
#'   }
#' @param auto_converge Tolerance for automatic convergence. NULL (default)
#'   disables.
#' @param return_robustness_weights Logical, whether to include robustness
#'   weights in output. Default: FALSE.
#' @param parallel Logical, whether to enable parallel processing
#'   (default: FALSE). Online mode typically processes points sequentially
#'   to minimize per-point latency.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{x}: Input independent variable values.
#'   \item \code{y}: Smoothed dependent variable values.
#'   \item \code{fraction_used}: The fraction used for smoothing.
#'   \item \code{robustness_weights}: Weights for the window points (if
#'     requested).
#' }
#'
#' @examples
#' # Real-time sensor data smoothing simulation
#' x <- 1:100
#' y <- sin(x / 10) + rnorm(100, sd = 0.3)
#'
#' # online approach processes points as a sequence
#' result <- fastlowess_online(x, y, window_capacity = 25L, min_points = 10L)
#'
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
