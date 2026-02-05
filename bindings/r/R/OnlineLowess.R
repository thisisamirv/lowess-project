#' LOWESS Online Smoothing
#'
#' @description
#' Create a stateful LOWESS model for real-time online data.
#'
#' @srrstats {G2.0} Input validation for fraction, window_capacity, min_points.
#' @srrstats {G1.6} Sliding window for incremental updates.
#'
#' @param fraction Smoothing fraction.
#' @param window_capacity Max points in sliding window.
#' @param min_points Minimum points before smoothing.
#' @param iterations Robustness iterations.
#' @param delta Interpolation threshold. NULL = auto.
#' @param weight_function Kernel name. Default: "tricube".
#' @param robustness_method Method: "bisquare", "huber", "talwar".
#' @param scaling_method Scale estimation: "mad", "mar".
#' @param boundary_policy Edge handling: "extend", "reflect", "zero",
#'   "noboundary".
#' @param update_mode Update strategy: "incremental".
#' @param auto_converge Convergence tolerance. NULL disables.
#' @param return_robustness_weights Return weights. Default: FALSE.
#' @param parallel Enable parallel processing. Default: TRUE.
#'
#' @return An OnlineLowess object.
#' @examples
#' model <- OnlineLowess(fraction = 0.2, window_capacity = 20)
#' x <- 1:50
#' y <- sin(x * 0.1) + rnorm(50, 0, 0.1)
#' result <- model$add_points(x, y)
#' plot(x, y)
#' lines(x, result$y, col = "red")
#' @export
OnlineLowess <- function(
    fraction = 0.2,
    window_capacity = 100L,
    min_points = 2L,
    iterations = 3L,
    delta = NULL,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    update_mode = "incremental",
    auto_converge = NULL,
    return_robustness_weights = FALSE,
    parallel = FALSE
) {
    # Validation
    if (fraction < 0 || fraction > 1) {
        stop("fraction must be between 0 and 1")
    }
    if (window_capacity <= 0) {
        stop("window_capacity must be a positive integer")
    }
    if (min_points < 0) {
        stop("min_points must be a non-negative integer")
    }

    if (is.null(delta)) delta <- Nullable(NULL)
    if (is.null(auto_converge)) auto_converge <- Nullable(NULL)

    handle <- ROnlineLowess$new(
        as.double(fraction),
        as.integer(window_capacity),
        as.integer(min_points),
        as.integer(iterations),
        delta,
        as.character(weight_function),
        as.character(robustness_method),
        as.character(scaling_method),
        as.character(boundary_policy),
        as.character(update_mode),
        auto_converge,
        as.logical(return_robustness_weights),
        as.logical(parallel)
    )

    structure(
        list(
            handle = handle,
            add_points = function(x, y) {
                if (length(x) != length(y)) {
                    stop("x and y must have the same length")
                }
                handle$add_points(as.double(x), as.double(y))
            },
            params = list(
                fraction = fraction,
                window_capacity = window_capacity,
                min_points = min_points,
                iterations = iterations,
                parallel = parallel
            )
        ),
        class = "OnlineLowess"
    )
}
