#' LOWESS Online Smoothing
#'
#' @description
#' Create a stateful LOWESS model for real-time online data.
#'
#' @srrstats {G2.0} Input validation for fraction, window_capacity, min_points.
#' @srrstats {G1.6} Sliding window for incremental updates.
#'
#' @inheritParams Lowess
#' @param window_capacity Max points in sliding window.
#' @param min_points Minimum points before smoothing.
#' @param update_mode Update strategy: "full" or "incremental".
#'
#' @return An OnlineLowess object.
#' @examples
#' model <- OnlineLowess(fraction = 0.2, window_capacity = 20)
#' x <- 1:50
#' y <- sin(x * 0.1) + rnorm(50, 0, 0.1)
#' result <- model$add_point(1.0, sin(1.0 * 0.1))
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
    update_mode = "full",
    auto_converge = NULL,
    return_robustness_weights = FALSE,
    parallel = FALSE
) {
    validate_params(
        fraction = fraction, window_capacity = window_capacity,
        min_points = min_points
    )
    handle <- do.call(ROnlineLowess$new, env_args(online_params))

    structure(
        list(
            handle = handle,
            add_point = function(x, y) {
                handle$add_point(as.double(x), as.double(y))
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
