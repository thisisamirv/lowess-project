#' LOWESS Online Smoothing
#'
#' @description
#' Create a stateful LOWESS model for real-time online data.
#'
#' @srrstats {G2.0} Input validation for fraction, window_capacity, min_points.
#' @srrstats {G1.6} Sliding window for incremental updates.
#'
#' @inheritParams Lowess
#' @param window_capacity Maximum number of points kept in the sliding window.
#' @param min_points Minimum number of points required before smoothing begins.
#' @param update_mode Window update strategy: \code{"full"} (default; alias:
#'   \code{"resmooth"}) re-smooths all window points after each addition;
#'   \code{"incremental"} (alias: \code{"single"}) updates only the newest
#'   point.
#'
#' @seealso \url{https://lowess.readthedocs.io/} for full documentation.
#' @return An OnlineLowess object.
#' @examples
#' model <- OnlineLowess(fraction = 0.2, window_capacity = 20)
#' x <- 1:50
#' y <- sin(x * 0.1) + rnorm(50, 0, 0.1)
#' for (i in seq_along(x)) {
#'     result <- add_point(model, x[i], y[i])
#'     if (!is.null(result)) cat("smoothed:", result$smoothed, "\n")
#' }
#' @export
OnlineLowess <- function(
    fraction = 0.67,
    window_capacity = 1000L,
    min_points = 3L,
    ...,
    iterations = 3L,
    delta = NULL,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    zero_weight_fallback = "use_local_mean",
    update_mode = "full",
    auto_converge = NULL,
    return_robustness_weights = FALSE,
    return_diagnostics = FALSE,
    return_residuals = FALSE,
    parallel = FALSE,
    confidence_intervals = NULL,
    prediction_intervals = NULL
) {
    if (...length() > 0) {
        stop("All arguments after 'min_points' must be named.", call. = FALSE)
    }
    validate_params(
        fraction = fraction,
        window_capacity = window_capacity,
        min_points = min_points
    )
    handle <- do.call(ROnlineLowess$new, env_args(online_params))

    structure(
        list(
            handle = handle,
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
