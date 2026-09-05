#' LOWESS Online Smoothing
#'
#' @description
#' Create a stateful LOWESS model for real-time online data. Maintains a
#' sliding window and processes each incoming point immediately via
#' \code{\link{add_point}}.
#'
#' @details
#' Best suited when data arrives incrementally (e.g. sensors or streams),
#' real-time smoothed values are needed, or memory is fixed. For datasets
#' that fit in memory, see \code{\link{Lowess}}; for large batches processed
#' in chunks, see \code{\link{StreamingLowess}}.
#'
#' @srrstats {G2.0} Input validation for fraction, window_capacity, min_points.
#' @srrstats {G1.6} Sliding window for incremental updates.
#'
#' @inheritParams Lowess
#' @param window_capacity Maximum number of points kept in the sliding
#'   window, at least 3. Default: 1000.
#' @param min_points Minimum number of points required before smoothing
#'   begins, between 2 and \code{window_capacity}. Default: 2.
#' @param update_mode Window update strategy: \code{"incremental"} (default;
#'   alias: \code{"single"}) updates only the newest point;
#'   \code{"full"} (alias: \code{"resmooth"}) re-smooths all window points
#'   after each addition.
#'
#' @return An OnlineLowess object.
#' @examples
#' model <- OnlineLowess(fraction = 0.2, window_capacity = 20)
#' x <- 1:50
#' y <- sin(x * 0.1) + rnorm(50, 0, 0.1)
#' smoothed <- numeric(0)
#' for (i in seq_along(x)) {
#'     result <- add_point(model, x[i], y[i])
#'     if (!is.null(result)) smoothed <- c(smoothed, result$y)
#' }
#' head(smoothed, 5)
#' @export
OnlineLowess <- function(
    fraction = 0.67,
    window_capacity = 1000L,
    min_points = 2L,
    ...,
    iterations = 3L,
    delta = NULL,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    zero_weight_fallback = "use_local_mean",
    update_mode = "incremental",
    auto_converge = NULL,
    return_robustness_weights = FALSE,
    missing = "error"
) {
    reject_extra_positional_args(sys.call(), "min_points")
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
                iterations = iterations
            )
        ),
        class = "OnlineLowess"
    )
}
