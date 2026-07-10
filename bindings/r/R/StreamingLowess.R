#' LOWESS Streaming Smoothing
#'
#' @description
#' Create a stateful LOWESS model for streaming data.
#'
#' @srrstats {G2.0} Input validation for fraction, chunk_size.
#' @srrstats {G1.6} Memory-efficient streaming for large datasets.
#'
#' @inheritParams Lowess
#' @param chunk_size Points per chunk.
#' @param overlap Overlap between chunks.
#' @param merge_strategy Strategy for reconciling overlapping chunk regions:
#'   \code{"weighted_average"} (default), \code{"average"},
#'   \code{"take_first"}, or \code{"take_last"}.
#'
#' @return A StreamingLowess object.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- StreamingLowess(fraction = 0.2, chunk_size = 50)
#' res1 <- model$process_chunk(x[1:50], y[1:50])
#' res2 <- model$process_chunk(x[51:100], y[51:100])
#' final <- model$finalize()
#' @export
StreamingLowess <- function(
    fraction = 0.67,
    chunk_size = 5000L,
    overlap = NULL,
    iterations = 3L,
    delta = NULL,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    zero_weight_fallback = "use_local_mean",
    auto_converge = NULL,
    return_diagnostics = FALSE,
    return_residuals = FALSE,
    return_robustness_weights = FALSE,
    merge_strategy = "weighted_average",
    parallel = TRUE,
    confidence_intervals = NULL,
    prediction_intervals = NULL
) {
    validate_params(fraction = fraction, chunk_size = chunk_size)
    handle <- do.call(RStreamingLowess$new, env_args(streaming_params))

    structure(
        list(
            handle = handle,
            process_chunk = function(x, y) {
                args <- validate_common_args(x, y, fraction, iterations)
                handle$process_chunk(args$x, args$y)
            },
            finalize = function() {
                handle$finalize()
            },
            params = list(
                fraction = fraction,
                chunk_size = chunk_size,
                iterations = iterations,
                parallel = parallel
            )
        ),
        class = "StreamingLowess"
    )
}
