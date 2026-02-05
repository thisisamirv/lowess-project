#' LOWESS Streaming Smoothing
#'
#' @description
#' Create a stateful LOWESS model for streaming data.
#'
#' @srrstats {G2.0} Input validation for fraction, chunk_size.
#' @srrstats {G1.6} Memory-efficient streaming for large datasets.
#'
#' @param fraction Smoothing fraction.
#' @param chunk_size Points per chunk.
#' @param overlap Overlap between chunks.
#' @param iterations Robustness iterations.
#' @param delta Interpolation threshold. NULL = auto.
#' @param weight_function Kernel name. Default: "tricube".
#' @param robustness_method Method: "bisquare", "huber", "talwar".
#' @param scaling_method Scale estimation: "mad", "mar".
#' @param boundary_policy Edge handling: "extend", "reflect", "zero",
#'   "noboundary".
#' @param auto_converge Convergence tolerance. NULL disables.
#' @param return_diagnostics Return fit metrics. Default: FALSE.
#' @param return_robustness_weights Return weights. Default: FALSE.
#' @param parallel Enable parallel processing. Default: TRUE.
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
    validate_params(fraction = fraction, chunk_size = chunk_size)
    handle <- do.call(RStreamingLowess$new, env_args(streaming_params))

    structure(
        list(
            handle = handle,
            process_chunk = function(x, y) {
                if (length(x) != length(y)) {
                    stop("x and y must have the same length")
                }
                handle$process_chunk(as.double(x), as.double(y))
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
