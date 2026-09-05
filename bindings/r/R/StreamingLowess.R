#' LOWESS Streaming Smoothing
#'
#' @description
#' Create a stateful LOWESS model for streaming data. Processes data in
#' fixed-size chunks with configurable overlap: results for each chunk are
#' returned by \code{\link{process_chunk}}, and \code{\link{finalize}}
#' flushes any remaining buffered points after the last chunk.
#'
#' @details
#' Best suited for datasets over 100,000 points, memory-constrained
#' environments, or batch processing pipelines. For smaller datasets that fit
#' in memory, see \code{\link{Lowess}}; for point-by-point real-time data,
#' see \code{\link{OnlineLowess}}.
#'
#' Overlapping regions between chunks are reconciled via `merge_strategy`:
#'
#' | Strategy | Alias | Behavior |
#' | --- | --- | --- |
#' | `"weighted_average"` (default) | `"weighted"` | Distance-weighted blend |
#' | `"average"` | `"mean"` | Average overlapping values |
#' | `"take_first"` | `"first"` | Keep left chunk values |
#' | `"take_last"` | `"last"` | Keep right chunk values |
#'
#' @srrstats {G2.0} Input validation for fraction, chunk_size.
#' @srrstats {G1.6} Memory-efficient streaming for large datasets.
#'
#' @inheritParams Lowess
#' @param chunk_size Number of data points per processing chunk, at least 10.
#'   Default: 5000.
#' @param overlap Number of overlapping points between consecutive chunks,
#'   less than \code{chunk_size}. \code{NULL} (default) computes
#'   \code{chunk_size / 10}, clamped to at least 1 and less than
#'   \code{chunk_size}.
#' @param merge_strategy Strategy for reconciling overlapping chunk regions:
#'   \code{"weighted_average"} (default; alias: \code{"weighted"}),
#'   \code{"average"} (alias: \code{"mean"}),
#'   \code{"take_first"} (alias: \code{"first"}), or
#'   \code{"take_last"} (alias: \code{"last"}).
#'
#' @return A StreamingLowess object.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- StreamingLowess(fraction = 0.2, chunk_size = 50)
#' res1 <- process_chunk(model, x[1:50], y[1:50])
#' res2 <- process_chunk(model, x[51:100], y[51:100])
#' finalize(model)
#' @export
StreamingLowess <- function(
    fraction = 0.67,
    chunk_size = 5000L,
    ...,
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
    missing = "error"
) {
    reject_extra_positional_args(sys.call(), "chunk_size")
    validate_params(fraction = fraction, chunk_size = chunk_size)
    handle <- do.call(RStreamingLowess$new, env_args(streaming_params))

    structure(
        list(
            handle = handle,
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
