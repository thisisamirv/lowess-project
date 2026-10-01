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
#' @inheritParams Lowess fraction
#' @param ... Not used; forces all subsequent arguments to be named.
#' @inheritParams Lowess iterations:missing
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
#' @inheritParams Lowess parallel
#' @param outputs Character vector selecting optional output components:
#'   \code{"se"} (standard errors), \code{"diagnostics"},
#'   \code{"residuals"}, \code{"weights"} (robustness weights), and/or
#'   \code{"derivative"}. \code{NULL} (default) returns only the core result.
#' @inheritParams Lowess intervals seed
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
    ...,
    iterations = 3L,
    weight_function = "tricube",
    robustness_method = "bisquare",
    delta = NULL,
    zero_weight_fallback = "use_local_mean",
    boundary_policy = "extend",
    scaling_method = "mad",
    auto_converge = NULL,
    missing = "error",
    chunk_size = 5000L,
    overlap = NULL,
    merge_strategy = "weighted_average",
    parallel = TRUE,
    outputs = NULL,
    intervals = NULL,
    seed = NULL
) {
    reject_extra_positional_args(sys.call(), "fraction")
    validate_params(fraction = fraction, chunk_size = chunk_size)

    flags <- parse_outputs_flags(
        outputs,
        c("se", "diagnostics", "residuals", "weights", "derivative")
    )
    return_diagnostics <- flags[["diagnostics"]]
    return_residuals <- flags[["residuals"]]
    return_robustness_weights <- flags[["weights"]]
    return_derivative <- flags[["derivative"]]
    return_se <- flags[["se"]]

    iv <- expand_intervals(intervals)
    confidence_intervals <- iv$confidence
    prediction_intervals <- iv$prediction
    bootstrap <- iv$bootstrap

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
