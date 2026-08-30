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
#' | Strategy | Behavior |
#' | --- | --- |
#' | `"average"` | Arithmetic mean of both estimates |
#' | `"weighted_average"` | Distance-weighted blend (recommended, default) |
#' | `"take_first"` | Keep left-chunk estimate |
#' | `"take_last"` | Keep right-chunk estimate |
#'
#' @srrstats {G2.0} Input validation for fraction, chunk_size.
#' @srrstats {G1.6} Memory-efficient streaming for large datasets.
#'
#' @inheritParams Lowess
#' @param chunk_size Number of data points per processing chunk, at least 10.
#'   Default: 5000.
#' @param overlap Number of overlapping points between consecutive chunks,
#'   less than \code{chunk_size}. \code{NULL} (default) uses the backend's
#'   default of 500.
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
#' final <- finalize(model)
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
    confidence_intervals = NULL,
    prediction_intervals = NULL
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
code{"zero"}, or
#'   \code{"noboundary"} (alias: \code{"none"}).
#' @param confidence_intervals Confidence level for confidence intervals,
#'   greater than 0 and less than 1 (e.g., 0.95). \code{NULL} (default)
#'   disables confidence intervals.
#' @param prediction_intervals Confidence level for prediction intervals,
#'   greater than 0 and less than 1 (e.g., 0.95). \code{NULL} (default)
#'   disables prediction intervals.
#' @param return_diagnostics Logical; if \code{TRUE}, return fit-quality
#'   metrics (RMSE, MAE, R-squared, AIC, etc.). Default: \code{FALSE}.
#' @param return_residuals Logical; if \code{TRUE}, return residuals in the
#'   result. Default: \code{FALSE}.
#' @param return_robustness_weights Logical; if \code{TRUE}, return per-point
#'   robustness weights. Default: \code{FALSE}.
#' @param zero_weight_fallback Fallback policy when all robustness weights drop
#'   to zero: \code{"use_local_mean"} (default; aliases: \code{"local_mean"},
#'   \code{"mean"}), \code{"return_original"} (alias: \code{"original"}), or
#'   \code{"return_none"} (alias: \code{"none"}).
#' @param auto_converge Convergence tolerance for early stopping of robustness
#'   iterations. \code{NULL} (default) disables early stopping.
#' @param cv_fractions Numeric vector of candidate fractions for
#'   cross-validation. \code{NULL} (default) disables CV.
#' @param cv_method Cross-validation method: \code{"kfold"} (default) or
#'   \code{"loocv"}.
#' @param cv_k Number of folds for k-fold CV. Default: 5.
#' @param parallel Logical; enable parallel processing. Default: \code{TRUE}.
#' @param cv_seed Integer seed for the cross-validation random number
#'   generator. \code{NULL} (default) uses a random seed.
#' @param return_se Logical; if \code{TRUE}, compute hat-matrix statistics
#'   (effective degrees of freedom, leverage, standard errors).
#'   Default: \code{FALSE}.
#' @param backend Execution backend: \code{"cpu"} (default) or \code{"gpu"}.
#'   GPU support requires the package to be built locally with
#'   \code{WITH_GPU=1} (see \code{bindings/r/Makefile}) and a
#'   Vulkan/Metal/DX12-capable GPU driver; not available in released
#'   CRAN/Bioconductor binaries.
#'
#' @return A Lowess object.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- Lowess(fraction = 0.2)
#' result <- fit(model, x, y)
#' plot(x, y)
#' lines(x, result$y, col = "red")
#' @export
Lowess <- function(
    fraction = 0.67,
    ...,
    iterations = 3L,
    delta = NULL,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    confidence_intervals = NULL,
    prediction_intervals = NULL,
    return_diagnostics = FALSE,
    return_residuals = FALSE,
    return_robustness_weights = FALSE,
    zero_weight_fallback = "use_local_mean",
    auto_converge = NULL,
    cv_fractions = NULL,
    cv_method = "kfold",
    cv_k = 5L,
    parallel = TRUE,
    cv_seed = NULL,
    return_se = FALSE,
    backend = "cpu"
) {
    reject_extra_positional_args(sys.call(), "fraction")
    check_gpu_backend(backend)
    validate_params(fraction = fraction, iterations = iterations)
    handle <- do.call(RLowess$new, env_args(lowess_params))

    structure(
        list(
            handle = handle,
            params = list(
                fraction = fraction,
                iterations = iterations,
                weight_function = weight_function,
                robustness_method = robustness_method,
                scaling_method = scaling_method,
                parallel = parallel
            )
        ),
        class = "Lowess"
    )
}
