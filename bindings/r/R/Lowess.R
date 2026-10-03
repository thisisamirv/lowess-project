#' LOWESS Batch Smoothing
#'
#' @description
#' Create a stateful LOWESS model for batch smoothing. This is the default
#' mode: it processes the entire dataset at once and supports every feature
#' (confidence/prediction intervals, cross-validation, GPU backend).
#'
#' @details
#' Best suited when the dataset fits in memory and you need intervals,
#' cross-validation, or diagnostics. For datasets that don't fit in memory or
#' arrive in chunks, see \code{\link{StreamingLowess}}; for point-by-point
#' real-time data, see \code{\link{OnlineLowess}}.
#'
#' `fraction` is the most important parameter: it controls the size of the
#' local neighbourhood used at each point.
#'
#' | Range | Effect | Use case |
#' | --- | --- | --- |
#' | 0.1-0.3 | Fine detail | Rapidly changing signals |
#' | 0.3-0.5 | Balanced | General purpose |
#' | 0.5-0.7 | Heavy smoothing | Noisy data |
#' | 0.7-1.0 | Very smooth | Trend extraction |
#'
#' @srrstats {G2.0} Input validation for fraction and iterations.
#' @srrstats {G2.1} Parameter bounds checking (fraction 0-1, iterations >= 0).
#' @srrstats {RE2.0} Kernel, robustness, boundary, and scaling configurable.
#' @srrstats {RE2.1, RE2.2} NA handling options available via Rust backend.
#' @srrstats {RE3.0, RE3.1} Convergence warnings; thresholds settable.
#' @srrstats {RE4.0, RE4.1} Model object returned; fitting via S3 generic fit().
#' @srrstats {RE4.7} Convergence stats returned in result.
#' @srrstats {RE4.8, RE4.9, RE4.10} Response, fitted, residuals returned.
#' @srrstats {RE4.11} Goodness-of-fit metrics via return_diagnostics.
#' @srrstats {RE5.0} O(n) scaling documented in README.
#'
#' @param fraction Smoothing fraction, greater than 0 and up to 1. Default:
#'   0.67.
#' @param ... Not used; forces all subsequent arguments to be named.
#' @param iterations Number of robustness iterations, between 0 and 1000
#'   (inclusive). Default: 3.
#' @param weight_function Kernel weight function. One of \code{"tricube"}
#'   (default), \code{"gaussian"}, \code{"uniform"} (alias: \code{"boxcar"}),
#'   \code{"cosine"}, \code{"epanechnikov"},
#'   \code{"biweight"} (alias: \code{"bisquare"}), or
#'   \code{"triangle"} (alias: \code{"triangular"}).
#' @param robustness_method Outlier downweighting method: \code{"bisquare"}
#'   (default; alias: \code{"biweight"}), \code{"huber"}, or \code{"talwar"}.
#' @param delta Interpolation distance threshold, as a non-negative fraction
#'   of the x range; points within \code{delta} of each other on x share the
#'   same local fit. \code{NULL} (default) sets it automatically to 1/100th
#'   of the x range.
#' @param zero_weight_fallback Fallback policy when all robustness weights drop
#'   to zero: \code{"use_local_mean"} (default; aliases: \code{"local_mean"},
#'   \code{"mean"}), \code{"return_original"} (alias: \code{"original"}), or
#'   \code{"return_none"} (alias: \code{"none"}).
#' @param boundary_policy Boundary handling strategy: \code{"extend"}
#'   (default; alias: \code{"pad"}), \code{"reflect"} (alias:
#'   \code{"mirror"}), \code{"zero"}, or
#'   \code{"noboundary"} (alias: \code{"none"}).
#' @param scaling_method Residual scale estimation for robustness weights:
#'   \code{"mad"} (default; alias: \code{"median_absolute_deviation"}),
#'   \code{"mar"} (alias: \code{"median_absolute_residual"}), or
#'   \code{"mean"} (alias: \code{"mean_absolute_residual"}).
#' @param auto_converge Convergence tolerance for early stopping of robustness
#'   iterations. \code{NULL} (default) disables early stopping.
#' @param missing Policy for non-finite (NaN/Infinity) values in input data:
#'   \code{"error"} (default) raises an error, \code{"drop"} silently removes
#'   affected observations before fitting.
#' @param parallel Logical; enable parallel processing. Default: \code{TRUE}.
#' @param backend Execution backend: \code{"cpu"} (default) or \code{"gpu"}.
#'   GPU support requires the package to be built locally with
#'   \code{WITH_GPU=1} (see \code{bindings/r/Makefile}) and a
#'   Vulkan/Metal/DX12-capable GPU driver; not available in released
#'   CRAN/Bioconductor binaries.
#' @param outputs Character vector selecting optional output components:
#'   \code{"se"} (standard errors), \code{"diagnostics"},
#'   \code{"residuals"}, \code{"weights"} (robustness weights),
#'   \code{"derivative"}, and/or \code{"sorted"}. \code{NULL} (default)
#'   returns only the core result (\code{x}, \code{y}, and fit metadata).
#' @param intervals Interval options, created with
#'   \code{\link{intervals_opts}} (or a named list with any of
#'   \code{confidence}, \code{prediction}, \code{bootstrap}): e.g.
#'   \code{intervals = intervals_opts(confidence = 0.90, prediction = 0.99,
#'   bootstrap = 200)}. Confidence and prediction coverage levels are
#'   independent and may differ.
#'   \code{NULL} (default) disables intervals.
#' @param cv Cross-validation options, created with \code{\link{cv_opts}}:
#'   e.g. \code{cv = cv_opts(method = "kfold", k = 5,
#'   fractions = c(0.2, 0.3, 0.5))}.
#'   \code{NULL} (default) disables cross-validation.
#' @param seed Non-negative whole-number seed shared by cross-validation and
#'   bootstrap resampling for reproducible results. When cross-validation is
#'   unavailable, it controls bootstrap resampling only.
#'   \code{NULL} (default) uses a random seed.
#' @param retain_model Logical; if \code{TRUE}, retain the fitted model's
#'   training data, enabling \code{\link{predict.Lowess}} for out-of-sample
#'   prediction. Default: \code{FALSE}.
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
    weight_function = "tricube",
    robustness_method = "bisquare",
    delta = NULL,
    zero_weight_fallback = "use_local_mean",
    boundary_policy = "extend",
    scaling_method = "mad",
    auto_converge = NULL,
    missing = "error",
    parallel = TRUE,
    backend = "cpu",
    outputs = NULL,
    intervals = NULL,
    cv = NULL,
    seed = NULL,
    retain_model = FALSE
) {
    reject_extra_positional_args(sys.call(), "fraction")
    if (...length() > 0L) {
        stop("unused arguments (...)", call. = FALSE)
    }
    check_gpu_backend(backend)
    validate_params(fraction = fraction, iterations = iterations)

    grouped <- lowess_grouped_args(outputs, intervals, cv)
    handle <- do.call(RLowess$new, env_args(lowess_params, grouped))

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

#' Cross-validation options for \code{\link{Lowess}}
#'
#' @description
#' Build a cross-validation options list to pass to
#' \code{Lowess(cv = ...)}. Cross-validation is disabled by default
#' (\code{cv = NULL}); supplying \code{cv_opts()} enables it and lets the
#' model pick the best smoothing fraction from the candidates.
#'
#' @param method Cross-validation method: \code{"kfold"} (default) or
#'   \code{"loocv"}.
#' @param k Number of folds for k-fold cross-validation. Default: 5.
#' @param fractions Numeric vector of candidate smoothing fractions, each
#'   greater than 0 and up to 1 (e.g. \code{c(0.2, 0.3, 0.5)}).
#'
#' @return A \code{cv_opts} list for \code{Lowess(cv = ...)}.
#'   Use \code{Lowess(seed = ...)} for reproducible fold assignment.
#' @examples
#' model <- Lowess(
#'     cv = cv_opts(method = "kfold", k = 5, fractions = c(0.2, 0.3, 0.5)),
#'     seed = 42
#' )
#' @export
cv_opts <- function(method = "kfold", k = 5L, fractions) {
    if (missing(fractions) || is.null(fractions)) {
        stop(
            "`fractions` must be a numeric vector of candidate fractions",
            call. = FALSE
        )
    }
    if (!is.numeric(fractions) || length(fractions) == 0L) {
        stop("`fractions` must be a non-empty numeric vector", call. = FALSE)
    }
    validate_numeric_vector(fractions, "fractions")
    validate_optional_count(k, "cv.k", allow_zero = FALSE)
    structure(
        list(
            method = as.character(method),
            k = as.integer(k),
            fractions = as.double(fractions)
        ),
        class = "cv_opts"
    )
}

#' Interval options for LOWESS models
#'
#' @description
#' Build an interval options list to pass to \code{intervals = ...} in
#' \code{\link{Lowess}}, \code{\link{StreamingLowess}},
#' \code{\link{OnlineLowess}}, or \code{\link{predict.Lowess}}.
#'
#' @param confidence Coverage for confidence intervals, greater than 0 and
#'   less than 1 (e.g. 0.90). \code{NULL} (default) disables them.
#' @param prediction Coverage for prediction intervals, greater than 0 and
#'   less than 1 (e.g. 0.99). \code{NULL} (default) disables them. This level
#'   is independent of \code{confidence}.
#' @param bootstrap Number of bootstrap resamples used to compute the
#'   intervals. \code{0} (default) uses the analytic intervals.
#'
#' @return An \code{intervals_opts} list.
#' @examples
#' model <- Lowess(
#'     intervals = intervals_opts(
#'         confidence = 0.90, prediction = 0.99, bootstrap = 200
#'     ),
#'     seed = 42
#' )
#' @export
intervals_opts <- function(
    confidence = NULL,
    prediction = NULL,
    bootstrap = 0L
) {
    validate_optional_count(bootstrap, "bootstrap")
    structure(
        list(
            confidence = confidence,
            prediction = prediction,
            bootstrap = as.integer(bootstrap)
        ),
        class = "intervals_opts"
    )
}
