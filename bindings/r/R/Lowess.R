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
#' @param ... Not used; forces all subsequent arguments to be named.
#' @param fraction Smoothing fraction, greater than 0 and up to 1. Default:
#'   0.67. See Details for guidance on choosing a value.
#' @param iterations Number of robustness iterations, between 0 and 1000
#'   (inclusive). Default: 3.
#' @param delta Interpolation distance threshold, as a non-negative fraction
#'   of the x range; points within \code{delta} of each other on x share the
#'   same local fit. \code{NULL} (default) sets it automatically to 1/100th
#'   of the x range.
#' @param weight_function Kernel weight function. One of \code{"tricube"}
#'   (default), \code{"gaussian"}, \code{"uniform"} (alias: \code{"boxcar"}),
#'   \code{"cosine"}, \code{"epanechnikov"},
#'   \code{"biweight"} (alias: \code{"bisquare"}), or
#'   \code{"triangle"} (alias: \code{"triangular"}).
#' @param robustness_method Outlier downweighting method: \code{"bisquare"}
#'   (default; alias: \code{"biweight"}), \code{"huber"}, or \code{"talwar"}.
#' @param scaling_method Residual scale estimation for robustness weights:
#'   \code{"mad"} (default; alias: \code{"median_absolute_deviation"}),
#'   \code{"mar"} (alias: \code{"median_absolute_residual"}), or
#'   \code{"mean"} (alias: \code{"mean_absolute_residual"}).
#' @param boundary_policy Boundary handling strategy: \code{"extend"}
#'   (default; alias: \code{"pad"}), \code{"reflect"} (alias:
#'   \code{"mirror"}), \code{"zero"}, or
#'   \code{"noboundary"} (alias: \code{"none"}).
#' @param confidence_intervals Confidence level for confidence intervals,
#'   greater than 0 and less than 1 (e.g., 0.95). \code{NULL} (default)
#'   disables confidence intervals.
#' @param prediction_intervals Confidence level for prediction intervals,
#'   greater than 0 and less than 1 (e.g., 0.95). \code{NULL} (default)
#'   disables prediction intervals.
#' @param outputs Character vector selecting optional output components:
#'   \code{"diagnostics"}, \code{"residuals"}, \code{"weights"} (robustness
#'   weights), \code{"derivative"}, \code{"se"} (standard errors), and/or
#'   \code{"sorted"}. \code{NULL} (default) returns only the core result
#'   (\code{x}, \code{y}, and fit metadata).
#' @param zero_weight_fallback Fallback policy when all robustness weights drop
#'   to zero: \code{"use_local_mean"} (default; aliases: \code{"local_mean"},
#'   \code{"mean"}), \code{"return_original"} (alias: \code{"original"}), or
#'   \code{"return_none"} (alias: \code{"none"}).
#' @param auto_converge Convergence tolerance for early stopping of robustness
#'   iterations. \code{NULL} (default) disables early stopping.
#' @param cv Cross-validation options, created with \code{\link{cv_opts}}:
#'   e.g. \code{cv = cv_opts(fractions = c(0.2, 0.3, 0.5))}. \code{NULL}
#'   (default) disables cross-validation.
#' @param parallel Logical; enable parallel processing. Default: \code{TRUE}.
#' @param backend Execution backend: \code{"cpu"} (default) or \code{"gpu"}.
#'   GPU support requires the package to be built locally with
#'   \code{WITH_GPU=1} (see \code{bindings/r/Makefile}) and a
#'   Vulkan/Metal/DX12-capable GPU driver; not available in released
#'   CRAN/Bioconductor binaries.
#' @param missing Policy for non-finite (NaN/Infinity) values in input data:
#'   \code{"error"} (default) raises an error, \code{"drop"} silently removes
#'   affected observations before fitting.
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
    delta = NULL,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    confidence_intervals = NULL,
    prediction_intervals = NULL,
    zero_weight_fallback = "use_local_mean",
    auto_converge = NULL,
    parallel = TRUE,
    backend = "cpu",
    missing = "error",
    retain_model = FALSE,
    outputs = NULL,
    cv = NULL
) {
    reject_extra_positional_args(sys.call(), "fraction")
    check_gpu_backend(backend)
    validate_params(fraction = fraction, iterations = iterations)

    # Expand `return` into the flat boolean flags the Rust FFI expects.
    flags <- parse_outputs_flags(
        outputs,
        c("diagnostics", "residuals", "weights", "derivative", "se", "sorted")
    )
    return_diagnostics <- flags[["diagnostics"]]
    return_residuals <- flags[["residuals"]]
    return_robustness_weights <- flags[["weights"]]
    return_derivative <- flags[["derivative"]]
    return_se <- flags[["se"]]
    return_sorted <- flags[["sorted"]]

    # Expand `cv` into the flat FFI args.
    if (is.null(cv)) {
        cv_fractions <- NULL
        cv_method <- "kfold"
        cv_k <- 5L
        cv_seed <- NULL
    } else {
        cv_fractions <- cv$fractions
        cv_method <- if (is.null(cv$method)) "kfold" else cv$method
        cv_k <- if (is.null(cv$k)) 5L else cv$k
        cv_seed <- if (is.null(cv$seed)) NULL else cv$seed
    }

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

#' Cross-validation options for \code{\link{Lowess}}
#'
#' @description
#' Build a cross-validation options list to pass to
#' \code{Lowess(cv = ...)}. Cross-validation is disabled by default
#' (\code{cv = NULL}); supplying \code{cv_opts()} enables it and lets the
#' model pick the best smoothing fraction from the candidates.
#'
#' @param fractions Numeric vector of candidate smoothing fractions, each
#'   greater than 0 and up to 1 (e.g. \code{c(0.2, 0.3, 0.5)}).
#' @param method Cross-validation method: \code{"kfold"} (default) or
#'   \code{"loocv"}.
#' @param k Number of folds for k-fold cross-validation. Default: 5.
#' @param seed Integer seed for reproducible fold assignment. \code{NULL}
#'   (default) uses a random seed.
#'
#' @return A \code{cv_opts} list for \code{Lowess(cv = ...)}.
#' @examples
#' model <- Lowess(cv = cv_opts(fractions = c(0.2, 0.3, 0.5)))
#' @export
cv_opts <- function(fractions, method = "kfold", k = 5L, seed = NULL) {
    if (missing(fractions) || is.null(fractions)) {
        stop(
            "`fractions` must be a numeric vector of candidate fractions",
            call. = FALSE
        )
    }
    if (!is.numeric(fractions) || length(fractions) == 0L) {
        stop("`fractions` must be a non-empty numeric vector", call. = FALSE)
    }
    structure(
        list(
            fractions = as.double(fractions),
            method = as.character(method),
            k = as.integer(k),
            seed = seed
        ),
        class = "cv_opts"
    )
}
