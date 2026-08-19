#' LOWESS Batch Smoothing
#'
#' @description
#' Create a stateful LOWESS model for batch smoothing.
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
#' @param fraction Smoothing fraction (between 0 and 1). Default: 0.67.
#' @param iterations Number of robustness iterations (non-negative integer).
#'   Default: 3.
#' @param delta Interpolation distance threshold; points within \code{delta}
#'   of each other on x share the same local fit. \code{NULL} (default) sets
#'   it automatically to 1/100th of the x range.
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
#' @param confidence_intervals Confidence level for confidence intervals
#'   (e.g., 0.95). \code{NULL} (default) disables confidence intervals.
#' @param prediction_intervals Confidence level for prediction intervals
#'   (e.g., 0.95). \code{NULL} (default) disables prediction intervals.
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
#' @seealso \url{https://lowess.readthedocs.io/} for full documentation.
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
    if (identical(backend, "gpu") && !gpu_available()) {
        stop(
            "GPU backend not installed in this build. Run `install_gpu()` ",
            "once to download and install a GPU-enabled build, then ",
            "restart R. See https://lowess.readthedocs.io/api/r/",
            "#gpu-acceleration for details.",
            call. = FALSE
        )
    }
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
