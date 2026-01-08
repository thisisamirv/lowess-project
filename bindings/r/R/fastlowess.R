#' LOWESS Batch Smoothing
#'
#' @description
#' Primary interface for LOWESS smoothing. Processes the entire dataset in
#' memory with optional parallel execution.
#'
#' For full documentation, see: \url{https://lowess.readthedocs.io/}
#'
#' @param x Numeric vector of independent variable values.
#' @param y Numeric vector of dependent variable values (same length as x).
#' @param fraction Smoothing fraction (0 to 1). Default: 0.67.
#' @param iterations Robustness iterations. Default: 3.
#' @param delta Interpolation threshold. NULL = auto.
#' @param weight_function Kernel: "tricube", "epanechnikov", "gaussian",
#'   "uniform", "biweight", "triangle", "cosine".
#' @param robustness_method Method: "bisquare", "huber", "talwar".
#' @param scaling_method Scale estimation: "mad", "mar".
#' @param boundary_policy Edge handling: "extend", "reflect", "zero",
#'   "noboundary".
#' @param confidence_intervals Confidence level (e.g., 0.95). NULL disables.
#' @param prediction_intervals Prediction level (e.g., 0.95). NULL disables.
#' @param return_diagnostics Return fit metrics. Default: FALSE.
#' @param return_residuals Return residuals. Default: FALSE.
#' @param return_robustness_weights Return weights. Default: FALSE.
#' @param zero_weight_fallback Fallback: "use_local_mean", "return_original",
#'   "return_none".
#' @param auto_converge Convergence tolerance. NULL disables.
#' @param cv_fractions Fractions for cross-validation. NULL disables.
#' @param cv_method CV method: "kfold", "loocv".
#' @param cv_k Folds for k-fold CV. Default: 5.
#' @param parallel Enable parallel processing. Default: TRUE.
#'
#' @return A list with x, y (smoothed), fraction_used, and optional fields.
#'
#' @examples
#' x <- seq(1, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, sd = 0.2)
#' result <- fastlowess(x, y, fraction = 0.3)
#' plot(x, y)
#' lines(result$x, result$y, col = "red", lwd = 2)
#'
#' @seealso \code{\link{fastlowess_online}}, \code{\link{fastlowess_streaming}}
#' @family fastlowess
#' @export
fastlowess <- function(
    x,
    y,
    fraction = 0.67,
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
    parallel = TRUE
) {
    args <- validate_common_args(x, y, fraction, iterations)
    x <- args$x
    y <- args$y
    fraction <- args$fraction
    iterations <- args$iterations

    cv_k <- as.integer(cv_k)

    # Call the Rust function
    .Call("wrap__fastlowess", x, y, fraction, iterations, delta,
        weight_function, robustness_method, scaling_method, boundary_policy,
        confidence_intervals, prediction_intervals, return_diagnostics,
        return_residuals, return_robustness_weights, zero_weight_fallback,
        auto_converge, cv_fractions, cv_method, cv_k, parallel,
        PACKAGE = "rfastlowess"
    )
}
