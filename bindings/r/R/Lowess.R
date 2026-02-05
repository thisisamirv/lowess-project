#' LOWESS Batch Smoothing
#'
#' @description
#' Create a stateful LOWESS model for batch smoothing.
#'
#' @srrstats {G2.0} Input validation for fraction and iterations.
#' @srrstats {G2.1} Parameter bounds checking (fraction 0-1, iterations >= 0).
#' @srrstats {RE2.0} Kernel, robustness, boundary, and scaling configurable.
#'
#' @param fraction Smoothing fraction (0 to 1). Default: 0.67.
#' @param iterations Robustness iterations. Default: 3.
#' @param delta Interpolation threshold. NULL = auto.
#' @param weight_function Kernel name. Default: "tricube".
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
#' @return A Lowess object.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- Lowess(fraction = 0.2)
#' result <- model$fit(x, y)
#' plot(x, y)
#' lines(x, result$y, col = "red")
#' @export
Lowess <- function(
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
    validate_params(fraction = fraction, iterations = iterations)
    handle <- do.call(RLowess$new, env_args(lowess_params))

    # Return a wrapper that coerces inputs for methods
    structure(
        list(
            handle = handle,
            fit = function(x, y) {
                if (length(x) != length(y)) {
                    stop("x and y must have the same length")
                }
                handle$fit(as.double(x), as.double(y))
            },
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
