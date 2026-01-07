#' LOWESS Smoothing with Batch Adapter
#'
#' @description
#' Perform LOWESS (Locally Weighted Scatterplot Smoothing) on the input data.
#' This is the primary interface for LOWESS smoothing, processing the entire
#' dataset in memory with optional parallel execution.
#'
#' ## When to use batch smoothing:
#' \itemize{
#'   \item Dataset fits comfortably in memory.
#'   \item Need all features (confidence intervals, cross-validation,
#'     diagnostics).
#'   \item Processing complete offline datasets for analysis or visualization.
#' }
#'
#' @param x Numeric vector of independent variable values.
#' @param y Numeric vector of dependent variable values (same length as x).
#' @param fraction Smoothing fraction, the proportion of data to use for each
#'   local regression (default: 0.67). Values between 0 and 1.
#'   \itemize{
#'     \item **0.1-0.3**: Fine detail, captures rapid changes (may be noisy).
#'     \item **0.4-0.6**: Balanced, good for most general-purpose applications.
#'     \item **0.7-1.0**: Heavy smoothing, emphasizes global trends.
#'   }
#' @param iterations Number of robustness iterations for outlier handling
#'   (default: 3). Use 0 for no robustness.
#'   \itemize{
#'     \item **0**: No robustness (fastest, but sensitive to outliers).
#'     \item **1-2**: Light contamination; standard recommended setting.
#'     \item **3**: Default; good balance of speed and resistance.
#'     \item **4-6**: Strong resistance; for heavily contaminated datasets.
#'   }
#' @param delta Interpolation optimization threshold. Points closer than delta
#'   will use linear interpolation instead of full regression. NULL (default)
#'   auto-calculates based on 1 percent of the data range. Set to 0 for no
#'   interpolation.
#' @param weight_function Kernel function for distance weighting. Options:
#'   "tricube" (default), "epanechnikov", "gaussian", "uniform", "biweight",
#'   "triangle", "cosine".
#' @param robustness_method Method for computing robustness weights. Options:
#'   "bisquare" (default), "huber", "talwar".
#' @param scaling_method Scaling method for robustness weight calculation.
#'   Options: "mad" (default), "mar" (Median Absolute Residual).
#' @param boundary_policy Handling of edge effects. Options: "extend" (default),
#'   "reflect", "zero", "noboundary". "extend" pads data at boundaries to
#'   maintain symmetric local neighborhoods, reducing boundary bias.
#' @param confidence_intervals Confidence level for confidence intervals
#'   (e.g., 0.95 for 95 percent CI). NULL (default) disables.
#' @param prediction_intervals Confidence level for prediction intervals
#'   (e.g., 0.95 for 95 percent PI). NULL (default) disables.
#' @param return_diagnostics Logical, whether to compute fit quality metrics
#'   (RMSE, MAE, R-squared, AIC, etc.). Default: FALSE.
#' @param return_residuals Logical, whether to include residuals in output.
#'   Default: FALSE.
#' @param return_robustness_weights Logical, whether to include final
#'   robustness weights in output. Default: FALSE.
#' @param zero_weight_fallback Fallback strategy when all weights are zero.
#'   Options: "use_local_mean" (default), "return_original", "return_none".
#' @param auto_converge Tolerance for automatic convergence detection in
#'   robustness iterations. NULL (default) disables auto-convergence.
#' @param cv_fractions Numeric vector of fractions to test for cross-validation.
#'   NULL (default) disables cross-validation. When provided, the internal
#'   engine selects the fraction that minimizes RMSE.
#' @param cv_method Cross-validation method. Options: "kfold" (default), "loocv"
#'   (leave-one-out).
#' @param cv_k Number of folds for k-fold cross-validation (default: 5).
#' @param parallel Logical, whether to enable parallel processing via Rayon
#'   (default: TRUE). Significant speedups for large datasets or many
#'   robustness iterations.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{x}: Sorted independent variable values.
#'   \item \code{y}: Smoothed dependent variable values.
#'   \item \code{fraction_used}: The fraction actually used (helpful if CV was
#'     used).
#'   \item \code{standard_errors}: Point-wise standard errors (if requested).
#'   \item \code{confidence_lower}/\code{upper}: CI bounds (if requested).
#'   \item \code{prediction_lower}/\code{upper}: PI bounds (if requested).
#'   \item \code{residuals}: Model residuals (if requested).
#'   \item \code{robustness_weights}: Final robustness weights (if requested).
#'   \item \code{iterations_used}: Number of iterations actually performed.
#'   \item \code{cv_scores}: RMSE scores for tested fractions (if CV was used).
#'   \item \code{diagnostics}: List of metrics (RMSE, MAE, R2, AIC, etc.)
#'     (if requested).
#' }
#'
#' @examples
#' # Basic smoothing
#' x <- seq(1, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, sd = 0.2)
#' result <- fastlowess(x, y, fraction = 0.3)
#' plot(x, y)
#' lines(result$x, result$y, col = "red", lwd = 2)
#'
#' # Robust smoothing with intervals
#' result <- fastlowess(x, y,
#'   fraction = 0.5, iterations = 5,
#'   confidence_intervals = 0.95,
#'   prediction_intervals = 0.95
#' )
#' lines(result$x, result$confidence_lower, col = "blue", lty = 2)
#' lines(result$x, result$confidence_upper, col = "blue", lty = 2)
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
