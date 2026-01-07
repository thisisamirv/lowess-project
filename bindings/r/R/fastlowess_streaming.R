#' Streaming LOWESS for Large Datasets
#'
#' @description
#' Perform LOWESS smoothing using a streaming/chunked approach for large
#' datasets. Processes data in chunks to maintain constant memory usage,
#' suitable for datasets too large to fit in memory.
#'
#' ## When to use streaming smoothing:
#' \itemize{
#'   \item Dataset size exceeds available system memory (>100K points).
#'   \item Processing large data files in a pipeline.
#'   \item Memory usage must be strictly bounded.
#' }
#'
#' @param x Numeric vector of independent variable values.
#' @param y Numeric vector of dependent variable values (same length as x).
#' @param fraction Smoothing fraction (default: 0.3). Lower values (0.1-0.3)
#'   are typically recommended for streaming to maintain good local precision
#'   within small chunks.
#' @param chunk_size Number of points to process in each chunk (default: 5000).
#'   Larger chunks improve smoothness but increase memory usage.
#' @param overlap Number of points to overlap between chunks (default: 10
#'   percent of chunk_size). Overlap ensures smooth transitions and consistent
#'   fits across chunk boundaries. Overlapping values are merged (typically
#'   averaged).
#' @param iterations Number of robustness iterations (default: 3).
#'   \itemize{
#'     \item **0**: Fastest; standard least-squares within chunks.
#'     \item **1-2**: Recommended for light to moderate outliers.
#'     \item **3**: Default; high resistance to noise.
#'   }
#' @param delta Interpolation optimization threshold. NULL (default)
#'   auto-calculates. Set to 0 to disable interpolation.
#' @param weight_function Kernel function for distance weighting. Options:
#'   "tricube" (default), "epanechnikov", "gaussian", "uniform", "biweight",
#'   "triangle", "cosine".
#' @param robustness_method Method for computing robustness weights. Options:
#'   "bisquare" (default), "huber", "talwar".
#' @param scaling_method Scaling method for robustness weight calculation.
#'   Options: "mad" (default), "mar" (Median Absolute Residual).
#' @param boundary_policy Handling of edge effects. Options: "extend" (default),
#'   "reflect", "zero", "noboundary".
#' @param auto_converge Tolerance for automatic convergence. NULL (default)
#'   disables.
#' @param return_diagnostics Logical, whether to compute cumulative fit
#'   quality metrics across all chunks. Default: FALSE.
#' @param return_robustness_weights Logical, whether to include robustness
#'   weights in output. Default: FALSE.
#' @param parallel Logical, whether to enable parallel chunk processing
#'   (default: TRUE). Chunks are processed in parallel via Rayon, significantly
#'   improving throughput for large datasets.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{x}: Sorted independent variable values.
#'   \item \code{y}: Smoothed dependent variable values.
#'   \item \code{fraction_used}: The fraction used for smoothing.
#'   \item \code{diagnostics}: Cumulative metrics (RMSE, etc.) (if requested).
#'   \item \code{robustness_weights}: Weights for each point (if requested).
#' }
#'
#' @examples
#' # Process a large dataset in chunks
#' n <- 20000
#' x <- seq(0, 100, length.out = n)
#' y <- sin(x / 10) + rnorm(n, sd = 0.5)
#'
#' # streaming approach maintains low memory footprint
#' result <- fastlowess_streaming(x, y, chunk_size = 5000L, overlap = 500L)
#'
#' plot(x, y, pch = ".")
#' lines(result$x, result$y, col = "red", lwd = 2)
#'
#' @seealso \code{\link{fastlowess}}, \code{\link{fastlowess_online}}
#' @family fastlowess
#' @export
fastlowess_streaming <- function(
  x,
  y,
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
  args <- validate_common_args(x, y, fraction, iterations)
  x <- args$x
  y <- args$y
  fraction <- args$fraction
  iterations <- args$iterations

  if (!is.numeric(chunk_size) || length(chunk_size) != 1 || chunk_size <= 0) {
    stop("chunk_size must be a positive integer")
  }

  chunk_size <- as.integer(chunk_size)

  if (!is.null(overlap)) {
    overlap <- as.integer(overlap)
  }

  # Call the Rust function
  .Call("wrap__fastlowess_streaming", x, y, fraction, chunk_size, overlap,
    iterations, delta, weight_function, robustness_method, scaling_method,
    boundary_policy, auto_converge, return_diagnostics,
    return_robustness_weights, parallel,
    PACKAGE = "rfastlowess"
  )
}
