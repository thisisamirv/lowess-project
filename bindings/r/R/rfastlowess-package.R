#' rfastlowess: High-performance LOWESS Smoothing for R
#'
#' @description
#' A high-performance LOWESS (Locally Weighted Scatterplot Smoothing)
#' implementation built on the Rust `fastLowess` crate.
#'
#' @srrstats G1.0 Package-level documentation for statistical software review.
#' @srrstats G1.1 thin R wrapper interface for core Rust algorithms.
#'
#' @section Main Classes:
#' \itemize{
#'   \item \code{\link{Lowess}}: Primary interface for batch processing
#'   \item \code{\link{StreamingLowess}}: Chunked processing for large
#'     datasets
#'   \item \code{\link{OnlineLowess}}: Sliding window for real-time data
#' }
#'
#' @section Documentation:
#' For comprehensive documentation, tutorials, and API reference, see:
#' \url{https://lowess.readthedocs.io/}
#'
#' @examples
#' # Basic smoothing
#' x <- seq(1, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, sd = 0.2)
#' model <- Lowess(fraction = 0.3)
#' result <- model$fit(x, y)
#' plot(x, y)
#' lines(result$x, result$y, col = "red", lwd = 2)
#'
#' @useDynLib rfastlowess, .registration = TRUE
#' @importFrom stats smooth
"_PACKAGE"
