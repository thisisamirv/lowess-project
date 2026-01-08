#' rfastlowess: High-performance LOWESS Smoothing for R
#'
#' @description
#' A high-performance LOWESS (Locally Weighted Scatterplot Smoothing)
#' implementation built on the Rust `fastLowess` crate.
#'
#' @section Main Functions:
#' \itemize{
#'   \item \code{\link{fastlowess}}: Primary interface for batch processing
#'   \item \code{\link{fastlowess_streaming}}: Chunked processing for large
#'     datasets
#'   \item \code{\link{fastlowess_online}}: Sliding window for real-time data
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
#' result <- fastlowess(x, y, fraction = 0.3)
#' plot(x, y)
#' lines(result$x, result$y, col = "red", lwd = 2)
#'
#' @useDynLib rfastlowess, .registration = TRUE
#' @importFrom stats smooth
"_PACKAGE"
