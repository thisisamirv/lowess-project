#' rfastlowess: High-performance LOWESS (Locally Weighted Scatterplot Smoothing)
#' for R
#'
#' @description
#' A production-ready, high-performance LOWESS implementation with comprehensive
#' features for robust nonparametric regression and trend estimation. Built on
#' the high-performance Rust crate `fastLowess`.
#'
#' ## What is LOWESS?
#'
#' LOWESS (Locally Weighted Scatterplot Smoothing) is a nonparametric regression
#' method that fits smooth curves through scatter plots. At each point, it fits
#' a weighted polynomial (typically linear) using nearby data points, with
#' weights decreasing smoothly with distance. This creates flexible,
#' data-adaptive curves without assuming a global functional form.
#'
#' **Key advantages:**
#' \itemize{
#'   \item No parametric assumptions about the underlying relationship
#'   \item Automatic adaptation to local data structure
#'   \item Robust to outliers (with robustness iterations enabled)
#'   \item Provides uncertainty estimates via confidence/prediction intervals
#'   \item Handles irregular sampling and missing regions gracefully
#' }
#'
#' **Common applications:**
#' \itemize{
#'   \item Exploratory data analysis and visualization
#'   \item Trend estimation in time series
#'   \item Baseline correction in spectroscopy and signal processing
#'   \item Quality control and process monitoring
#'   \item Genomic and epigenomic data smoothing
#'   \item Removing systematic effects in scientific measurements
#' }
#'
#' @section Main Functions:
#' \itemize{
#'   \item \code{\link{fastlowess}}: Primary interface for batch processing
#'   \item \code{\link{fastlowess_streaming}}: Chunked for large datasets
#'   \item \code{\link{fastlowess_online}}: Sliding window for real-time
#'     data
#' }
#'
#' @section Design Notes:
#' rfastlowess is designed for performance and production use. It supports:
#' \itemize{
#'   \item **Parallel Execution**: Multi-core fits via Rust's Rayon (enabled by
#'     default)
#'   \item **Robust Statistics**: MAD-based scale estimation and IRLS with
#'     Bisquare, Huber, or Talwar weighting
#'   \item **Optimized Performance**: Delta optimization for skipping dense
#'     regions
#'   \item **Uncertainty**: Numerical calculation of point-wise standard errors
#'     and intervals
#'   \item **Parameters**: Automated bandwidth selection via cross-validation
#' }
#'
#' @examples
#' # Run available demos:
#' demo(package = "rfastlowess")
#'
#' # 1. Batch Smoothing Demo
#' # Demonstrates basic batch processing on a static dataset.
#' demo("batch_smoothing", package = "rfastlowess", ask = FALSE)
#'
#' # 2. Online Smoothing Demo
#' # Shows sliding window smoothing simulating real-time data updates.
#' demo("online_smoothing", package = "rfastlowess", ask = FALSE)
#'
#' # 3. Streaming Smoothing Demo
#' # Illustrates chunked processing for datasets larger than memory.
#' demo("streaming_smoothing", package = "rfastlowess", ask = FALSE)
#'
#' @useDynLib rfastlowess, .registration = TRUE
#' @importFrom stats smooth
"_PACKAGE"
