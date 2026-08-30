# LOWESS Online Smoothing

Create a stateful LOWESS model for real-time online data. Maintains a
sliding window and processes each incoming point immediately via
[`add_point`](https://thisisamirv.github.io/lowess-project/r/reference/add_point.md).

## Usage

``` r
OnlineLowess(
  fraction = 0.67,
  window_capacity = 1000L,
  min_points = 3L,
  ...,
  iterations = 3L,
  delta = NULL,
  weight_function = "tricube",
  robustness_method = "bisquare",
  scaling_method = "mad",
  boundary_policy = "extend",
  zero_weight_fallback = "use_local_mean",
  update_mode = "full",
  auto_converge = NULL,
  return_robustness_weights = FALSE,
  return_diagnostics = FALSE,
  return_residuals = FALSE,
  parallel = FALSE,
  confidence_intervals = NULL,
  prediction_intervals = NULL
)
```

## Arguments

- fraction:

  Smoothing fraction, greater than 0 and up to 1. Default: 0.67. See
  Details for guidance on choosing a value.

- window_capacity:

  Maximum number of points kept in the sliding window, at least 3.
  Default: 1000.

- min_points:

  Minimum number of points required before smoothing begins, between 2
  and `window_capacity`. Default: 3.

- ...:

  Not used; forces all subsequent arguments to be named.

- iterations:

  Number of robustness iterations, between 0 and 1000 (inclusive).
  Default: 3.

- delta:

  Interpolation distance threshold, as a non-negative fraction of the x
  range; points within `delta` of each other on x share the same local
  fit. `NULL` (default) sets it automatically to 1/100th of the x range.

- weight_function:

  Kernel weight function. One of `"tricube"` (default), `"gaussian"`,
  `"uniform"` (alias: `"boxcar"`), `"cosine"`, `"epanechnikov"`,
  `"biweight"` (alias: `"bisquare"`), or `"triangle"` (alias:
  `"triangular"`).

- robustness_method:

  Outlier downweighting method: `"bisquare"` (default; alias:
  `"biweight"`), `"huber"`, or `"talwar"`.

- scaling_method:

  Residual scale estimation for robustness weights: `"mad"` (default;
  alias: `"median_absolute_deviation"`), `"mar"` (alias:
  `"median_absolute_residual"`), or `"mean"` (alias:
  `"mean_absolute_residual"`).

- boundary_policy:

  Boundary handling strategy: `"extend"` (default; alias: `"pad"`),
  `"reflect"` (alias: `"mirror"`), `"zero"`, or `"noboundary"` (alias:
  `"none"`).

- zero_weight_fallback:

  Fallback policy when all robustness weights drop to zero:
  `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`),
  `"return_original"` (alias: `"original"`), or `"return_none"` (alias:
  `"none"`).

- update_mode:

  Window update strategy: `"full"` (default; alias: `"resmooth"`)
  re-smooths all window points after each addition; `"incremental"`
  (alias: `"single"`) updates only the newest point.

- auto_converge:

  Convergence tolerance for early stopping of robustness iterations.
  `NULL` (default) disables early stopping.

- return_robustness_weights:

  Logical; if `TRUE`, return per-point robustness weights. Default:
  `FALSE`.

- return_diagnostics:

  Logical; if `TRUE`, return fit-quality metrics (RMSE, MAE, R-squared,
  AIC, etc.). Default: `FALSE`.

- return_residuals:

  Logical; if `TRUE`, return residuals in the result. Default: `FALSE`.

- parallel:

  Logical; enable parallel processing. Default: `TRUE`.

- confidence_intervals:

  Confidence level for confidence intervals, greater than 0 and less
  than 1 (e.g., 0.95). `NULL` (default) disables confidence intervals.

- prediction_intervals:

  Confidence level for prediction intervals, greater than 0 and less
  than 1 (e.g., 0.95). `NULL` (default) disables prediction intervals.

## Value

An OnlineLowess object.

## Details

Best suited when data arrives incrementally (e.g. sensors or streams),
real-time smoothed values are needed, or memory is fixed. For datasets
that fit in memory, see
[`Lowess`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md);
for large batches processed in chunks, see
[`StreamingLowess`](https://thisisamirv.github.io/lowess-project/r/reference/StreamingLowess.md).

## Examples

``` r
model <- OnlineLowess(fraction = 0.2, window_capacity = 20)
x <- 1:50
y <- sin(x * 0.1) + rnorm(50, 0, 0.1)
smoothed <- numeric(0)
for (i in seq_along(x)) {
    result <- add_point(model, x[i], y[i])
    if (!is.null(result)) smoothed <- c(smoothed, result$y)
}
head(smoothed, 5)
#> [1] 0.1898465 0.3098642 0.3037980 0.4955887 0.5883635
```
