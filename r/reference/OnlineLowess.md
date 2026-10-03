# LOWESS Online Smoothing

Create a stateful LOWESS model for real-time online data. Maintains a
sliding window and processes each incoming point immediately via
[`add_point`](https://thisisamirv.github.io/lowess-project/r/reference/add_point.md).

## Usage

``` r
OnlineLowess(
    fraction = 0.67,
    ...,
    iterations = 0L,
    weight_function = "tricube",
    robustness_method = "bisquare",
    delta = NULL,
    zero_weight_fallback = "use_local_mean",
    boundary_policy = "extend",
    scaling_method = "mad",
    auto_converge = NULL,
    missing = "error",
    window_capacity = 1000L,
    min_points = 2L,
    update_mode = "incremental",
    outputs = NULL,
    intervals = NULL,
    seed = NULL
)
```

## Arguments

- fraction:

  Smoothing fraction, greater than 0 and up to 1. Default: 0.67.

- ...:

  Not used; forces all subsequent arguments to be named.

- iterations:

  Number of robustness iterations. Requires `update_mode = "full"`; the
  default `"incremental"` mode is a non-robust single-point fit that
  ignores robustness iterations. Default: 0.

- weight_function:

  Kernel weight function. One of `"tricube"` (default), `"gaussian"`,
  `"uniform"` (alias: `"boxcar"`), `"cosine"`, `"epanechnikov"`,
  `"biweight"` (alias: `"bisquare"`), or `"triangle"` (alias:
  `"triangular"`).

- robustness_method:

  Outlier downweighting method: `"bisquare"` (default; alias:
  `"biweight"`), `"huber"`, or `"talwar"`.

- delta:

  Interpolation distance as a non-negative fraction of the x range. In
  Online mode, `NULL` (default) disables interpolation; positive values
  require `update_mode = "full"`.

- zero_weight_fallback:

  Fallback policy when all robustness weights drop to zero:
  `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`),
  `"return_original"` (alias: `"original"`), or `"return_none"` (alias:
  `"none"`).

- boundary_policy:

  Boundary handling strategy: `"extend"` (default; alias: `"pad"`),
  `"reflect"` (alias: `"mirror"`), `"zero"`, or `"noboundary"` (alias:
  `"none"`).

- scaling_method:

  Residual scale estimation for robustness weights: `"mad"` (default;
  alias: `"median_absolute_deviation"`), `"mar"` (alias:
  `"median_absolute_residual"`), or `"mean"` (alias:
  `"mean_absolute_residual"`).

- auto_converge:

  Convergence tolerance for early stopping. Requires
  `update_mode = "full"` and `iterations > 0`; `NULL` (default) disables
  early stopping.

- missing:

  Policy for non-finite (NaN/Infinity) values in input data: `"error"`
  (default) raises an error, `"drop"` silently removes affected
  observations before fitting.

- window_capacity:

  Maximum number of points kept in the sliding window, at least 3.
  Default: 1000.

- min_points:

  Minimum number of points required before smoothing begins, between 2
  and `window_capacity`. Default: 2.

- update_mode:

  Window update strategy: `"incremental"` (default; alias: `"single"`)
  updates only the newest point and does not run robustness iterations;
  `"full"` (alias: `"resmooth"`) re-smooths all window points after each
  addition and supports robustness iterations, positive `delta`, and
  `auto_converge`.

- outputs:

  Character vector selecting optional Online output components: `"se"`
  (standard errors), `"weights"` (robustness weights), and/or
  `"derivative"`. `NULL` (default) returns only the core result.

- intervals:

  Interval options, created with
  [`intervals_opts`](https://thisisamirv.github.io/lowess-project/r/reference/intervals_opts.md)
  (or a named list with any of `confidence`, `prediction`, `bootstrap`):
  e.g.
  `intervals = intervals_opts(confidence = 0.90, prediction = 0.99, bootstrap = 200)`.
  Confidence and prediction coverage levels are independent and may
  differ. `NULL` (default) disables intervals.

- seed:

  Non-negative whole-number seed shared by cross-validation and
  bootstrap resampling for reproducible results. When cross-validation
  is unavailable, it controls bootstrap resampling only. `NULL`
  (default) uses a random seed.

## Value

An OnlineLowess object.

## Details

Best suited when data arrives incrementally (e.g. sensors or streams),
real-time smoothed values are needed, or memory is fixed. For datasets
that fit in memory, see
[`Lowess`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md);
for large batches processed in chunks, see
[`StreamingLowess`](https://thisisamirv.github.io/lowess-project/r/reference/StreamingLowess.md).

Confidence and prediction interval coverage levels are independent; both
interval types require `update_mode = "full"`. Positive `delta` values
also require `update_mode = "full"`. A non-`NULL` `auto_converge`
requires full mode and at least one robustness iteration; unsupported
combinations are rejected.

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
#> [1] 0.1201261 0.1898465 0.3098642 0.3037980 0.4955887
```
