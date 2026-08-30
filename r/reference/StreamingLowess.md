# LOWESS Streaming Smoothing

Create a stateful LOWESS model for streaming data. Processes data in
fixed-size chunks with configurable overlap: results for each chunk are
returned by
[`process_chunk`](https://thisisamirv.github.io/lowess-project/r/reference/process_chunk.md),
and
[`finalize`](https://thisisamirv.github.io/lowess-project/r/reference/finalize.md)
flushes any remaining buffered points after the last chunk.

## Usage

``` r
StreamingLowess(
  fraction = 0.67,
  chunk_size = 5000L,
  ...,
  overlap = NULL,
  iterations = 3L,
  delta = NULL,
  weight_function = "tricube",
  robustness_method = "bisquare",
  scaling_method = "mad",
  boundary_policy = "extend",
  zero_weight_fallback = "use_local_mean",
  auto_converge = NULL,
  return_diagnostics = FALSE,
  return_residuals = FALSE,
  return_robustness_weights = FALSE,
  merge_strategy = "weighted_average",
  parallel = TRUE,
  confidence_intervals = NULL,
  prediction_intervals = NULL
)
```

## Arguments

- fraction:

  Smoothing fraction, greater than 0 and up to 1. Default: 0.67. See
  Details for guidance on choosing a value.

- chunk_size:

  Number of data points per processing chunk, at least 10. Default:
  5000.

- ...:

  Not used; forces all subsequent arguments to be named.

- overlap:

  Number of overlapping points between consecutive chunks, less than
  `chunk_size`. `NULL` (default) uses the backend's default of 500.

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

- auto_converge:

  Convergence tolerance for early stopping of robustness iterations.
  `NULL` (default) disables early stopping.

- return_diagnostics:

  Logical; if `TRUE`, return fit-quality metrics (RMSE, MAE, R-squared,
  AIC, etc.). Default: `FALSE`.

- return_residuals:

  Logical; if `TRUE`, return residuals in the result. Default: `FALSE`.

- return_robustness_weights:

  Logical; if `TRUE`, return per-point robustness weights. Default:
  `FALSE`.

- merge_strategy:

  Strategy for reconciling overlapping chunk regions:
  `"weighted_average"` (default; alias: `"weighted"`), `"average"`
  (alias: `"mean"`), `"take_first"` (alias: `"first"`), or `"take_last"`
  (alias: `"last"`).

- parallel:

  Logical; enable parallel processing. Default: `TRUE`.

- confidence_intervals:

  Confidence level for confidence intervals, greater than 0 and less
  than 1 (e.g., 0.95). `NULL` (default) disables confidence intervals.

- prediction_intervals:

  Confidence level for prediction intervals, greater than 0 and less
  than 1 (e.g., 0.95). `NULL` (default) disables prediction intervals.

## Value

A StreamingLowess object.

## Details

Best suited for datasets over 100,000 points, memory-constrained
environments, or batch processing pipelines. For smaller datasets that
fit in memory, see
[`Lowess`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md);
for point-by-point real-time data, see
[`OnlineLowess`](https://thisisamirv.github.io/lowess-project/r/reference/OnlineLowess.md).

Overlapping regions between chunks are reconciled via `merge_strategy`:

|                      |                                                |
|----------------------|------------------------------------------------|
| Strategy             | Behavior                                       |
| `"average"`          | Arithmetic mean of both estimates              |
| `"weighted_average"` | Distance-weighted blend (recommended, default) |
| `"take_first"`       | Keep left-chunk estimate                       |
| `"take_last"`        | Keep right-chunk estimate                      |

## Examples

``` r
x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, 0, 0.1)
model <- StreamingLowess(fraction = 0.2, chunk_size = 50)
res1 <- process_chunk(model, x[1:50], y[1:50])
res2 <- process_chunk(model, x[51:100], y[51:100])
finalize(model)
#> <LowessResult>
#>   Points:            5 
#>   Fraction Used:     0.2 
```
