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
    ...,
    iterations = 3L,
    weight_function = "tricube",
    robustness_method = "bisquare",
    delta = NULL,
    zero_weight_fallback = "use_local_mean",
    boundary_policy = "extend",
    scaling_method = "mad",
    auto_converge = NULL,
    missing = "error",
    chunk_size = 5000L,
    overlap = NULL,
    merge_strategy = "weighted_average",
    parallel = TRUE,
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

  Number of robustness iterations, between 0 and 1000 (inclusive).
  Default: 3.

- weight_function:

  Kernel weight function. One of `"tricube"` (default), `"gaussian"`,
  `"uniform"` (alias: `"boxcar"`), `"cosine"`, `"epanechnikov"`,
  `"biweight"` (alias: `"bisquare"`), or `"triangle"` (alias:
  `"triangular"`).

- robustness_method:

  Outlier downweighting method: `"bisquare"` (default; alias:
  `"biweight"`), `"huber"`, or `"talwar"`.

- delta:

  Interpolation distance threshold, as a non-negative fraction of the x
  range; points within `delta` of each other on x share the same local
  fit. `NULL` (default) sets it automatically to 1/100th of the x range.

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

  Convergence tolerance for early stopping of robustness iterations.
  `NULL` (default) disables early stopping.

- missing:

  Policy for non-finite (NaN/Infinity) values in input data: `"error"`
  (default) raises an error, `"drop"` silently removes affected
  observations before fitting.

- chunk_size:

  Number of data points per processing chunk, at least 10. Default:
  5000.

- overlap:

  Number of overlapping points between consecutive chunks, less than
  `chunk_size`. `NULL` (default) computes `chunk_size / 10`, clamped to
  at least 1 and less than `chunk_size`.

- merge_strategy:

  Strategy for reconciling overlapping chunk regions:
  `"weighted_average"` (default; alias: `"weighted"`), `"average"`
  (alias: `"mean"`), `"take_first"` (alias: `"first"`), or `"take_last"`
  (alias: `"last"`).

- parallel:

  Logical; enable parallel processing. Default: `TRUE`.

- outputs:

  Streaming output choices: `"se"` (standard errors), `"diagnostics"`,
  `"residuals"`, `"weights"` (robustness weights), and/or
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

A StreamingLowess object.

## Details

Best suited for datasets over 100,000 points, memory-constrained
environments, or batch processing pipelines. For smaller datasets that
fit in memory, see
[`Lowess`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md);
for point-by-point real-time data, see
[`OnlineLowess`](https://thisisamirv.github.io/lowess-project/r/reference/OnlineLowess.md).
When `"diagnostics"` is selected, `residual_sd` is the cumulative sample
standard deviation of emitted residuals.

Confidence and prediction interval coverage levels are independent.

Overlapping regions between chunks are reconciled via `merge_strategy`:

|                              |            |                            |
|------------------------------|------------|----------------------------|
| Strategy                     | Alias      | Behavior                   |
| "weighted_average" (default) | "weighted" | Distance-weighted blend    |
| "average"                    | "mean"     | Average overlapping values |
| "take_first"                 | "first"    | Keep left chunk values     |
| "take_last"                  | "last"     | Keep right chunk values    |

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
