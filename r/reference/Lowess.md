# LOWESS Batch Smoothing

Create a stateful LOWESS model for batch smoothing. This is the default
mode: it processes the entire dataset at once and supports every feature
(confidence/prediction intervals, cross-validation, GPU backend).

## Usage

``` r
Lowess(
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
    parallel = TRUE,
    backend = "cpu",
    outputs = NULL,
    intervals = NULL,
    cv = NULL,
    seed = NULL,
    retain_model = FALSE
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

- parallel:

  Logical; enable parallel processing. Default: `TRUE`.

- backend:

  Execution backend: `"cpu"` (default) or `"gpu"`. GPU support requires
  the package to be built locally with `WITH_GPU=1` (see
  `bindings/r/Makefile`) and a Vulkan/Metal/DX12-capable GPU driver; not
  available in released CRAN/Bioconductor binaries.

- outputs:

  Character vector selecting optional output components: `"se"`
  (standard errors), `"diagnostics"`, `"residuals"`, `"weights"`
  (robustness weights), `"derivative"`, and/or `"sorted"`. `NULL`
  (default) returns only the core result (`x`, `y`, and fit metadata).

- intervals:

  Interval options, created with
  [`intervals_opts`](https://thisisamirv.github.io/lowess-project/r/reference/intervals_opts.md)
  (or a named list with any of `confidence`, `prediction`, `bootstrap`):
  e.g.
  `intervals = intervals_opts(confidence = 0.90, prediction = 0.99, bootstrap = 200)`.
  Confidence and prediction coverage levels are independent and may
  differ. `NULL` (default) disables intervals.

- cv:

  Cross-validation options, created with
  [`cv_opts`](https://thisisamirv.github.io/lowess-project/r/reference/cv_opts.md):
  e.g.
  `cv = cv_opts(method = "kfold", k = 5, fractions = c(0.2, 0.3, 0.5))`.
  `NULL` (default) disables cross-validation.

- seed:

  Non-negative whole-number seed shared by cross-validation and
  bootstrap resampling for reproducible results. When cross-validation
  is unavailable, it controls bootstrap resampling only. `NULL`
  (default) uses a random seed.

- retain_model:

  Logical; if `TRUE`, retain the fitted model's training data, enabling
  [`predict.Lowess`](https://thisisamirv.github.io/lowess-project/r/reference/predict.Lowess.md)
  for out-of-sample prediction. Default: `FALSE`.

## Value

A Lowess object.

## Details

Best suited when the dataset fits in memory and you need intervals,
cross-validation, or diagnostics. For datasets that don't fit in memory
or arrive in chunks, see
[`StreamingLowess`](https://thisisamirv.github.io/lowess-project/r/reference/StreamingLowess.md);
for point-by-point real-time data, see
[`OnlineLowess`](https://thisisamirv.github.io/lowess-project/r/reference/OnlineLowess.md).
When `"diagnostics"` is selected, `residual_sd` is the robust residual
scale estimate (`1.4826 * MAD`).

`fraction` is the most important parameter: it controls the size of the
local neighbourhood used at each point.

|         |                 |                          |
|---------|-----------------|--------------------------|
| Range   | Effect          | Use case                 |
| 0.1-0.3 | Fine detail     | Rapidly changing signals |
| 0.3-0.5 | Balanced        | General purpose          |
| 0.5-0.7 | Heavy smoothing | Noisy data               |
| 0.7-1.0 | Very smooth     | Trend extraction         |

## Examples

``` r
x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, 0, 0.1)
model <- Lowess(fraction = 0.2)
result <- fit(model, x, y)
plot(x, y)
lines(x, result$y, col = "red")
```
