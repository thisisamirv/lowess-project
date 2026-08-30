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
    parallel = TRUE,
    cv_seed = NULL,
    return_se = FALSE,
    backend = "cpu"
)
```

## Arguments

- fraction:

  Smoothing fraction, greater than 0 and up to 1. Default: 0.67. See
  Details for guidance on choosing a value.

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

- confidence_intervals:

  Confidence level for confidence intervals, greater than 0 and less
  than 1 (e.g., 0.95). `NULL` (default) disables confidence intervals.

- prediction_intervals:

  Confidence level for prediction intervals, greater than 0 and less
  than 1 (e.g., 0.95). `NULL` (default) disables prediction intervals.

- return_diagnostics:

  Logical; if `TRUE`, return fit-quality metrics (RMSE, MAE, R-squared,
  AIC, etc.). Default: `FALSE`.

- return_residuals:

  Logical; if `TRUE`, return residuals in the result. Default: `FALSE`.

- return_robustness_weights:

  Logical; if `TRUE`, return per-point robustness weights. Default:
  `FALSE`.

- zero_weight_fallback:

  Fallback policy when all robustness weights drop to zero:
  `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`),
  `"return_original"` (alias: `"original"`), or `"return_none"` (alias:
  `"none"`).

- auto_converge:

  Convergence tolerance for early stopping of robustness iterations.
  `NULL` (default) disables early stopping.

- cv_fractions:

  Numeric vector of candidate fractions for cross-validation. `NULL`
  (default) disables CV.

- cv_method:

  Cross-validation method: `"kfold"` (default) or `"loocv"`.

- cv_k:

  Number of folds for k-fold CV. Default: 5.

- parallel:

  Logical; enable parallel processing. Default: `TRUE`.

- cv_seed:

  Integer seed for the cross-validation random number generator. `NULL`
  (default) uses a random seed.

- return_se:

  Logical; if `TRUE`, compute hat-matrix statistics (effective degrees
  of freedom, leverage, standard errors). Default: `FALSE`.

- backend:

  Execution backend: `"cpu"` (default) or `"gpu"`. GPU support requires
  the package to be built locally with `WITH_GPU=1` (see
  `bindings/r/Makefile`) and a Vulkan/Metal/DX12-capable GPU driver; not
  available in released CRAN/Bioconductor binaries.

## Value

A Lowess object.

## Details

Best suited when the dataset fits in memory and you need intervals,
cross-validation, or diagnostics. For datasets that don't fit in memory
or arrive in chunks, see
[`StreamingLowess`](https://thisisamirv.github.io/lowess-project/r/reference/StreamingLowess.md);
for point-by-point real-time data, see
[`OnlineLowess`](https://thisisamirv.github.io/lowess-project/r/reference/OnlineLowess.md).

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
