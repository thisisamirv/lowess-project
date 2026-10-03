# Interval options for LOWESS models

Build an interval options list to pass to `intervals = ...` in
[`Lowess`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md),
[`StreamingLowess`](https://thisisamirv.github.io/lowess-project/r/reference/StreamingLowess.md),
[`OnlineLowess`](https://thisisamirv.github.io/lowess-project/r/reference/OnlineLowess.md),
or
[`predict.Lowess`](https://thisisamirv.github.io/lowess-project/r/reference/predict.Lowess.md).

## Usage

``` r
intervals_opts(confidence = NULL, prediction = NULL, bootstrap = 0L)
```

## Arguments

- confidence:

  Coverage for confidence intervals, greater than 0 and less than 1
  (e.g. 0.90). `NULL` (default) disables them.

- prediction:

  Coverage for prediction intervals, greater than 0 and less than 1
  (e.g. 0.99). `NULL` (default) disables them. This level is independent
  of `confidence`.

- bootstrap:

  Number of bootstrap resamples used to compute the intervals. `0`
  (default) uses the analytic intervals.

## Value

An `intervals_opts` list.

## Examples

``` r
model <- Lowess(
    intervals = intervals_opts(
        confidence = 0.90, prediction = 0.99, bootstrap = 200
    ),
    seed = 42
)
```
