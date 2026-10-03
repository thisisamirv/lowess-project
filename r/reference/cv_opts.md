# Cross-validation options for [`Lowess`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md)

Build a cross-validation options list to pass to `Lowess(cv = ...)`.
Cross-validation is disabled by default (`cv = NULL`); supplying
`cv_opts()` enables it and lets the model pick the best smoothing
fraction from the candidates.

## Usage

``` r
cv_opts(method = "kfold", k = 5L, fractions)
```

## Arguments

- method:

  Cross-validation method: `"kfold"` (default) or `"loocv"`.

- k:

  Number of folds for k-fold cross-validation. Default: 5.

- fractions:

  Numeric vector of candidate smoothing fractions, each greater than 0
  and up to 1 (e.g. `c(0.2, 0.3, 0.5)`).

## Value

A `cv_opts` list for `Lowess(cv = ...)`. Use `Lowess(seed = ...)` for
reproducible fold assignment.

## Examples

``` r
model <- Lowess(
    cv = cv_opts(method = "kfold", k = 5, fractions = c(0.2, 0.3, 0.5)),
    seed = 42
)
```
