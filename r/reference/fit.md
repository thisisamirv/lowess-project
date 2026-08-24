# Fit a LOWESS model to data

Fit a LOWESS model to data

## Usage

``` r
fit(model, ...)

# S3 method for class 'Lowess'
fit(model, x, y, custom_weights = NULL, ...)
```

## Arguments

- model:

  A `Lowess` object.

- ...:

  Must be empty.

- x:

  Numeric vector of predictor values.

- y:

  Numeric vector of response values.

- custom_weights:

  Optional numeric vector of non-negative per-observation weights.
  `NULL` (default) applies no custom weighting.

## Value

A `LowessResult` object.

## Examples

``` r
x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, 0, 0.1)
model <- Lowess(fraction = 0.2)
result <- fit(model, x, y)
```
