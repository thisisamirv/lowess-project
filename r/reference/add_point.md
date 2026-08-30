# Add a single point to an online LOWESS model

Add a single point to an online LOWESS model

## Usage

``` r
add_point(model, ...)

# S3 method for class 'OnlineLowess'
add_point(model, x, y, ...)
```

## Arguments

- model:

  An `OnlineLowess` object.

- ...:

  Must be empty.

- x:

  A single numeric x value.

- y:

  A single numeric y value.

## Value

An online result list, or `NULL` if fewer than `min_points` have been
added.

## Examples

``` r
model <- OnlineLowess(fraction = 0.2, window_capacity = 20L)
add_point(model, 1.0, 0.5)
#> NULL
```
