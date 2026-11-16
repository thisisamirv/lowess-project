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

A list with `status` (`"filling"`, `"ready"`, or `"dropped"`) and
`point` (an online-output list when ready; otherwise `NULL`).

## Examples

``` r
model <- OnlineLowess(fraction = 0.2, window_capacity = 20L, min_points = 2L)
invisible(add_point(model, 1.0, 0.5))
add_point(model, 2.0, 0.6)
#> $status
#> [1] "ready"
#> 
#> $point
#> $point$y
#> [1] 0.6
#> 
#> $point$residual
#> [1] 0
#> 
#> $point$robustness_weight
#> [1] 1
#> 
#> $point$iterations_used
#> [1] 0
#> 
#> 
```
