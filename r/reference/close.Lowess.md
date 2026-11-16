# Release native resources held by a LOWESS model

S3 methods for base [`close`](https://rdrr.io/r/base/connections.html).
For `Lowess`, clears retained predict state early. For `StreamingLowess`
and `OnlineLowess`, releases the native adapter. Safe to call multiple
times.

## Usage

``` r
# S3 method for class 'Lowess'
close(con, ...)

# S3 method for class 'StreamingLowess'
close(con, ...)

# S3 method for class 'OnlineLowess'
close(con, ...)
```

## Arguments

- con:

  A `Lowess`, `StreamingLowess`, or `OnlineLowess` object.

- ...:

  Must be empty.

## Value

`NULL`, invisibly.

## Examples

``` r
model <- Lowess(fraction = 0.3, retain_model = TRUE)
invisible(fit(model, 1:10, sin(1:10)))
close(model)
```
