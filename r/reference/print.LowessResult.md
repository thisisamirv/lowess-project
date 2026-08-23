# Print Lowess Result

Print Lowess Result

## Usage

``` r
# S3 method for class 'LowessResult'
print(x, ...)
```

## Arguments

- x:

  A LowessResult object.

- ...:

  Additional arguments (ignored).

## Value

The input object `x`, invisibly.

## Examples

``` r
x <- seq(0, 10, length.out = 50)
y <- sin(x) + rnorm(50, 0, 0.1)
model <- Lowess(fraction = 0.3)
result <- fit(model, x, y)
print(result)
#> <LowessResult>
#>   Points:            50 
#>   Fraction Used:     0.3 
```
