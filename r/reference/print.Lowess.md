# Print Lowess Model

Print Lowess Model

## Usage

``` r
# S3 method for class 'Lowess'
print(x, ...)
```

## Arguments

- x:

  A Lowess object.

- ...:

  Additional arguments (ignored).

## Value

The input object `x`, invisibly.

## Examples

``` r
model <- Lowess(fraction = 0.3)
print(model)
#> <Lowess Model>
#>   Fraction:          0.3 
#>   Iterations:        3 
#>   Weight Function:   tricube 
#>   Parallel:          TRUE 
```
