# Print OnlineLowess Model

Print OnlineLowess Model

## Usage

``` r
# S3 method for class 'OnlineLowess'
print(x, ...)
```

## Arguments

- x:

  An OnlineLowess object.

- ...:

  Additional arguments.

## Value

The input object `x`, invisibly.

## Examples

``` r
model <- OnlineLowess(fraction = 0.2, window_capacity = 20L)
print(model)
#> <OnlineLowess Model>
#>   Fraction:          0.2 
#>   Window Capacity:   20 
#>   Min Points:        2 
#>   Update Mode:       
```
