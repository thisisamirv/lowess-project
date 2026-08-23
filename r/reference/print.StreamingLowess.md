# Print StreamingLowess Model

Print StreamingLowess Model

## Usage

``` r
# S3 method for class 'StreamingLowess'
print(x, ...)
```

## Arguments

- x:

  A StreamingLowess object.

- ...:

  Additional arguments.

## Value

The input object `x`, invisibly.

## Examples

``` r
model <- StreamingLowess(fraction = 0.3, chunk_size = 50L)
print(model)
#> <StreamingLowess Model>
#>   Fraction:          0.3 
#>   Chunk Size:        50 
#>   Parallel:          TRUE 
```
