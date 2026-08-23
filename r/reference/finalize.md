# Finalize a streaming LOWESS model

Finalize a streaming LOWESS model

## Usage

``` r
finalize(model, ...)
```

## Arguments

- model:

  A `StreamingLowess` object.

- ...:

  Not used.

## Value

A `LowessResult` combining all processed chunks.

## Examples

``` r
x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, 0, 0.1)
model <- StreamingLowess(fraction = 0.2, chunk_size = 50L)
invisible(process_chunk(model, x[1:50], y[1:50]))
final <- finalize(model)
```
