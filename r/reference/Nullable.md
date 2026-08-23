# Nullable Value Wrapper

Wraps a value to be passed to Rust as an Option.

## Usage

``` r
Nullable(x)
```

## Arguments

- x:

  Value to wrap or NULL.

## Value

The value itself. This is a helper for rextendr conversion.

## Examples

``` r
Nullable(5)
#> [1] 5
Nullable(NULL)
#> NULL
```
