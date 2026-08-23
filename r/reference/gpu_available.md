# Check GPU Backend Availability

Returns whether the currently loaded rfastlowess shared library was
built with the GPU backend enabled.

## Usage

``` r
gpu_available()
```

## Value

Logical; `TRUE` if the GPU backend is active.

## See also

[`install_gpu`](https://thisisamirv.github.io/lowess-project/r/reference/install_gpu.md)
to download and install it.

## Examples

``` r
gpu_available()
#> [1] FALSE
```
