# Download and Install the GPU-Enabled Backend

Downloads a prebuilt GPU-enabled rfastlowess shared library for the
current platform from the matching GitHub Release and installs it in
place of the current (CPU-only) library. GPU support is opt-in and not
included in CRAN/Bioconductor releases.

A running R session cannot swap an already-loaded shared library, so
**restart R** after installing for the change to take effect.

## Usage

``` r
install_gpu(yes = FALSE, local_path = NULL)
```

## Arguments

- yes:

  Logical; skip the interactive `y/N` confirmation prompt. Must be
  `TRUE` when the session is not interactive.

- local_path:

  Character; path to a GPU-enabled shared library already built locally
  (e.g. via `WITH_GPU=1 make install` in `benchmarks/`). When given,
  skips the GitHub Release lookup/download and installs directly from
  this path — useful for testing the installer itself, or installing an
  unreleased build.

## Value

Invisibly, `TRUE` if a GPU-enabled library is available at the printed
path (already active, or freshly installed); `FALSE` if the user
aborted.

## See also

[`gpu_available`](https://thisisamirv.github.io/lowess-project/r/reference/gpu_available.md)
to check the current status.

## Examples

``` r
# Check whether the GPU backend is already active before installing it
gpu_available()
#> [1] FALSE
if (interactive()) {
    install_gpu()
}
```
