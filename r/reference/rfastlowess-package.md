# rfastlowess: High-performance LOWESS Smoothing for R

A high-performance LOWESS (Locally Weighted Scatterplot Smoothing)
implementation built on the Rust `fastLowess` crate.

## Main Classes

- [`Lowess`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md):
  Primary interface for batch processing

- [`StreamingLowess`](https://thisisamirv.github.io/lowess-project/r/reference/StreamingLowess.md):
  Chunked processing for large datasets

- [`OnlineLowess`](https://thisisamirv.github.io/lowess-project/r/reference/OnlineLowess.md):
  Sliding window for real-time data

## Documentation

For comprehensive documentation, tutorials, and API reference, see:
<https://lowess.readthedocs.io/>

## See also

Useful links:

- <https://github.com/thisisamirv/lowess-project>

- Report bugs at <https://github.com/thisisamirv/lowess-project/issues>

## Author

**Maintainer**: Amir Valizadeh <thisisamirv@gmail.com>
([ORCID](https://orcid.org/0000-0001-5983-8527)) \[funder\]

Authors:

- Amir Valizadeh <thisisamirv@gmail.com>
  ([ORCID](https://orcid.org/0000-0001-5983-8527)) \[funder\]

## Examples

``` r
# Basic smoothing
x <- seq(1, 10, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.2)
model <- Lowess(fraction = 0.3)
result <- fit(model, x, y)
plot(x, y)
lines(result$x, result$y, col = "red", lwd = 2)

```
