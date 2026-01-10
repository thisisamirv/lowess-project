# R Examples

Complete R examples demonstrating rfastlowess capabilities with base R and visualization.

## Batch Smoothing

Process complete datasets with confidence intervals, diagnostics, and cross-validation.

```r
--8<-- "examples/r/batch_smoothing.R"
```

[:material-download: Download batch_smoothing.R](https://github.com/thisisamirv/lowess-project/blob/main/examples/r/batch_smoothing.R)

---

## Streaming Smoothing

Process large datasets in memory-efficient chunks with overlap merging.

```r
--8<-- "examples/r/streaming_smoothing.R"
```

[:material-download: Download streaming_smoothing.R](https://github.com/thisisamirv/lowess-project/blob/main/examples/r/streaming_smoothing.R)

---

## Online Smoothing

Real-time smoothing with sliding window for streaming data applications.

```r
--8<-- "examples/r/online_smoothing.R"
```

[:material-download: Download online_smoothing.R](https://github.com/thisisamirv/lowess-project/blob/main/examples/r/online_smoothing.R)

---

## Running the Examples

```bash
# Install the package first
# From R:
# install.packages("rfastlowess")

# Or from source:
# R CMD INSTALL bindings/r

# Run examples
Rscript examples/r/batch_smoothing.R
Rscript examples/r/streaming_smoothing.R
Rscript examples/r/online_smoothing.R
```

## Quick Start

```r
library(rfastlowess)

# Generate sample data
set.seed(42)
x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

# Basic smoothing
result <- fastlowess(x, y, fraction = 0.3)

# With confidence intervals
result <- fastlowess(x, y, 
                     fraction = 0.3, 
                     confidence_intervals = 0.95,
                     return_diagnostics = TRUE)

# Access results
plot(x, y, pch = 19, col = "gray")
lines(result$x, result$y, col = "blue", lwd = 2)
```
