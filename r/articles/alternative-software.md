# Alternative Software

## Overview

`rfastlowess` is presented throughout this package’s documentation as a
faster, more feature-rich alternative to base R’s
[`stats::lowess()`](https://rdrr.io/r/stats/lowess.html). This page is
for readers who already use
[`stats::lowess()`](https://rdrr.io/r/stats/lowess.html) and want to
know: *can I get the exact same numbers out of `rfastlowess`, and if
not, why not?*

In short:

- With two options set (`boundary_policy = "noboundary"` and
  `scaling_method = "mar"`), `rfastlowess` reproduces
  [`stats::lowess()`](https://rdrr.io/r/stats/lowess.html) to within
  floating-point tolerance (~1e-10) — see [Reproducing
  `stats::lowess()`](#reproducing-statslowess) below.
- Outside of those two options, `rfastlowess`’s *defaults* intentionally
  differ from [`stats::lowess()`](https://rdrr.io/r/stats/lowess.html)
  (see [Why the defaults differ](#why-the-defaults-differ)), and it
  supports a number of features
  [`stats::lowess()`](https://rdrr.io/r/stats/lowess.html) doesn’t have
  (see [What this package adds](#what-this-package-adds)).

------------------------------------------------------------------------

## Reproducing `stats::lowess()`

[`stats::lowess()`](https://rdrr.io/r/stats/lowess.html) and
`rfastlowess`’s
[`Lowess()`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md)
implement the same underlying algorithm, but two defaults differ:

| Option | [`stats::lowess()`](https://rdrr.io/r/stats/lowess.html) | [`Lowess()`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md) default |
|----|----|----|
| Boundary padding | None (`"noboundary"`) | `"extend"` |
| Residual scaling | MAR: `median(\|r\|)` | `"mad"`: `median(\|r - median(r)\|)` |
| `fraction` | `f = 2/3` | `0.67` |
| `iterations` | `iter = 3` | `3` |
| `delta` | `0.01 * diff(range(x))` | same (`NULL` -\> auto) |

Setting the first two options to match R,
[`Lowess()`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md)
reproduces [`stats::lowess()`](https://rdrr.io/r/stats/lowess.html)
exactly (up to floating-point rounding):

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 60)
y <- sin(x) + rnorm(60, sd = 0.2)

reference <- stats::lowess(x, y, f = 2 / 3, iter = 3)

model <- Lowess(
    fraction = 2 / 3,
    iterations = 3L,
    boundary_policy = "noboundary",
    scaling_method = "mar"
)
result <- fit(model, x, y)

cat("Max abs difference:", max(abs(result$y - reference$y)), "\n")
#> Max abs difference: 1.44329e-15
```

If your `x` is not already sorted, add `outputs = "sorted"` to get
results ordered the same way
[`stats::lowess()`](https://rdrr.io/r/stats/lowess.html) returns them
(ascending by `x`):

``` r

set.seed(1)
x <- seq(0, 2 * pi, length.out = 60)
x_unsorted <- sample(x)
y_unsorted <- sin(x_unsorted) + rnorm(60, sd = 0.2)

reference <- stats::lowess(x_unsorted, y_unsorted, f = 2 / 3, iter = 3)

model <- Lowess(
    fraction = 2 / 3,
    iterations = 3L,
    boundary_policy = "noboundary",
    scaling_method = "mar",
    outputs = "sorted"
)
result <- fit(model, x_unsorted, y_unsorted)

cat("x is sorted ascending:", all(diff(result$x) >= 0), "\n")
#> x is sorted ascending: TRUE
cat("Max abs difference:", max(abs(result$y - reference$y)), "\n")
#> Max abs difference: 1.554312e-15
```

Without `outputs = "sorted"`,
[`Lowess()`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md)
still fits on sorted `x` internally (as
[`stats::lowess()`](https://rdrr.io/r/stats/lowess.html) requires) but
returns values reordered back to match your original input order —
useful when you need the fit aligned with other columns in a data frame,
but not what you want for a value-by-value comparison against
[`stats::lowess()`](https://rdrr.io/r/stats/lowess.html)’s own (always
sorted) output.

------------------------------------------------------------------------

## Why the defaults differ

`rfastlowess`’s *defaults* (as opposed to what it’s capable of
reproducing) deliberately depart from
[`stats::lowess()`](https://rdrr.io/r/stats/lowess.html) in two places:

**Boundary padding** (default `boundary_policy = "extend"` vs. no
padding). Without padding, the local neighbourhood at the first and last
few points is one-sided, which biases the fit toward the interior and
increases variance right at the edges. `"extend"` (and the other padding
policies — see
[`vignette("boundary")`](https://thisisamirv.github.io/lowess-project/r/articles/boundary.md))
mitigate this at the cost of no longer being a direct reproduction of
Cleveland’s original algorithm. `"noboundary"` is kept as an explicit
option specifically so reference-matching remains possible.

**Residual scaling** (default `scaling_method = "mad"` vs. MAR). MAD
(`median(|r - median(r)|)`) centers residuals at their median before
taking the median absolute value, which is a breakdown-point-optimal
scale estimator; MAR (`median(|r|)`, what R uses) does not center first,
so it can be biased when residuals are systematically skewed. See
[`vignette("scaling")`](https://thisisamirv.github.io/lowess-project/r/articles/scaling.md)
for the full comparison, including `"mean"`.

------------------------------------------------------------------------

## What this package adds

[`stats::lowess()`](https://rdrr.io/r/stats/lowess.html) doesn’t
support:

| Feature | `rfastlowess` | [`stats::lowess()`](https://rdrr.io/r/stats/lowess.html) |
|----|:--:|:--:|
| Kernel functions | 7 options | Tricube only |
| Robustness weighting | 3 options | Bisquare only |
| Scale estimation | MAD, MAR, mean | MAR only |
| Boundary padding | 4 policies | none |
| Confidence / prediction intervals | yes | no |
| Cross-validation for `fraction` | K-fold, LOOCV | no |
| Streaming / online modes | yes | no |
| Custom per-observation weights | yes | no |
| Parallel / GPU execution | yes | no |

See
[`vignette("concepts")`](https://thisisamirv.github.io/lowess-project/r/articles/concepts.md)
for an overview of these, or the
[Benchmarks](https://thisisamirv.github.io/lowess-project/r/articles/benchmarks.md)
article for performance comparisons.
