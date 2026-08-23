# Boundary Handling

## Overview

![Boundary policy
comparison](../reference/figures/boundary_comparison.svg)

Boundary policy comparison

Standard LOWESS neighbourhoods become asymmetric at the boundaries:
fewer points exist on one side, pulling the local fit toward the data
interior. The `boundary_policy` parameter controls how the data is
padded to mitigate this effect.

| Policy | Padding Strategy | Best For |
|----|----|----|
| `"extend"` | Repeat first / last value | Most datasets (default) |
| `"reflect"` | Mirror data at boundaries | Periodic or symmetric data |
| `"zero"` | Pad with zeros | Data known to approach zero |
| `"noboundary"` | No padding (Cleveland original) | Reproducing reference behaviour |

------------------------------------------------------------------------

## Extend (Default)

Pads beyond both endpoints by replicating the first and last observed
values. Prevents the fit from curling toward zero.

**Use when**: No strong prior on boundary behaviour; general-purpose
smoothing.

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(boundary_policy = "extend")
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## Reflect

Mirrors data at both endpoints. Prevents inflection artifacts when the
signal is periodic or symmetric.

**Use when**: Signal is periodic (e.g. time-of-day) or the boundary is a
known symmetry point.

``` r

model <- Lowess(boundary_policy = "reflect")
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## Zero

Pads with zeros beyond the endpoints.

**Use when**: The signal is known to decay to zero at the boundaries
(e.g. impulse responses, genomic signals at chromosome ends).

``` r

model <- Lowess(boundary_policy = "zero")
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## No Boundary Padding

Reproduces Cleveland’s original behaviour — no padding applied.

**Use when**: Comparing against reference implementations; the original
Cleveland algorithm behaviour is required.

``` r

model <- Lowess(boundary_policy = "noboundary")
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## Comparing Policies

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

policies <- c("extend", "reflect", "zero", "noboundary")
colors   <- c("blue", "red", "green", "purple")

plot(x, y, pch = 16, col = "gray",
     main = "Boundary Policy Comparison")

for (i in seq_along(policies)) {
    model  <- Lowess(boundary_policy = policies[i])
    result <- fit(model, x, y)
    lines(result$x, result$y, col = colors[i], lwd = 2)
}

legend("topright", policies, col = colors, lwd = 2)
```
