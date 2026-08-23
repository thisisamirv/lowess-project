# Confidence and Prediction Intervals

## Overview

![Confidence and prediction
intervals](../reference/figures/intervals_comparison.svg)

Confidence and prediction intervals

> **Note:** Confidence and prediction intervals are available in
> **Batch** mode only. Streaming and Online modes do not support
> intervals.

| Type           | Represents                 | Width  | Use                       |
|----------------|----------------------------|--------|---------------------------|
| **Confidence** | Uncertainty in mean curve  | Narrow | Where is the true trend?  |
| **Prediction** | Uncertainty for new points | Wide   | Where will new data fall? |

------------------------------------------------------------------------

## Confidence Intervals

Estimate uncertainty in the smoothed curve itself.

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(fraction = 0.5, confidence_intervals = 0.95)
result <- fit(model, x, y)

# Plot with bands
plot(x, y, pch = 16, col = "gray",
     main = "LOWESS with 95% Confidence Intervals")
lines(result$x, result$y, col = "blue", lwd = 2)
lines(result$x, result$confidence_lower, col = "blue", lty = 2)
lines(result$x, result$confidence_upper, col = "blue", lty = 2)
```

------------------------------------------------------------------------

## Prediction Intervals

Wider than confidence intervals — cover where new observations will
fall.

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(fraction = 0.5, prediction_intervals = 0.95)
result <- fit(model, x, y)

plot(x, y, pch = 16, col = "gray",
     main = "LOWESS with 95% Prediction Intervals")
lines(result$x, result$y, col = "blue", lwd = 2)
lines(result$x, result$prediction_lower, col = "red", lty = 2)
lines(result$x, result$prediction_upper, col = "red", lty = 2)
```

------------------------------------------------------------------------

## Both Intervals Together

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(
    fraction = 0.5,
    confidence_intervals = 0.95,
    prediction_intervals = 0.95
)
result <- fit(model, x, y)

plot(x, y, pch = 16, col = "gray",
     main = "Confidence vs Prediction Intervals")
lines(result$x, result$y, col = "blue", lwd = 2)

# Confidence interval (narrow, blue)
lines(result$x, result$confidence_lower, col = "blue", lty = 2)
lines(result$x, result$confidence_upper, col = "blue", lty = 2)

# Prediction interval (wide, red)
lines(result$x, result$prediction_lower, col = "red", lty = 3)
lines(result$x, result$prediction_upper, col = "red", lty = 3)

legend("topright",
       c("Data", "Smoothed", "95% CI", "95% PI"),
       pch = c(16, NA, NA, NA), lty = c(NA, 1, 2, 3),
       col = c("gray", "blue", "blue", "red"))
```

------------------------------------------------------------------------

## Choosing a Coverage Level

Pass any value between 0 and 1 (exclusive). Common choices:

| Level | Interpretation                                    |
|-------|---------------------------------------------------|
| 0.90  | 90% of the true curve / future points fall inside |
| 0.95  | Standard choice (95%)                             |
| 0.99  | Conservative (wider bands)                        |

``` r

# 99% confidence intervals
model <- Lowess(fraction = 0.5, confidence_intervals = 0.99)
result <- fit(model, x, y)
```
