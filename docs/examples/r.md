# R Examples

Complete code examples for the `rfastlowess` R package.

## Basic Smoothing

```r
library(rfastlowess)

# Generate noisy data
set.seed(42)
x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

# Smooth
result <- fastlowess(x, y, fraction = 0.3, iterations = 3)

# Plot
plot(x, y, pch = 16, col = "gray", main = "Basic LOWESS")
lines(result$x, result$y, col = "red", lwd = 2)
```

---

## With Confidence Intervals

```r
library(rfastlowess)

set.seed(42)
x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

result <- fastlowess(
    x, y,
    fraction = 0.3,
    iterations = 3,
    confidence_intervals = 0.95,
    prediction_intervals = 0.95,
    return_diagnostics = TRUE
)

# Plot with bands
plot(x, y, pch = 16, col = "gray",
     main = sprintf("LOWESS (R² = %.4f)", result$diagnostics$r_squared))

# Prediction interval (wider)
polygon(
    c(x, rev(x)),
    c(result$prediction_lower, rev(result$prediction_upper)),
    col = rgb(0, 0, 1, 0.1), border = NA
)

# Confidence interval (narrower)
polygon(
    c(x, rev(x)),
    c(result$confidence_lower, rev(result$confidence_upper)),
    col = rgb(0, 0, 1, 0.3), border = NA
)

lines(result$x, result$y, col = "blue", lwd = 2)
legend("topright", 
       legend = c("Data", "LOWESS", "95% CI", "95% PI"),
       pch = c(16, NA, 15, 15),
       lty = c(NA, 1, NA, NA),
       col = c("gray", "blue", rgb(0, 0, 1, 0.3), rgb(0, 0, 1, 0.1)),
       lwd = c(NA, 2, NA, NA))
```

---

## Cross-Validation

```r
library(rfastlowess)

set.seed(42)
x <- seq(0, 20, length.out = 200)
y <- 0.5 * x + sin(x) + rnorm(200)

result <- fastlowess(
    x, y,
    cv_method = "kfold",
    cv_k = 5,
    cv_fractions = c(0.1, 0.2, 0.3, 0.5, 0.7),
    cv_seed = 42
)

cat("Best fraction:", result$fraction_used, "\n")
cat("CV scores:", result$cv_scores, "\n")
```

---

## Outlier Detection

```r
library(rfastlowess)

x <- 0:49
y <- x * 2
y[11] <- 100  # Outlier (R is 1-indexed)
y[26] <- -50  # Outlier
y[41] <- 150  # Outlier

result <- fastlowess(
    x, y,
    fraction = 0.3,
    iterations = 5,
    robustness_method = "bisquare",
    return_robustness_weights = TRUE
)

# Find outliers
outliers <- which(result$robustness_weights < 0.5)
cat("Outliers at indices:", outliers, "\n")

# Plot
plot(x, y, pch = 16, col = ifelse(result$robustness_weights < 0.5, "red", "gray"),
     main = "Outlier Detection")
lines(result$x, result$y, col = "blue", lwd = 2)
legend("topleft", legend = c("Normal", "Outlier"), pch = 16, col = c("gray", "red"))
```

---

## Streaming Large Data

```r
library(rfastlowess)

# Large dataset
n <- 100000
x <- seq(0, 100, length.out = n)
y <- sin(x / 10) + rnorm(n, sd = 0.1)

result <- fastlowess_streaming(
    x, y,
    fraction = 0.01,
    chunk_size = 10000,
    overlap = 1000,
    merge_strategy = "weighted"
)

cat("Processed", length(result$y), "points\n")
```

---

## Online (Real-Time) Processing

```r
library(rfastlowess)

# Sensor simulation
times <- 1:100
values <- 20 + 5 * sin(times / 10) + rnorm(100)

result <- fastlowess_online(
    times, values,
    fraction = 0.3,
    window_capacity = 25,
    min_points = 5,
    update_mode = "incremental"
)

plot(times, values, pch = 16, col = "gray",
     main = "Online Smoothing")
lines(result$x, result$y, col = "green", lwd = 2)
```

---

## Compare with stats::lowess

```r
library(rfastlowess)

set.seed(42)
x <- 1:100
y <- sin(x / 10) + rnorm(100, sd = 0.3)

# rfastlowess
result_fast <- fastlowess(x, y, fraction = 0.3, iterations = 3)

# Base R
result_base <- lowess(x, y, f = 0.3, iter = 3)

# Compare
plot(x, y, pch = 16, col = "gray", main = "rfastlowess vs stats::lowess")
lines(result_fast$x, result_fast$y, col = "red", lwd = 2)
lines(result_base$x, result_base$y, col = "blue", lwd = 2, lty = 2)
legend("topright", 
       legend = c("rfastlowess", "stats::lowess"),
       col = c("red", "blue"),
       lty = c(1, 2),
       lwd = 2)
```

---

## Multiple Fractions

```r
library(rfastlowess)

set.seed(42)
x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

fractions <- c(0.1, 0.3, 0.5, 0.7)
colors <- c("red", "blue", "green", "purple")

plot(x, y, pch = 16, col = "gray", main = "Effect of Fraction")

for (i in seq_along(fractions)) {
    result <- fastlowess(x, y, fraction = fractions[i])
    lines(result$x, result$y, col = colors[i], lwd = 2)
}

legend("topright", 
       legend = paste("f =", fractions),
       col = colors,
       lwd = 2)
```

---

## Kernel Comparison

```r
library(rfastlowess)

set.seed(42)
x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.2)

kernels <- c("tricube", "epanechnikov", "gaussian", "uniform")
colors <- c("red", "blue", "green", "purple")

plot(x, y, pch = 16, col = "gray", main = "Kernel Comparison")

for (i in seq_along(kernels)) {
    result <- fastlowess(x, y, fraction = 0.3, weight_function = kernels[i])
    lines(result$x, result$y, col = colors[i], lwd = 2)
}

legend("topright", legend = kernels, col = colors, lwd = 2)
```

---

## With ggplot2

```r
library(rfastlowess)
library(ggplot2)

set.seed(42)
df <- data.frame(
    x = seq(0, 10, length.out = 100),
    y = sin(seq(0, 10, length.out = 100)) + rnorm(100, sd = 0.3)
)

result <- fastlowess(df$x, df$y, fraction = 0.3, confidence_intervals = 0.95)

df$smoothed <- result$y
df$ci_lower <- result$confidence_lower
df$ci_upper <- result$confidence_upper

ggplot(df, aes(x = x)) +
    geom_point(aes(y = y), alpha = 0.5) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.2, fill = "blue") +
    geom_line(aes(y = smoothed), color = "blue", size = 1) +
    labs(title = "LOWESS with ggplot2", x = "X", y = "Y") +
    theme_minimal()
```
