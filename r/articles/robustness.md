# Robustness

## How Robustness Works

Standard LOWESS can be biased by outliers. Robustness iterations
downweight points with large residuals:

1.  Fit initial LOWESS
2.  Compute residuals
3.  Assign robustness weights (large residuals → low weight)
4.  Refit using combined distance × robustness weights
5.  Repeat steps 2–4

------------------------------------------------------------------------

## Robustness Methods

![Robustness method
comparison](../reference/figures/robust_method_comparison.svg)

Robustness method comparison

![Effect of robustness
iterations](../reference/figures/robust_iter_comparison.svg)

Effect of robustness iterations

### Bisquare (Default)

Smooth downweighting. Points transition gradually from full weight to
zero.

``` math
w(u) = \begin{cases} (1 - u^2)^2 & |u| < 1 \\ 0 & |u| \geq 1 \end{cases}
```

**Use when**: General purpose, balanced approach.

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(iterations = 3, robustness_method = "bisquare")
result <- fit(model, x, y)
```

### Huber

Less aggressive than bisquare; still downweights outliers but keeps
moderate residuals at reduced weight.

**Use when**: Mild outlier contamination; want to preserve moderate
deviations.

``` r

model <- Lowess(iterations = 3, robustness_method = "huber")
result <- fit(model, x, y)
```

### Talwar

Hard thresholding — points above the threshold are excluded completely.

**Use when**: Known contamination that should be fully excluded.

``` r

model <- Lowess(iterations = 3, robustness_method = "talwar")
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## Choosing the Number of Iterations

| Iterations | Effect                  | When to Use                |
|------------|-------------------------|----------------------------|
| 0          | No robustness (fastest) | Clean data, speed-critical |
| 1–3        | Moderate                | Most applications          |
| 4–6        | Strong                  | Data with clear outliers   |
| 7+         | Very strong             | Heavy contamination        |

``` r

library(rfastlowess)
set.seed(42)
x <- 1:100
y <- sin(x / 10) + rnorm(100, sd = 0.3)

# Inject outliers
y[c(25, 50, 75)] <- y[c(25, 50, 75)] + 5

# Without robustness
model_0 <- Lowess(iterations = 0)
result_0 <- fit(model_0, x, y)

# With robustness
model_3 <- Lowess(iterations = 3)
result_3 <- fit(model_3, x, y)

plot(x, y, pch = 16, col = "gray", main = "Effect of Robustness Iterations")
lines(result_0$x, result_0$y, col = "red", lwd = 2, lty = 2)
lines(result_3$x, result_3$y, col = "blue", lwd = 2)
legend("topright", c("Data", "iterations=0", "iterations=3"),
       pch = c(16, NA, NA), lty = c(NA, 2, 1),
       col = c("gray", "red", "blue"))
```

------------------------------------------------------------------------

## Method Comparison

| Method       | Handling                | Best For              |
|--------------|-------------------------|-----------------------|
| `"bisquare"` | Smooth to zero          | General purpose       |
| `"huber"`    | Linear then downweights | Mild contamination    |
| `"talwar"`   | Hard threshold (0/1)    | Severe point outliers |

------------------------------------------------------------------------

## Detecting Outliers

Use robustness weights to identify potential outliers:

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)
y[c(20, 50, 80)] <- y[c(20, 50, 80)] + 5  # inject outliers

model <- Lowess(iterations = 5, return_robustness_weights = TRUE)
result <- fit(model, x, y)

for (i in seq_along(result$robustness_weights)) {
    if (result$robustness_weights[i] < 0.5)
        cat(sprintf("Point %d is likely an outlier (weight: %.3f)\n",
                    i, result$robustness_weights[i]))
}
```
