# R API

API reference for the `rfastlowess` R package.

## Installation

```r
# From R-universe (recommended)
install.packages("rfastlowess", repos = "https://thisisamirv.r-universe.dev")

# From source (requires Rust)
devtools::install_github("thisisamirv/lowess-project", subdir = "bindings/r")
```

---

## Functions

### fastlowess

Main function for batch smoothing.

```r
fastlowess(
  x,
  y,
  fraction = 0.67,
  iterations = 3,
  delta = NULL,
  parallel = TRUE,
  weight_function = "tricube",
  robustness_method = "bisquare",
  scaling_method = "mad",
  zero_weight_fallback = "use_local_mean",
  boundary_policy = "extend",
  auto_converge = NULL,
  return_residuals = FALSE,
  return_diagnostics = FALSE,
  return_robustness_weights = FALSE,
  confidence_intervals = NULL,
  prediction_intervals = NULL,
  cv_method = NULL,
  cv_k = 5,
  cv_fractions = NULL,
  cv_seed = NULL
)
```

**Arguments:**

| Argument     | Type    | Default  | Description                            |
|--------------|---------|----------|----------------------------------------|
| `x`          | numeric | required | Independent variable                   |
| `y`          | numeric | required | Dependent variable                     |
| `fraction`   | numeric | 0.67     | Smoothing span (0, 1]                  |
| `iterations` | integer | 3        | Robustness iterations                  |
| `delta`      | numeric | NULL     | Interpolation threshold (auto if NULL) |
| `parallel`   | logical | TRUE     | Enable parallel execution              |

**Value:** A list with components:

| Component            | Type    | Description                           |
|----------------------|---------|---------------------------------------|
| `x`                  | numeric | Input x values                        |
| `y`                  | numeric | Smoothed y values                     |
| `fraction_used`      | numeric | Actual fraction used                  |
| `residuals`          | numeric | If `return_residuals = TRUE`          |
| `confidence_lower`   | numeric | If `confidence_intervals` set         |
| `confidence_upper`   | numeric | If `confidence_intervals` set         |
| `prediction_lower`   | numeric | If `prediction_intervals` set         |
| `prediction_upper`   | numeric | If `prediction_intervals` set         |
| `robustness_weights` | numeric | If `return_robustness_weights = TRUE` |
| `diagnostics`        | list    | If `return_diagnostics = TRUE`        |
| `cv_scores`          | numeric | If cross-validation used              |

**Example:**

```r
library(rfastlowess)

x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.2)

result <- fastlowess(x, y, fraction = 0.3, iterations = 3)
plot(x, y, pch = 16, col = "gray")
lines(result$x, result$y, col = "red", lwd = 2)
```

---

### fastlowess_streaming

Streaming mode for large datasets.

```r
fastlowess_streaming(
  x,
  y,
  fraction = 0.67,
  iterations = 3,
  chunk_size = 5000,
  overlap = 500,
  merge_strategy = "average",
  parallel = TRUE,
  ...
)
```

**Additional Arguments:**

| Argument         | Type      | Default   | Description            |
|------------------|-----------|-----------|------------------------|
| `chunk_size`     | integer   | 5000      | Points per chunk       |
| `overlap`        | integer   | 500       | Overlap between chunks |
| `merge_strategy` | character | "average" | Merge method           |

**Example:**

```r
# Large dataset
x <- seq(0, 100, length.out = 100000)
y <- sin(x / 10) + rnorm(100000, sd = 0.1)

result <- fastlowess_streaming(x, y, chunk_size = 10000)
```

---

### fastlowess_online

Online mode for real-time data.

```r
fastlowess_online(
  x,
  y,
  fraction = 0.2,
  window_capacity = 100,
  min_points = 2,
  iterations = 3,
  update_mode = "incremental",
  ...
)
```

**Additional Arguments:**

| Argument          | Type      | Default       | Description          |
|-------------------|-----------|---------------|----------------------|
| `window_capacity` | integer   | 100           | Max points in window |
| `min_points`      | integer   | 2             | Points before output |
| `update_mode`     | character | "incremental" | Update strategy      |

**Example:**

```r
# Sensor simulation
sensor_times <- 1:100
sensor_values <- 20 + 5 * sin(sensor_times / 10) + rnorm(100)

result <- fastlowess_online(sensor_times, sensor_values, window_capacity = 25)
```

---

## String Options

### weight_function

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"biweight"`
- `"cosine"`
- `"triangle"`
- `"uniform"`

### robustness_method

- `"bisquare"` (default)
- `"huber"`
- `"talwar"`

### boundary_policy

- `"extend"` (default)
- `"reflect"`
- `"zero"`
- `"no_boundary"`

### merge_strategy

- `"average"` (default)
- `"left"`
- `"right"`
- `"weighted"`

### cv_method

- `"kfold"` — K-fold cross-validation
- `"loocv"` — Leave-one-out cross-validation

---

## Diagnostics

When `return_diagnostics = TRUE`:

```r
result$diagnostics
# $rmse        - Root Mean Square Error
# $mae         - Mean Absolute Error
# $r_squared   - R² coefficient
# $residual_sd - Residual standard deviation
```

---

## Comparison with stats::lowess

| Feature              | rfastlowess | stats::lowess |
|----------------------|:-----------:|:-------------:|
| Parallel execution   | ✓           | ✗             |
| Confidence intervals | ✓           | ✗             |
| Prediction intervals | ✓           | ✗             |
| Cross-validation     | ✓           | ✗             |
| Streaming mode       | ✓           | ✗             |
| Online mode          | ✓           | ✗             |
| Kernel options       | 7           | 1             |
| Robustness methods   | 3           | 1             |

---

## See Also

- `stats::lowess` — Base R implementation
- `stats::loess` — Local polynomial regression
