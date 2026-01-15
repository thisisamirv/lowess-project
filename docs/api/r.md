# R API

API reference for the `rfastlowess` R package.

---

## Classes

The `rfastlowess` package provides three stateful classes for different smoothing scenarios. Each class is initialized with configuration parameters and provides methods for processing data.

### Lowess

Main class for batch smoothing of complete datasets.

#### Lowess Constructor

```r
Lowess(
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

| Argument     | Type      | Default       | Description                            |
| :----------- | :-------- | :------------ | :------------------------------------- |
| `fraction`   | numeric   | 0.67          | Smoothing span (0, 1]                  |
| `iterations` | integer   | 3             | Robustness iterations                  |
| `delta`      | numeric   | NULL          | Interpolation threshold (auto if NULL) |
| `parallel`   | logical   | TRUE          | Enable parallel execution              |

#### Lowess Methods

**`fit(x, y)`**

Fits the LOWESS model to the provided data.

- **`x`**: Independent variable (numeric vector)
- **`y`**: Dependent variable (numeric vector)

**Value:** A list with components:

| Component            | Type    | Description                           |
| :------------------- | :------ | :------------------------------------ |
| `x`                  | numeric | Input x values (sorted)               |
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

model <- Lowess(fraction = 0.3, iterations = 3)
result <- model$fit(x, y)

plot(x, y, pch = 16, col = "gray")
lines(result$x, result$y, col = "red", lwd = 2)
```

---

### StreamingLowess

Streaming mode for memory-efficient processing of large datasets in chunks.

#### StreamingLowess Constructor

```r
StreamingLowess(
  fraction = 0.67,
  iterations = 3,
  chunk_size = 5000L,
  overlap = 500L,
  merge_strategy = "average",
  parallel = TRUE,
  ...
)
```

**Additional Arguments:**

| Argument         | Type      | Default   | Description            |
| :--------------- | :-------- | :-------- | :--------------------- |
| `chunk_size`     | integer   | 5000      | Points per chunk       |
| `overlap`        | integer   | 500       | Overlap between chunks |
| `merge_strategy` | character | "average" | Merge method           |

#### StreamingLowess Methods

**`process_chunk(x, y)`**

Processes a chunk of data. May return a partial result if enough data has been accumulated to finalize a section.

**`finalize()`**

Finalizes processing and returns any remaining smoothed points.

**Example:**

```r
# Large dataset simulation
x <- seq(0, 100, length.out = 100000)
y <- sin(x / 10) + rnorm(100000, sd = 0.1)

model <- StreamingLowess(fraction = 0.3, chunk_size = 10000L)
result <- model$process_chunk(x, y)
remaining <- model$finalize()

smoothed_y <- c(result$y, remaining$y)
```

---

### OnlineLowess

Online mode for real-time smoothing of streaming data with a sliding window.

#### OnlineLowess Constructor

```r
OnlineLowess(
  fraction = 0.2,
  window_capacity = 100L,
  min_points = 2L,
  iterations = 3,
  update_mode = "incremental",
  ...
)
```

**Additional Arguments:**

| Argument          | Type      | Default       | Description          |
| :---------------- | :-------- | :------------ | :------------------- |
| `window_capacity` | integer   | 100           | Max points in window |
| `min_points`      | integer   | 2             | Points before output |
| `update_mode`     | character | "incremental" | Update strategy      |

#### OnlineLowess Methods

**`add_points(x, y)`**

Adds new points to the online buffer and returns smoothed values for any points that have fallen out of the active update window.

**Example:**

```r
# Sensor simulation
model <- OnlineLowess(window_capacity = 25L, fraction = 0.5)

for (i in 1:100) {
    x_new <- i
    y_new <- 20 + 5 * sin(i / 10) + rnorm(1)
   
    result <- model$add_points(x_new, y_new)
    if (length(result$y) > 0) {
        cat(sprintf("Time %d: Smoothed = %.2f\n", result$x[1], result$y[1]))
    }
}
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

### update_mode

- `"incremental"` (default)
- `"full"`

---

## Diagnostics

When `return_diagnostics = TRUE`, the result includes:

```r
result$diagnostics = list(
    rmse = numeric,        # Root Mean Square Error
    mae = numeric,         # Mean Absolute Error
    r_squared = numeric,   # R² coefficient
    residual_sd = numeric, # Residual standard deviation
    effective_df = numeric # Effective degrees of freedom
)
```
