# Rust API

API reference for the `lowess` and `fastLowess` Rust crates.

---

## Functions

### Batch Smoothing

Main function for batch smoothing using the builder pattern.

```rust
use fastLowess::prelude::*;

let model = Lowess::new()
    .fraction(0.5)
    .iterations(3)
    .delta(0.01)
    .parallel(true)
    .weight_function(WeightFunction::Tricube)
    .robustness_method(RobustnessMethod::Bisquare)
    .scaling_method(ScalingMethod::MAD)
    .zero_weight_fallback(ZeroWeightFallback::UseLocalMean)
    .boundary_policy(BoundaryPolicy::Extend)
    .auto_converge(1e-4)
    .return_residuals()
    .return_diagnostics()
    .return_robustness_weights()
    .confidence_intervals(0.95)
    .prediction_intervals(0.95)
    .cross_validate(KFold::new(5, &[0.3, 0.5, 0.7]).seed(123))
    .adapter(Batch)
    .backend(CPU)  // fastLowess only
    .build()?;

let result = model.fit(&x, &y)?;
```

**Builder Methods:**

| Method                       | Type                  | Default            | Description                       |
|------------------------------|-----------------------|--------------------|-----------------------------------|
| `fraction(f64)`              | `f64`                 | `0.67`             | Smoothing span (0, 1]             |
| `iterations(usize)`          | `usize`               | `3`                | Robustness iterations             |
| `delta(f64)`                 | `f64`                 | `0.0`              | Interpolation threshold (0=auto)  |
| `parallel(bool)`             | `bool`                | `false`            | Enable parallelism                |
| `weight_function()`          | `WeightFunction`      | `Tricube`          | Kernel function                   |
| `robustness_method()`        | `RobustnessMethod`    | `Bisquare`         | Outlier handling method           |
| `scaling_method()`           | `ScalingMethod`       | `MAD`              | Scale estimation method           |
| `zero_weight_fallback()`     | `ZeroWeightFallback`  | `UseLocalMean`     | Zero weight handling              |
| `boundary_policy()`          | `BoundaryPolicy`      | `Extend`           | Boundary handling                 |
| `auto_converge(f64)`         | `f64`                 | disabled           | Auto-convergence tolerance        |
| `return_residuals()`         | -                     | `false`            | Return residuals                  |
| `return_diagnostics()`       | -                     | `false`            | Return fit diagnostics            |
| `return_robustness_weights()`| -                     | `false`            | Return robustness weights         |
| `confidence_intervals(f64)`  | `f64`                 | disabled           | Confidence level (e.g., 0.95)     |
| `prediction_intervals(f64)`  | `f64`                 | disabled           | Prediction interval level         |
| `cross_validate(CV)`         | `impl CrossValidation`| disabled           | Cross-validation strategy         |

**Returns:** `LowessResult<T>` with fields:

| Field                  | Type                     | Description                         |
|------------------------|--------------------------|-------------------------------------|
| `x`                    | `Array1<T>`              | Input x values                      |
| `y`                    | `Array1<T>`              | Smoothed y values                   |
| `fraction_used`        | `T`                      | Actual fraction used                |
| `residuals`            | `Option<Array1<T>>`      | If `return_residuals()` called      |
| `confidence_lower`     | `Option<Array1<T>>`      | If `confidence_intervals()` set     |
| `confidence_upper`     | `Option<Array1<T>>`      | If `confidence_intervals()` set     |
| `prediction_lower`     | `Option<Array1<T>>`      | If `prediction_intervals()` set     |
| `prediction_upper`     | `Option<Array1<T>>`      | If `prediction_intervals()` set     |
| `robustness_weights`   | `Option<Array1<T>>`      | If `return_robustness_weights()`    |
| `diagnostics`          | `Option<Diagnostics<T>>` | If `return_diagnostics()`           |
| `cv_results`           | `Option<CVResults<T>>`   | If cross-validation used            |

**Example:**

```rust
use fastLowess::prelude::*;
use ndarray::array;

let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
let y = array![2.1, 3.9, 6.2, 8.0, 10.1];

let model = Lowess::new()
    .fraction(0.3)
    .iterations(3)
    .adapter(Batch)
    .build()?;

let result = model.fit(&x.view(), &y.view())?;
println!("Smoothed: {:?}", result.y);
```

---

### Streaming Mode

Streaming mode for large datasets with constant memory usage.

```rust
use fastLowess::prelude::*;

let mut processor = Lowess::new()
    .fraction(0.1)
    .adapter(Streaming)
    .chunk_size(5000)
    .overlap(500)
    .merge_strategy(MergeStrategy::Average)
    .build()?;

let result = processor.process_chunk(&x, &y)?;
let final_result = processor.finalize()?;
```

**Additional Methods:**

| Method                  | Type            | Default     | Description                        |
|-------------------------|-----------------|-------------|------------------------------------|
| `chunk_size(usize)`     | `usize`         | `5000`      | Points per chunk                   |
| `overlap(usize)`        | `usize`         | `500`       | Overlap between chunks             |
| `merge_strategy()`      | `MergeStrategy` | `Average`   | How to merge overlaps              |

**Example:**

```rust
// Process 1 million points
let x = Array1::linspace(0.0, 1000.0, 1_000_000);
let y = x.mapv(|v| (v / 100.0).sin()) + Array1::random(1_000_000, Normal::new(0.0, 0.1)?);

let mut processor = Lowess::new()
    .fraction(0.05)
    .adapter(Streaming)
    .chunk_size(10000)
    .build()?;

let result = processor.process_chunk(&x.view(), &y.view())?;
```

---

### Online Mode

Online mode for real-time data with sliding window.

```rust
use fastLowess::prelude::*;

let mut processor = Lowess::new()
    .fraction(0.2)
    .adapter(Online)
    .window_capacity(100)
    .min_points(5)
    .update_mode(UpdateMode::Incremental)
    .build()?;

if let Some(output) = processor.add_point(x, y)? {
    println!("Smoothed: {}", output.smoothed);
}
```

**Additional Methods:**

| Method                  | Type         | Default       | Description          |
|-------------------------|--------------|---------------|----------------------|
| `window_capacity(usize)`| `usize`      | `100`         | Max points in window |
| `min_points(usize)`     | `usize`      | `2`           | Points before output |
| `update_mode()`         | `UpdateMode` | `Incremental` | Update strategy      |

**Example:**

```rust
// Sensor data simulation
let mut processor = Lowess::new()
    .fraction(0.3)
    .adapter(Online)
    .window_capacity(25)
    .build()?;

for (time, value) in sensor_data {
    if let Some(output) = processor.add_point(time, value)? {
        println!("Time: {}, Smoothed: {}", time, output.smoothed);
    }
}
```

---

## Enum Options

### WeightFunction

- `Tricube` (default)
- `Epanechnikov`
- `Gaussian`
- `Biweight`
- `Cosine`
- `Triangle`
- `Uniform`

### RobustnessMethod

- `Bisquare` (default)
- `Huber`
- `Talwar`

### BoundaryPolicy

- `Extend` (default)
- `Reflect`
- `Zero`
- `NoBoundary`

### MergeStrategy

- `Average` (default)
- `Left`
- `Right`
- `Weighted`

### UpdateMode

- `Incremental` (default)
- `Full`

---

## Diagnostics

When `return_diagnostics()` is called, the result includes:

```rust
pub struct Diagnostics<T> {
    pub rmse: T,        // Root Mean Square Error
    pub mae: T,         // Mean Absolute Error
    pub r_squared: T,   // R² coefficient
    pub residual_sd: T, // Residual standard deviation
    pub effective_df: T // Effective degrees of freedom
}

// Access via:
if let Some(diag) = result.diagnostics {
    println!("R² = {}", diag.r_squared);
}
```

---
