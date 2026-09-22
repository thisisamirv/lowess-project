# lowess

The Rust bindings provide the core implementation and high-performance extensions. The API uses a Builder pattern consistent across both the `lowess` (pure Rust) and `fastLowess` (accelerated) crates.

> **StreamingLowess** and **OnlineLowess** are documented separately: [Streaming Adapter](crate::doc::api::streaming), [Online Adapter](crate::doc::api::online)

## When to Use Batch Adapter

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

## Classes

The `lowess` crate exposes a `Lowess` wrapper struct that mirrors the class available in other language bindings. It wraps a `LowessBuilder<f64>`, and its `build()` method delegates to the adapter. The `lowess` crate has no `parallel` or `gpu` feature — for CPU-parallel and GPU-accelerated execution, use the `fastLowess` crate instead.

### `Lowess`

Standard in-memory smoothing (batch).

**Constructor:**

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let builder = Lowess::<f64>::new(); // Batch is default

    Ok(())
}
```

#### `fit(x, y)`

Fits the model to the provided `x` and `y` arrays. Returns `Result<LowessResult<T>, LowessError>`.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new().fraction(0.5f64).build()?;
    let result = model.fit(&x, &y)?;
    println!("Fraction used: {}", result.fraction_used);
    println!("Iterations used: {:?}", result.iterations_used);

    Ok(())
}
```

```output
Fraction used: 0.5
Iterations used: Some(3)
```

## Options Structures

These chained methods configure the builder. They correspond to the "Options Structures" in other bindings.

### Lowess Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `fraction(T)` | `T: Float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations(usize)` | `usize` | `3` | Number of robustifying iterations |
| `delta(T)` | `T: Float` | `NaN` | Interpolation distance (`NaN` auto-sets it to 1% of the x-range) |
| `weight_function(...)` | `weight_function` | `"tricube"` | Weight function |
| `robustness_method(...)` | `robustness_method` | `"bisquare"` | Robustness method |
| `scaling_method(...)` | `scaling_method` | `"mad"` | Residual scaling method |
| `boundary_policy(...)` | `boundary_policy` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback(...)` | `zero_weight_fallback` | `"use_local_mean"` | Zero-weight handling |
| `missing(...)` | `missing` | `"error"` | Policy for non-finite (NaN/Inf) values in input data |
| `auto_converge(T)` | `T: Float` | `NaN` | Auto-convergence tolerance |
| `confidence_intervals(T)` | `T: Float` | `NaN` | Confidence level (e.g., 0.95) |
| `prediction_intervals(T)` | `T: Float` | `NaN` | Prediction level (e.g., 0.95) |
| `outputs([&str])` | `&[&str]` | `[]` | Select optional result components: `"diagnostics"`, `"residuals"`, `"weights"`, `"derivative"`, `"se"`, `"sorted"` |
| `cv(CVOptions)` | `CVOptions` | disabled | Cross-validation config built via `CVBuilder::method("kfold"\|"loocv").k(n).fractions(vec![..]).seed(n)` |
| `custom_weights(Vec<T>)` | `Vec<T: Float>` | `None` | Per-observation weights |
| `retain_model(bool)` | `bool` | `false` | Retain training data, enabling `Predict::call()` on the result |

## Options

### fraction

`fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

### iterations

`iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

### delta

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `NaN` (default) auto-sets it to 1% of the x-range. Set it to `0.0` explicitly to disable interpolation and fit every point exactly.

### weight_function

*See: [Weight Functions](crate::doc::weighting::kernels)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](crate::doc::weighting::robustness)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### scaling_method

*See: [Scaling Methods](crate::doc::weighting::scaling)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### boundary_policy

*See: [Boundary Handling](crate::doc::advanced::boundary)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### missing

Policy for handling non-finite (NaN/Inf) values in `x`/`y` (and, in Batch, `custom_weights`):

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Return an error (`InvalidNumericValue`) if any value is non-finite |
| `"drop"` | Silently remove observations where `x` or `y` is non-finite before fitting |

**Note:** A length mismatch between `x` and `y` always errors, even under `"drop"` — that is a caller error, not a data-quality issue for `missing` to mask. In Online, `"drop"` silently ignores a non-finite point (`add_point` returns `Ok(None)`) instead of adding it to the window.

### auto_converge

*See: [Robustness](crate::doc::weighting::robustness)*

Convergence tolerance for early stopping of robustness iterations. `NaN` (default) disables early stopping.

### confidence_intervals

*See: [Intervals](crate::doc::guide::intervals)*

Confidence level for the confidence interval around the mean response (e.g. `0.95`). `NaN` (default) disables confidence intervals.

### prediction_intervals

*See: [Intervals](crate::doc::guide::intervals)*

Confidence level for the prediction interval for new observations (e.g. `0.95`). `NaN` (default) disables prediction intervals.

### outputs

Select which optional result components to include, via a single `.outputs([...])` call:

```rust
use lowess::prelude::*;

Lowess::<f64>::new().outputs(["diagnostics", "residuals", "weights", "derivative", "se", "sorted"]);
```

| Name | Populates | Notes |
| --- | --- | --- |
| `"diagnostics"` | `result.diagnostics` | AIC/AICc/`effective_df` additionally require `"se"` (or confidence/prediction intervals) |
| `"residuals"` | `result.residuals` | Per-point `y - fitted` |
| `"weights"` | `result.robustness_weights` | Final per-point robustness weights |
| `"derivative"` | `result.derivative` | Per-point local fit slope |
| `"se"` | `result.standard_errors` | Hat-matrix statistics (effective df, leverage) |
| `"sorted"` | (reorders output) | Sort every result field ascending by `x` instead of input order |

Unknown names are collected and reported together by `.build()` as `LowessError::ParseErrors`.

### CV Options

*See: [Cross-Validation](crate::doc::guide::cross_validation)*

Cross-validation is configured with a single `.cv(...)` call whose argument is built with `CVBuilder`:

```rust
use lowess::prelude::*;

Lowess::new().cv(CVBuilder::method("kfold").k(5).fractions(vec![0.3, 0.5, 0.7]).seed(42));
```

- `CVBuilder::method("kfold" | "loocv")` — CV strategy: `"kfold"` (fast) or `"loocv"` (slow, exhaustive).
- `.k(n)` — number of folds for k-fold CV (ignored for `"loocv"`).
- `.fractions(vec![..])` — candidate fractions to evaluate (required; CV is disabled unless this is set).
- `.seed(n)` — seed for reproducible k-fold shuffling (ignored for `"loocv"`).

### custom_weights

*See: [Custom Weights](crate::doc::weighting::custom_weights)*

**Note:** In other language bindings `custom_weights` is a `fit()` argument; in Rust it is a builder step because all configuration lives on the builder and `fit()` consumes `self`.

### retain_model

*See: [Predict](crate::doc::guide::predict)*

Retains the fitted model's training data, enabling `Predict::call(&result, new_x)` to evaluate the fit at out-of-sample query points not in the training set. Off by default (no extra memory/clone cost unless requested).

## Result Structure

### `LowessResult<T>`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Array1<T>` | x values (same order as input) |
| `y` | `Array1<T>` | Smoothed y values |
| `fraction_used` | `T` | Fraction used (set or selected by CV) |
| `iterations_used` | `Option<usize>` | Robustness iterations actually performed |
| `standard_errors` | `Option<Array1<T>>` | Per-point standard errors |
| `confidence_lower` | `Option<Array1<T>>` | Lower confidence bounds |
| `confidence_upper` | `Option<Array1<T>>` | Upper confidence bounds |
| `prediction_lower` | `Option<Array1<T>>` | Lower prediction bounds |
| `prediction_upper` | `Option<Array1<T>>` | Upper prediction bounds |
| `residuals` | `Option<Array1<T>>` | Residuals (if `"residuals"` was requested) |
| `robustness_weights` | `Option<Array1<T>>` | Robustness weights (if `"weights"` was requested) |
| `derivative` | `Option<Array1<T>>` | Per-point local fit derivative/slope (if `"derivative"` was requested) |
| `cv_scores` | `Option<Array1<T>>` | CV score per tested fraction |
| `diagnostics` | `Option<Diagnostics<T>>` | Fit metrics (if `"diagnostics"` was requested) |

### `Diagnostics<T>`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `T` | Root Mean Squared Error |
| `mae` | `T` | Mean Absolute Error |
| `r_squared` | `T` | R-squared |
| `residual_sd` | `T` | Residual standard deviation |
| `effective_df` | `Option<T>` | Effective degrees of freedom (`None` if not computed) |
| `aic` | `Option<T>` | AIC (`None` if not computed) |
| `aicc` | `Option<T>` | AICc (`None` if not computed) |

## Predict

*See: [Predict](crate::doc::guide::predict)*

### `Predict::call(&result, new_x) -> PredictOutput<T>`

Evaluates the fitted model at out-of-sample query points. Requires `.retain_model(true)` on the builder before `fit()`, otherwise returns `LowessError::PredictionUnavailable`.

## Example

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.1_f64, 4.0, 6.2, 8.0, 10.1];

    // Configure model
    let model = Lowess::new()
        .fraction(0.5)
        .iterations(3)
        .build()?;

    // Fit data
    let result = model.fit(&x, &y)?;

    println!("Smoothed Y: {:?}", result.y);
    Ok(())
}
```

```output
Smoothed Y: [2.1, 4.0, 6.2, 8.0, 10.1]
```
