<!-- markdownlint-disable MD024 MD046 MD033 MD037 -->
# Time Series Analysis

LOWESS for trend extraction and temporal smoothing.

## Overview

Time series data often contains noise, seasonality, and trends. LOWESS provides flexible trend extraction without parametric assumptions.

---

## Basic Trend Extraction

`fraction = 0.1` sizes the neighbourhood as 10% of the data at each evaluation point — narrow enough to follow a slowly varying trend without smearing periodic variation. Three robustness `iterations` down-weight noise spikes so they cannot bias the fitted curve; this is especially important when the signal-to-noise ratio is low or when occasional outliers are expected.

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let n = 500usize;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * 100.0 / (n - 1) as f64).collect();
    let y: Vec<f64> = t.iter().enumerate()
        .map(|(i, &ti)| 10.0 + 0.5 * ti + 3.0 * (ti / 10.0).sin()
                      + ((i * 7 + 3) as f64 % 1.7 - 0.85) * 3.0)
        .collect();

    let model = Lowess::new()
        .fraction(0.1)
        .iterations(3)
        .build()?;

    let result = model.fit(&t, &y)?;
    // result.y contains the trend

    println!("First smoothed value (fraction=0.1): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (fraction=0.1): 11.321590922416165
```

---

## Detrending

Remove trend to analyze residual patterns.

Setting `return_residuals = True` stores `observed − smoothed` alongside the smooth. A slightly wider `fraction = 0.3` produces a smoother baseline trend, so short-duration oscillations end up in the residuals rather than being absorbed into the trend component. The residual series is then ready for spectral analysis, seasonality detection, or change-point methods.

```rust
use fastLowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = t.iter().map(|&ti| ti.sin() + 0.1).collect();

    let model = Lowess::new()
        .fraction(0.3)
        .iterations(3)
        .return_residuals()
        .build()?;

    let result = model.fit(&t, &y)?;
    let trend = &result.y;
    let detrended = result.residuals.as_ref().unwrap();

    if let Some(r) = &result.residuals {
        println!("First residual: {}", r[0]);
    }
    Ok(())
}
```

```output
First residual: -0.15824361645514948
```

---

## Forecasting with Prediction Intervals

Prediction intervals widen the uncertainty band to include both the uncertainty in the fitted curve (confidence interval) and the expected scatter of new observations around it. `fraction = 0.2` offers a balance between local detail and stable interval width — too small a fraction produces jagged interval edges; too large a fraction underestimates local variance near turning points.

```rust
use fastLowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = t.iter().map(|&ti| ti.sin() + 0.1).collect();

    let model = Lowess::new()
        .fraction(0.2)
        .iterations(3)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .build()?;

    let result = model.fit(&t, &y)?;
    // Access result.prediction_lower and result.prediction_upper

    if let (Some(lo), Some(hi)) = (&result.prediction_lower, &result.prediction_upper) {
        println!("95% PI: [{}, {}]", lo[0], hi[0]);
    }
    Ok(())
}
```

```output
95% PI: [0.15801046224996285, 0.2908827214492591]
```

---

## Handling Missing Data

LOWESS naturally handles irregular time sampling:

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let t_irregular: Vec<f64> = (0..100).map(|i| i as f64 * 1.0 + (i * 31 % 10) as f64 * 0.1).collect();
    let y_irregular: Vec<f64> = t_irregular.iter().map(|&t| 10.0 + t * 0.3 + 2.0 * (t * 0.1).sin()).collect();

    // Irregular sampling - no special handling needed
    let model = Lowess::new()
        .fraction(0.2)
        .build()?;

    let result = model.fit(&t_irregular, &y_irregular)?;

    println!("First smoothed value (fraction=0.2): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (fraction=0.2): 11.327309510260003
```

---

## Multi-Scale Analysis

Use different fractions to extract features at different scales:

```rust
use fastLowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = t.iter().map(|&ti| ti.sin() + 0.1).collect();

    let fractions = [0.05, 0.2, 0.5];

    for f in fractions {
        let model = Lowess::new()
            .fraction(f)
            .build()?;
        let result = model.fit(&t, &y)?;
        println!("First smoothed value (fraction={f}): {}", result.y[0]);
    }
    Ok(())
}
```

```output
First smoothed value (fraction=0.05): 0.13171195982828227
First smoothed value (fraction=0.2): 0.22444659184961097
First smoothed value (fraction=0.5): 0.33437036041791557
```

---

## Gene Expression Time Course

Biological application:

```rust
use fastLowess::prelude::*;
use std::f64::consts::PI;

fn main() -> Result<(), LowessError> {

    let hours: Vec<f64> = (0..49).map(|i| i as f64 * 0.5).collect(); // 0.0..24.0 step 0.5
    let expression: Vec<f64> = hours.iter().enumerate()
        .map(|(i, &h)| 100.0 * (1.0 + 0.5 * (h * PI / 12.0).sin())
                      + ((i * 7 + 3) as f64 % 1.7 - 0.85) * 10.0)
        .collect();

    let model = Lowess::new()
        .fraction(0.3)
        .iterations(3)
        .confidence_intervals(0.95)
        .return_diagnostics()
        .build()?;

    let result = model.fit(&hours, &expression)?;
    if let Some(diag) = &result.diagnostics {
        println!("R2: {:.3}", diag.r_squared);
    }

    Ok(())
}
```

```output
R2: 0.973
```

---

## Choosing Fraction for Time Series

| Data Type | Recommended Fraction | Rationale |
| --- | --- | --- |
| Daily data (years) | 0.3–0.5 | Capture annual trends |
| Hourly data (days) | 0.1–0.2 | Capture daily patterns |
| Sensor data (minutes) | 0.05–0.1 | Preserve short-term features |
| Noisy data | Higher | Reduce noise impact |
| Clean data | Lower | Preserve detail |

---

## See Also

- [Real-Time Processing](crate::doc::use_cases::real_time) — For streaming time series
- [Cross-Validation](crate::doc::cross_validation) — Optimal fraction selection
- [Boundary Handling](crate::doc::boundary) — Edge bias in trend extraction
- [API Reference](crate::doc::api) — Full parameter reference
