<!-- markdownlint-disable MD024 MD033 -->
# Intervals

Confidence and prediction intervals for uncertainty quantification.

## Overview

![Confidence and Prediction Intervals](https://raw.githubusercontent.com/thisisamirv/lowess-project/main/crates/fastLowess/assets/diagrams/intervals_comparison.svg)

!!! note "Adapter support"
    Analytic and residual-bootstrap intervals are available in **Batch**, **Streaming** (per chunk with overlap merging), and **Online** with `update_mode("full")`. Online's default `"incremental"` mode rejects intervals at `.build()`.

| Type | Represents | Width | Use |
| --- | --- | --- | --- |
| **Confidence** | Uncertainty in mean curve | Narrow | Where is the true trend? |
| **Prediction** | Uncertainty for new points | Wide | Where will new data fall? |

---

## Confidence Intervals

Estimate uncertainty in the smoothed curve itself.

```rust
use fastLowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();


    let model = Lowess::new()
        .fraction(0.5)
        .confidence_intervals(0.95)  // 95% CI
        .build()?;

    let result = model.fit(&x, &y)?;

    // Access intervals
    if let (Some(lower), Some(upper)) = (&result.confidence_lower, &result.confidence_upper) {
        for i in 0..3 {
            println!("x={:.2}: y={:.2} [{:.2}, {:.2}]",
                result.x[i], result.y[i], lower[i], upper[i]);
        }
    }

    Ok(())
}
```

```output
x=0.00: y=0.33 [0.29, 0.37]
x=0.06: y=0.36 [0.32, 0.40]
x=0.13: y=0.39 [0.35, 0.43]
```

---

## Prediction Intervals

Estimate where new observations might fall.

```rust
use fastLowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .fraction(0.5)
        .prediction_intervals(0.95)  // 95% PI
        .build()?;

    let result = model.fit(&x, &y)?;

    if let (Some(lower), Some(upper)) = (&result.prediction_lower, &result.prediction_upper) {
        println!("Prediction bounds: [{:.2}, {:.2}]", lower[0], upper[0]);
    }

    Ok(())
}
```

```output
Prediction bounds: [-0.04, 0.71]
```

---

## Both Intervals

Request both types simultaneously:

```rust
use fastLowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .fraction(0.5)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .build()?;
    let result = model.fit(&x, &y)?;

    if let (Some(lo), Some(hi)) = (&result.confidence_lower, &result.confidence_upper) {
        println!("First point 95% CI: [{}, {}]", lo[0], hi[0]);
    }
    Ok(())
}
```

```output
First point 95% CI: [0.2929290861928646, 0.3721536723881129]
```

---

## Confidence Levels

Common levels and their z-values:

| Level | z-value | Interpretation |
| --- | --- | --- |
| 0.90 | 1.645 | 90% of intervals contain true value |
| 0.95 | 1.960 | 95% of intervals contain true value |
| 0.99 | 2.576 | 99% of intervals contain true value |

```rust
use fastLowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    // 99% confidence interval
    let model = Lowess::new()
        .confidence_intervals(0.99)
        .build()?;
    let result = model.fit(&x, &y)?;

    if let Some(lo) = &result.confidence_lower {
        println!("First lower CI bound (99%): {}", lo[0]);
    }
    Ok(())
}
```

```output
First lower CI bound (99%): 0.31746448671917393
```

---

## Standard Errors

Access standard errors directly (available when intervals are computed):

```rust
use fastLowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new().outputs(["se"]).build()?;
    let result = model.fit(&x, &y)?;

    if let Some(standard_errors) = &result.standard_errors {
        for (i, &se) in standard_errors.iter().enumerate().take(3) {
            println!("Point {}: SE = {:.4}", i, se);
        }
    }

    Ok(())
}
```

```output
Point 0: SE = 0.0246
Point 1: SE = 0.0252
Point 2: SE = 0.0258
```

---

## Bootstrap Intervals

The intervals above are analytic: a local-linear standard error times a normal z-score. That assumes roughly normal residuals. `.bootstrap_intervals(n_boot)` replaces it with a residual bootstrap (Batch, Streaming, or full-update Online), which does not:

1. Fit once, and center the residuals.
2. Refit `n_boot` times on `y_hat + residuals drawn with replacement` (same `x` and smoothing settings; cross-validation is not repeated).
3. Per point, the standard error is the sample standard deviation of those refits. A confidence interval is their percentile interval. A prediction interval is the percentile interval of each refit plus a freshly drawn residual, so skewed noise produces an asymmetric tail.

`.bootstrap_seed(seed)` fixes the draws. If omitted, a fixed default seed is used, so results are reproducible either way. Fewer than 2 replicates is rejected at `.build()` as `InvalidBootstrapSamples`. In Streaming, each combined chunk (including its incoming overlap) is bootstrapped independently with the same seed; bounds and SEs are then merged according to `merge_strategy`. These are local chunk intervals, not whole-stream bootstrap intervals. In Online, each full update bootstraps the current sliding window with the same seed and returns only the newest point's SE and bounds (starting at 3 points). The default incremental mode rejects bootstrap with `StandardErrorRequiresFullUpdateMode`.

With `parallel(true)` (the default) Batch and Streaming refits run concurrently and match a one-at-a-time run. GPU Batch, Online, and `parallel(false)` refit one replicate at a time; a GPU refit still uses the GPU fit pass. Each replicate is a full refit, so this is much slower than the analytic intervals. It replaces `result.standard_errors` in Batch/Streaming or `output.standard_error` in Online when `"se"` or an interval was requested.

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let n = 40usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|xi| xi.sin()).collect();

    let model = Lowess::new()
        .fraction(0.5)
        .iterations(0)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .bootstrap_intervals(40)
        .bootstrap_seed(7)
        .build()?;
    let result = model.fit(&x, &y)?;

    let (lo, hi) = (
        &result.confidence_lower.unwrap(),
        &result.confidence_upper.unwrap(),
    );
    let (plo, phi) = (
        &result.prediction_lower.unwrap(),
        &result.prediction_upper.unwrap(),
    );
    println!(
        "x={:.0}: y={:.2} CI [{:.2}, {:.2}] PI [{:.2}, {:.2}]",
        result.x[0], result.y[0], lo[0], hi[0], plo[0], phi[0]
    );
    Ok(())
}
```

```output
x=0: y=0.07 CI [-0.53, 0.66] PI [-1.14, 1.08]
```

---

## Streaming Bootstrap

Streaming refits each combined chunk with its incoming overlap, then merges the resulting bounds. The seed is restarted for each chunk; this does not bootstrap the entire stream as one dataset. In fastLowess, replicate refits run concurrently by default.

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.2).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let mut model = StreamingLowess::new()
        .chunk_size(15)
        .overlap(3)
        .confidence_intervals(0.95)
        .bootstrap_intervals(40)
        .bootstrap_seed(7)
        .build()?;
    let first = model.process_chunk(&x[..15], &y[..15])?;
    let second = model.process_chunk(&x[15..], &y[15..])?;
    let tail = model.finalize()?;
    assert_eq!(first.y.len() + second.y.len() + tail.y.len(), x.len());
    assert!(second.confidence_lower.is_some());
    Ok(())
}
```

---

## Online Bootstrap

Full-update Online bootstraps the current sliding window at each new point, returning only the newest point's intervals. It refits sequentially and restarts the seed for each window. Set `min_points` to at least 3 to get intervals from the first emitted update.

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let mut model = OnlineLowess::new()
        .update_mode("full")
        .window_capacity(10)
        .min_points(3)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .bootstrap_intervals(40)
        .bootstrap_seed(7)
        .build()?;
    for i in 0..12 {
        let x = i as f64 * 0.2;
        if let Some(output) = model.add_point(x, x.sin())? {
            assert!(output.standard_error.is_some());
            assert!(output.confidence_lower.is_some());
            assert!(output.prediction_upper.is_some());
        }
    }
    Ok(())
}
```

---

## Availability

!!! note "Intervals in all three adapters"
    Analytic and residual-bootstrap intervals are available in **Batch**, **Streaming** (see [Streaming Adapter](crate::doc::api::streaming)), and **Online** mode (`update_mode("full")` only — see [Online Adapter](crate::doc::api::online)). Bootstrap resamples independently per chunk or sliding window. Batch and Streaming refits run concurrently with `parallel(true)`; Online refits sequentially.

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✓ | ✓ (`update_mode("full")` only) |
| Prediction intervals | ✓ | ✓ | ✓ (`update_mode("full")` only) |
| Standard errors | ✓ | ✓ | ✓ (`update_mode("full")` only) |
| Residual bootstrap | ✓ | ✓ (per chunk) | ✓ (`update_mode("full")`, per window) |
