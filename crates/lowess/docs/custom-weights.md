<!-- markdownlint-disable MD033 -->
# Custom Weights

Per-observation weights that encode data quality directly into the LOWESS fit.

## How Custom Weights Work

Standard LOWESS assigns equal prior trust to all observations. Custom weights
let you override this assumption point by point — before any distance or
robustness weighting is applied.

The effective weight of observation $j$ in a local fit centred at $x_i$ is:

$$w_{ij} = \text{custom\_weights}_j \times K\!\left(\frac{d_{ij}}{h_i}\right) \times r_j$$

where $K$ is the distance kernel, $h_i$ is the local bandwidth, and $r_j$ is
the robustness weight from the current iteration.

!!! note "Batch adapter only"
    `custom_weights` applies in **Batch** mode. It is silently ignored in
    Streaming and Online adapters.

---

## When to Use Custom Weights

| Situation | Recommended weight |
| --- | --- |
| Point known to be erroneous | `0.0` — fully excluded |
| Unreliable sensor / low precision | `0.1 – 0.5` |
| Standard observation | `1.0` (default) |
| Carefully calibrated measurement | `> 1.0` |
| Measurement uncertainty $\sigma_i$ | $1 / \sigma_i^2$ |

### Custom Weights vs. Robustness Iterations

Both mechanisms handle unreliable data, but they serve different purposes:

| | Custom Weights | Robustness Iterations |
| --- | --- | --- |
| **When known** | Before fitting | Computed from residuals |
| **Knowledge required** | Prior knowledge of quality | None — data-driven |
| **Effect** | Fixed throughout fit | Adapts each iteration |
| **Use case** | Known bad sensors, calibration | Unknown outlier contamination |

They compose: you can use both simultaneously. Custom weights suppress
*a priori* bad points; robustness iterations then handle any *residual*
outliers that remain.

---

## Basic Usage

### Suppress a Known Outlier

Set the weight to `0` at the bad point — it is excluded from every local fit
that would otherwise include it.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let mut y: Vec<f64> = x.iter().map(|v| v * 2.0).collect();
    y[5] = 100.0; // spike

    let mut weights = vec![1.0_f64; 10];
    weights[5] = 0.0; // exclude the spike

    let model = Lowess::new()
        .fraction(0.5)
        .iterations(0)
        .custom_weights(weights)
        .build()?;

    let result = model.fit(&x, &y)?;

    Ok(())
}
```

---

### Emphasize Important Points

Assign high weights to measurements you trust most — calibration standards,
reference instruments, or low-noise observations.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let calibration_indices = vec![5usize, 20, 40, 60, 80];
    let mut weights = vec![1.0_f64; x.len()];
    for &i in &calibration_indices {
        weights[i] = 10.0; // trust calibration 10× more
    }

    let model = Lowess::new()
        .fraction(0.5)
        .custom_weights(weights)
        .build()?;
    let result = model.fit(&x, &y)?;

    Ok(())
}
```

---

### Propagate Measurement Uncertainty

If each observation has a known standard deviation $\sigma_i$, set
$w_i = 1 / \sigma_i^2$ to give the fit information-theoretically optimal
weighting.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let sigma: Vec<f64> = (0..n).map(|i| 0.1 + (i % 4) as f64 * 0.1).collect();
    let weights: Vec<f64> = sigma.iter().map(|s| 1.0 / (s * s)).collect();

    let model = Lowess::new()
        .fraction(0.5)
        .custom_weights(weights)
        .build()?;
    let result = model.fit(&x, &y)?;

    Ok(())
}
```

---

## Combined with Robustness Iterations

Custom weights and robustness iterations compose naturally: use custom weights
for *known* bad points and robustness for *unknown* contamination.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let mut y: Vec<f64> = x.iter().map(|v| v * 1.5).collect();
    y[3]  = -50.0; // known bad
    y[12] = 80.0;  // unknown outlier

    let mut weights = vec![1.0_f64; 20];
    weights[3] = 0.0;

    let model = Lowess::new()
        .fraction(0.4)
        .iterations(3)
        .custom_weights(weights)
        .build()?;
    let result = model.fit(&x, &y)?;

    Ok(())
}
```

---

## Validation Rules

| Rule | Effect |
| --- | --- |
| Length must equal `n` | Error at fit time if mismatched |
| All values must be ≥ 0 | Negative weights are rejected |
| All-zero weight vector | Error: no points remain for any local fit |
| Uniform weights (`1.0` everywhere) | Identical result to omitting weights |

!!! warning "Zero-weight windows"
    If a local neighbourhood contains only zero-weight points, the fit at
    that centre point falls back to the behaviour specified by
    `zero_weight_fallback` (default: `"use_local_mean"`).

---

## See Also

- [Robustness](robustness.md) — adaptive outlier downweighting via IRLS
- [Parameters](parameters.md#custom_weights) — full parameter reference
