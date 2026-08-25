<!-- markdownlint-disable MD024 MD033 -->
# Parameters

Complete reference for all LOWESS configuration options.

## Quick Reference

| Parameter | Default | Range/Options | Description | Adapter |
| --- | --- | --- | --- | --- |
| **fraction** | 0.67 | (0, 1] | Smoothing span | All |
| **iterations** | 3 | [0, 1000] | Robustness iterations | All |
| **delta** | `None` | [0, ∞) | Interpolation threshold | All |
| **weight_function** | `"tricube"` | 7 options | Distance kernel | All |
| **robustness_method** | `"bisquare"` | 3 options | Outlier weighting | All |
| **zero_weight_fallback** | `"use_local_mean"` | 3 options | Zero-weight behavior | All |
| **boundary_policy** | `"extend"` | 4 options | Edge handling | All |
| **scaling_method** | `"mad"` | 3 options | Scale estimation | All |
| **auto_converge** | `None` | tolerance | Early stopping | All |
| **return_residuals** | `false` | logical | Include residuals | All |
| **return_robustness_weights** | `false` | logical | Include weights | All |
| **return_se** | `false` | logical | Return standard errors | All |
| **return_diagnostics** | `false` | logical | Include metrics | Batch, Streaming |
| **custom_weights** | `None` | positive | Per-observation weights | Batch |
| **confidence_intervals** | `None` | (0, 1) | CI level | Batch |
| **prediction_intervals** | `None` | (0, 1) | PI level | Batch |
| **cv_method** | `None` | method | Auto-select fraction | Batch |
| **chunk_size** | 5000 | [10, ∞) | Points per chunk | Streaming |
| **overlap** | 500 | [0, chunk) | Overlap between chunks | Streaming |
| **merge_strategy** | `"weighted_average"` | 4 options | Merge overlaps | Streaming |
| **window_capacity** | 1000 | [3, ∞) | Max window size | Online |
| **min_points** | 2 | [2, window] | Min before output | Online |
| **update_mode** | `"incremental"` | 2 options | Update strategy | Online |

!!! note "Rust option values"
    In Rust, pass option-like parameters as strings (case-insensitive), e.g. `"tricube"`, `"bisquare"`, `"extend"`, `"average"`.

---

## Parameter Options Summary

| Parameter | Available Options |
| --- | --- |
| **weight_function** | `"tricube"`, `"epanechnikov"`, `"gaussian"`, `"biweight"`, `"cosine"`, `"triangle"`, `"uniform"` |
| **robustness_method** | `"bisquare"`, `"huber"`, `"talwar"` |
| **zero_weight_fallback** | `"use_local_mean"`, `"return_original"`, `"return_none"` |
| **boundary_policy** | `"extend"`, `"reflect"`, `"zero"`, `"noboundary"` |
| **scaling_method** | `"mad"`, `"mar"`, `"mean"` |
| **merge_strategy** | `"average"`, `"weighted_average"`, `"take_first"`, `"take_last"` |
| **update_mode** | `"incremental"`, `"full"` |

---

## Core Parameters

### fraction

The proportion of data used for each local fit. **Most important parameter.**

| Value | Effect | Use Case |
| --- | --- | --- |
| 0.1–0.3 | Fine detail | Rapidly changing signals |
| 0.3–0.5 | Balanced | General purpose |
| 0.5–0.7 | Heavy smoothing | Noisy data |
| 0.7–1.0 | Very smooth | Trend extraction |

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .fraction(0.3)
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("First smoothed value (fraction=0.3): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (fraction=0.3): 0.2582436164551495
```

---

### iterations

Number of robustness iterations for outlier resistance.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1–3 | Moderate | Recommended |
| 4–6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .iterations(5)
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("First smoothed value (5 iterations): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (5 iterations): 0.3826026982388202
```

---

### delta

Interpolation optimization threshold. Points within `delta` distance reuse the previous fit.

- **Default**: 1% of x-range (Batch), 0.0 (Streaming/Online)
- **Effect**: Higher values = faster but less accurate

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .delta(0.05)
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("First smoothed value: {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value: 0.38260776436644134
```

---

### weight_function

Distance weighting kernel for local fits.

| Kernel | Efficiency | Smoothness |
| --- | --- | --- |
| `"tricube"` | 0.998 | Very smooth |
| `"epanechnikov"` | 1.000 | Smooth |
| `"gaussian"` | 0.961 | Infinite |
| `"biweight"` | 0.995 | Very smooth |
| `"cosine"` | 0.999 | Smooth |
| `"triangle"` | 0.989 | Moderate |
| `"uniform"` | 0.943 | None |

See [Weight Functions](kernels.md) for detailed comparison.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .weight_function("epanechnikov")
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("First smoothed value (epanechnikov kernel): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (epanechnikov kernel): 0.40672777844316566
```

---

### robustness_method

Method for downweighting outliers during iterative refinement.

| Method | Behavior | Use Case |
| --- | --- | --- |
| `"bisquare"` | Smooth downweighting | General-purpose |
| `"huber"` | Linear beyond threshold | Moderate outliers |
| `"talwar"` | Hard threshold (0 or 1) | Extreme contamination |

See [Robustness](robustness.md) for detailed comparison.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .robustness_method("talwar")
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("First smoothed value (talwar robustness): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (talwar robustness): 0.37877937542419027
```

---

### boundary_policy

Edge handling strategy to reduce boundary bias. See [Boundary Handling](boundary.md) for a detailed comparison.

![Boundary Policy](https://raw.githubusercontent.com/thisisamirv/lowess-project/main/crates/lowess/assets/diagrams/boundary_comparison.svg)

| Policy | Behavior | Use Case |
| --- | --- | --- |
| `"extend"` | Pad with first/last values | Most cases (default) |
| `"reflect"` | Mirror data at boundaries | Periodic/symmetric data |
| `"zero"` | Pad with zeros | Data approaches zero |
| `"noboundary"` | No padding | Original Cleveland behavior |

For example:

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .boundary_policy("reflect")
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("First smoothed value (reflect boundary): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (reflect boundary): 0.7127616908322939
```

---

### scaling_method

Method for estimating residual scale during robustness iterations. See [Scaling Methods](scaling.md) for a detailed comparison.

![Scaling Methods](https://raw.githubusercontent.com/thisisamirv/lowess-project/main/crates/lowess/assets/diagrams/scaling_comparison.svg)

| Method | Description | Robustness |
| --- | --- | --- |
| `"mad"` | Median Absolute Deviation | Very robust |
| `"mar"` | Median Absolute Residual | Robust |
| `"mean"` | Mean Absolute Residual | Less robust |

For example:

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .scaling_method("mad")
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("First smoothed value (mad scaling): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (mad scaling): 0.38260776436644134
```

---

### zero_weight_fallback

Behavior when all neighborhood weights are zero.

![Zero Weight Fallback](https://raw.githubusercontent.com/thisisamirv/lowess-project/main/crates/lowess/assets/diagrams/zero_weight_comparison.svg)

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` | Use mean of neighborhood (default) |
| `"return_original"` | Return original y value |
| `"return_none"` | Return NaN |

For example:

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .zero_weight_fallback("use_local_mean")
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("First smoothed value (use_local_mean fallback): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (use_local_mean fallback): 0.38260776436644134
```

---

### auto_converge

Enable early stopping when robustness weights stabilize.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .iterations(20)           // Maximum
        .auto_converge(1e-6)      // Stop when change < 1e-6
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("First smoothed value (auto-converge): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (auto-converge): 0.3826022035717678
```

---

### custom_weights

Per-observation weights applied before distance and robustness weighting. Only
available in the **Batch** adapter.

!!! note "Batch only"
    `custom_weights` is silently ignored in Streaming and Online adapters.

See [Custom Weights](custom-weights.md) for a full discussion.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut weights = vec![1.0_f64; x.len()];
    weights[5] = 0.0; // exclude index 5

    let model = Lowess::new()
        .fraction(0.5)
        .custom_weights(weights)
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("First smoothed value (custom weights): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (custom weights): 0.33693529914928816
```

---

## Output Options

### return_residuals

Include residuals (`y - smoothed`) in the output.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .return_residuals()
        .build()?;

    let result = model.fit(&x, &y)?;
    if let Some(residuals) = result.residuals {
        println!("Residuals: {:?}", residuals);
    }

    Ok(())
}
```

```output
Residuals: [-0.2826077643664413, -0.2421047055806419, -0.2023252539875934, -0.16342590937459034, -0.12555213599026638, -0.08883719953095592, -0.053401370889668454, -0.019351524231779327, 0.013218872508075186, 0.04422940839827638, 0.07361208376572947, 0.10131044977652515, 0.12727862173978144, 0.15148024666770088, 0.17388748554153954, 0.1944800439078267, 0.21324425828102933, 0.23017222768346568, 0.24526097464907848, 0.2585116296700084, 0.26992865497305396, 0.27951915226160773, 0.2872923279131634, 0.2932592118385985, 0.2974327382718558, 0.2998282956000793, 0.30046483644168487, 0.29936660623640554, 0.29656549323997594, 0.2921039142904486, 0.286038011569699, 0.2784407223112082, 0.26940397125971716, 0.2590388112491694, 0.24747182677419544, 0.2348374914444724, 0.22127288562328395, 0.20691364009340796, 0.19189015031950685, 0.17632427202040835, 0.16032667215588547, 0.14399495329756618, 0.1274126013495409, 0.11064873689863086, 0.09375858958453154, 0.07678457125178972, 0.059757802033954255, 0.04269994427587698, 0.025625218626307816, 0.008542508139116611, -0.008542508139116084, -0.025625218626307733, -0.04269994427587687, -0.05975780203395373, -0.07678457125178959, -0.09375858958453107, -0.11064873689863042, -0.12741260134954047, -0.14399495329756595, -0.16032667215588525, -0.17632427202040823, -0.19189015031950674, -0.20691364009340762, -0.2212728856232843, -0.23483749144447175, -0.24747182677419544, -0.25903881124916994, -0.26940397125971804, -0.27844072231120887, -0.28603801156970043, -0.2921039142904507, -0.29656549323997816, -0.299366606236408, -0.30046483644168753, -0.29982829560008195, -0.29743273827185934, -0.29325921183860204, -0.2872923279131677, -0.2795191522616116, -0.2699286549730582, -0.25851162967001284, -0.24526097464908236, -0.2301722276834699, -0.21324425828103277, -0.19448004390783025, -0.1738874855415431, -0.15148024666770377, -0.12727862173978444, -0.10131044977652759, -0.07361208376573114, -0.04422940839827738, -0.013218872508075519, 0.01935152423177966, 0.05340137088966934, 0.08883719953095723, 0.12555213599026835, 0.1634259093745926, 0.20232525398759607, 0.2421047055806452, 0.28260776436644586]
```

---

### return_diagnostics

Include fit quality metrics (Batch and Streaming only).

| Metric | Description |
| --- | --- |
| `rmse` | Root Mean Square Error |
| `mae` | Mean Absolute Error |
| `r_squared` | R² coefficient |
| `residual_sd` | Residual standard deviation |
| `effective_df` | Effective degrees of freedom |
| `aic` | Akaike Information Criterion |
| `aicc` | Corrected AIC |

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .return_diagnostics()
        .build()?;

    let result = model.fit(&x, &y)?;
    if let Some(diag) = result.diagnostics {
        println!("R²: {:.4}", diag.r_squared);
        println!("RMSE: {:.4}", diag.rmse);
    }

    Ok(())
}
```

```output
RÂ²: 0.9149
RMSE: 0.2052
```

---

### return_robustness_weights

Include final robustness weights (useful for outlier detection).

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .iterations(3)
        .return_robustness_weights()
        .build()?;

    let result = model.fit(&x, &y)?;
    // Points with weight < 0.5 are likely outliers

    if let Some(w) = &result.robustness_weights {
        println!("First robustness weight: {}", w[0]);
    }
    Ok(())
}
```

```output
First robustness weight: 0.8001006745815438
```

---

### return_se

Return per-point standard errors for the smoothed fit. Standard errors measure the uncertainty of each smoothed estimate and are used as the basis for confidence and prediction intervals when those are requested alongside `return_se`.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .return_se()
        .build()?;

    let result = model.fit(&x, &y)?;
    if let Some(se) = result.standard_errors {
        println!("SE: {:?}", se);
    }

    Ok(())
}
```

```output
SE: [0.02541344641903719, 0.02691432056725705, 0.02834063363287824, 0.029677718317099255, 0.030911785824824487, 0.03203069602390802, 0.03302459415183605, 0.033886383837986804, 0.034612024896797974, 0.03520065894796089, 0.03565457633047681, 0.03597904448758065, 0.03618202180328017, 0.036273782616398347, 0.036266479517377735, 0.0361736684497988, 0.036009820682424996, 0.03578984326403872, 0.03552862596948421, 0.03524062800886213, 0.03493951221991061, 0.034637828720073635, 0.03434674487866595, 0.034075814819733756, 0.033832780177647646, 0.03362339495931973, 0.033451271340416075, 0.033317750081426914, 0.03322180892412847, 0.03316003464862016, 0.03312669907678045, 0.03311399547682607, 0.03311250825993116, 0.033112003484920785, 0.03310263706239262, 0.033076373991181816, 0.033027533210436576, 0.03295294125701763, 0.032851929441962295, 0.03272618262173555, 0.032579459590700334, 0.03241721379208042, 0.03224614750715517, 0.03207373296169869, 0.03190773070774174, 0.031755730477186765, 0.031624733803547425, 0.031520792150834386, 0.03144870972214126, 0.031411816748052754, 0.031411816748052754, 0.031448709722141276, 0.03152079215083444, 0.03162473380354749, 0.03175573047718685, 0.03190773070774187, 0.032073732961698784, 0.03224614750715532, 0.03241721379208056, 0.03257945959070052, 0.03272618262173574, 0.03285192944196251, 0.032952941257017844, 0.033027533210436756, 0.03307637399118206, 0.03310263706239286, 0.03311200348492101, 0.03311250825993141, 0.03311399547682627, 0.03312669907678062, 0.033160034648620294, 0.03322180892412859, 0.03331775008142701, 0.03345127134041611, 0.03362339495931979, 0.033832780177647674, 0.03407581481973379, 0.03434674487866597, 0.03463782872007364, 0.03493951221991063, 0.03524062800886218, 0.03552862596948425, 0.03578984326403881, 0.03600982068242512, 0.036173668449798906, 0.036266479517377895, 0.03627378261639853, 0.036182021803280374, 0.03597904448758087, 0.03565457633047707, 0.03520065894796119, 0.03461202489679829, 0.03388638383798712, 0.0330245941518364, 0.03203069602390838, 0.03091178582482481, 0.02967771831709959, 0.028340633632878545, 0.026914320567257326, 0.025413446419037405]
```

---

### confidence_intervals / prediction_intervals

Request uncertainty estimates (Batch only).

See [Intervals](intervals.md) for detailed usage.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
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
First point 95% CI: [0.3327974093851285, 0.4324181193477542]
```

---

## CV Methods

### cv_method

Selection strategy for automated parameter tuning.

| Method | Description | Speed |
| --- | --- | --- |
| `"kfold"` | K-Fold Cross-Validation | Fast |
| `"loocv"` | Leave-One-Out Cross-Validation | Slow |

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Lowess::new()
        .cv_method("kfold")
        .cv_k(5)
        .cv_fractions(vec![0.1, 0.3, 0.5])
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("Selected fraction: {}", result.fraction_used);
    Ok(())
}
```

```output
Selected fraction: 0.3
```

---

## Adapter Parameters

### chunk_size

Points per chunk in Streaming mode.

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
        let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * std::f64::consts::TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut model = StreamingLowess::new()
        .chunk_size(10000)
        .build()?;
    let _ = model.process_chunk(&x[..50], &y[..50])?;
    let _ = model.process_chunk(&x[50..], &y[50..])?;
    let result = model.finalize()?;
    println!("First smoothed value (chunk_size=10000): {}", result.y[0]);

    Ok(())
}
```

```output
First smoothed value (chunk_size=10000): 0.38260776436644134
```

---

### overlap

Overlap between chunks in Streaming mode.

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
        let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * std::f64::consts::TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut model = StreamingLowess::new()
        .overlap(1000)
        .build()?;
    let _ = model.process_chunk(&x[..50], &y[..50])?;
    let _ = model.process_chunk(&x[50..], &y[50..])?;
    let result = model.finalize()?;
    println!("First smoothed value (overlap=1000): {}", result.y[0]);

    Ok(())
}
```

```output
First smoothed value (overlap=1000): 0.38260776436644134
```

---

### merge_strategy

Method for merging overlapping chunks. See [Merge Strategies](merge.md) for a detailed comparison.

| Strategy | Description | Robustness |
| --- | --- | --- |
| `"average"` | Average of overlapping chunks | Faster, less accurate |
| `"take_first"` | Use value from first chunk | Fastest, least accurate |
| `"take_last"` | Use value from last chunk | Fastest, least accurate |
| `"weighted_average"` | Weighted average of overlapping chunks | Most accurate |

For example:

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
        let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * std::f64::consts::TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut model = StreamingLowess::new()
        .merge_strategy("weighted_average")
        .build()?;
    let _ = model.process_chunk(&x[..50], &y[..50])?;
    let _ = model.process_chunk(&x[50..], &y[50..])?;
    let result = model.finalize()?;
    println!("First smoothed value (weighted_average merge): {}", result.y[0]);

    Ok(())
}
```

```output
First smoothed value (weighted_average merge): 0.38260776436644134
```

---

### window_capacity

Maximum points held in memory for Online mode.

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
        let x = vec![0.0f64, 0.5, 1.0, 2.0, 3.0];
    let y = vec![0.0f64, 0.5, 0.9, 1.9, 2.9];

    let mut model = OnlineLowess::new()
        .window_capacity(500)
        .build()?;
    let out = model.add_point(x[0], y[0])?;
    println!("add_point result (window_capacity=500): {:?}", out);

    Ok(())
}
```

```output
add_point result (window_capacity=500): None
```

---

### min_points

Minimum points required before Online filter starts producing outputs.

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
        let x = vec![0.0f64, 0.5, 1.0, 2.0, 3.0];
    let y = vec![0.0f64, 0.5, 0.9, 1.9, 2.9];

    let mut model = OnlineLowess::new()
        .min_points(10)
        .build()?;
    let out = model.add_point(x[0], y[0])?;
    println!("add_point result (min_points=10): {:?}", out);

    Ok(())
}
```

```output
add_point result (min_points=10): None
```

---

### update_mode

Optimization strategy for Online mode updates.

| Mode | Description | Speed |
| --- | --- | --- |
| `"full"` | Full update | Slow |
| `"incremental"` | Incremental update | Fast |

For example:

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
        let x = vec![0.0f64, 0.5, 1.0, 2.0, 3.0];
    let y = vec![0.0f64, 0.5, 0.9, 1.9, 2.9];

    let mut model = OnlineLowess::new()
        .update_mode("full")
        .build()?;
    let out = model.add_point(x[0], y[0])?;
    println!("add_point result (full mode): {:?}", out);

    Ok(())
}
```

```output
add_point result (full mode): None
```
