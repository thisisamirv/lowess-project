<!-- markdownlint-disable MD024 MD046 -->
# Quick Start

Get up and running with LOWESS in minutes.

## Basic Smoothing

Smooth a noisy sine wave — the kind of signal where LOWESS shines. Each example recovers the underlying trend from 100 points of Gaussian noise.

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    // 100-point noisy sine wave (deterministic)
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().enumerate()
        .map(|(i, &xi)| xi.sin() + ((i * 7 + 3) as f64 % 1.7 - 0.85) * 0.3)
        .collect();

    let model = Lowess::new()
        .fraction(0.3)
        .iterations(3)
        .build()?;

    let result = model.fit(&x, &y)?;
    println!("First smoothed value: {:.4}  (true: {:.4})", result.y[0], x[0].sin());
    Ok(())
}
```

```output
First smoothed value: 0.2250  (true: 0.0000)
```

---

## With Confidence Intervals

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();


    let model = Lowess::new()
        .fraction(0.5)
        .iterations(3)
        .confidence_intervals(0.95)  // 95% CI
        .prediction_intervals(0.95)  // 95% PI
        .return_diagnostics()
        .build()?;

    let result = model.fit(&x, &y)?;

    // Access intervals
    if let Some(ci_lower) = &result.confidence_lower {
        println!("CI Lower: {:?}", ci_lower);
    }

    Ok(())
}
```

```output
CI Lower: [0.2948175249759787, 0.32080619227428453, 0.3478423279901705, 0.3758601176114074, 0.40478599046090563, 0.43453346361477463, 0.46499767806973763, 0.4960498491766966, 0.5275320280601163, 0.5592527816273658, 0.5909845842362916, 0.6224637904475077, 0.653393941710148, 0.683452802357937, 0.7123029480037626, 0.739605059630653, 0.7650324926415059, 0.7882853726428943, 0.8091025156351604, 0.8272698444470288, 0.8426245255663976, 0.8550545908216344, 0.8644942016431678, 0.8709149587695116, 0.8743139449566188, 0.8746999358400595, 0.8720817719437766, 0.8664667258946991, 0.8578627739236058, 0.8462809011736225, 0.8317375822644447, 0.8142571327757437, 0.7938737241583093, 0.7706329546091751, 0.7445929528999539, 0.7158250493365063, 0.6844140750651886, 0.6504583531104524, 0.6140694324655691, 0.5753716023202715, 0.5345012168816068, 0.49160586728528677, 0.4468434552649927, 0.40038124784334805, 0.3523950136176588, 0.3030683470488073, 0.2525922651919784, 0.20116510333613422, 0.14899264483491506, 0.09628831677818059, 0.04327320584874625, -0.009824360043878304, -0.06276984020085154, -0.11532415459469025, -0.16724589223979686, -0.21829384937503196, -0.26822967042165974, -0.3168204013446326, -0.36384082668243356, -0.40907551942084974, -0.4523205750367585, -0.4933850293769903, -0.5320919815808494, -0.56827946268423, -0.6018011072945936, -0.6325266950321654, -0.6603426245311201, -0.685152362519144, -0.706876875821724, -0.7254550127186115, -0.7408437633711036, -0.7530183084501647, -0.7619717673900495, -0.76771458149053, -0.7702735023402745, -0.7696901874967994, -0.7660157463341739, -0.7592987407651929, -0.7495874905831251, -0.736939379000362, -0.721431440291892, -0.703169943915393, -0.6822976900176663, -0.658998116593949, -0.6334955667573697, -0.6060514311334207, -0.5769564276947133, -0.546519915920729, -0.5150576791933267, -0.48287986526310056, -0.4502806561651809, -0.4175307860851841, -0.3848733881885875, -0.352523017740556, -0.3206672201729892, -0.28946976035103716, -0.2590745954539769, -0.22960980225413508, -0.20119089108832097, -0.17392319585985247]
```

---

## Handling Outliers

LOWESS can robustly handle outliers through iterative reweighting:

```rust
use lowess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    // Data with an outlier at position 3
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y_with_outlier = vec![2.0, 4.0, 6.0, 50.0, 10.0, 12.0];  // 50.0 is outlier

    let model = Lowess::new()
        .fraction(0.7)
        .iterations(5)                    // More iterations for outliers
        .robustness_method("bisquare")    // Default, smooth downweighting
        .return_robustness_weights()      // See which points were downweighted
        .build()?;

    let result = model.fit(&x, &y_with_outlier)?;

    // Outliers will have low robustness weights
    if let Some(weights) = &result.robustness_weights {
        for (i, w) in weights.iter().enumerate() {
            if *w < 0.5 {
                println!("Point {} is likely an outlier (weight: {:.3})", i, w);
            }
        }
    }

    Ok(())
}
```

```output
Point 3 is likely an outlier (weight: 0.000)
```

---

## Streaming Mode

For datasets too large to fit in memory, stream them in fixed-size chunks with overlap.

```rust
use lowess::prelude::*;
use std::f64::consts::PI;

fn main() -> Result<(), LowessError> {
    let n = 5_000usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 10.0 * PI / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().enumerate()
        .map(|(i, &xi)| (xi / PI).sin() * (-xi / 30.0).exp()
                       + ((i * 7 + 3) as f64 % 1.7 - 0.85) * 0.15)
        .collect();

    let mut model = StreamingLowess::new()
        .fraction(0.2)
        .chunk_size(1000)
        .overlap(100)
        .build()?;

    for chunk in x.chunks(1000).zip(y.chunks(1000)) {
        model.process_chunk(chunk.0, chunk.1)?;
    }
    let result = model.finalize()?;
    println!("Smoothed {} points in streaming mode", result.y.len());
    Ok(())
}
```

```output
Smoothed 100 points in streaming mode
```

---

## Next Steps

| Topic | Link |
| --- | --- |
| How LOWESS works | [Concepts](crate::doc::introduction::concepts) |
| All parameters explained | [API Reference](crate::doc::api) |
| Batch vs Streaming vs Online | [Execution Modes](crate::doc::guide::adapter_choice) |
| Edge handling | [Boundary](crate::doc::advanced::boundary) |
| Outlier handling in depth | [Robustness](crate::doc::weighting::robustness) |
| Full API per language | [API Reference](crate::doc::api) |
