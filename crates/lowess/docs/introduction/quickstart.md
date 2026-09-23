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
        .outputs(["diagnostics"])
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
CI Lower: [0.29481752497597885, 0.3208061922742844, 0.3478423279901704, 0.3758601176114076, 0.4047859904609055, 0.43453346361477463, 0.46499767806973763, 0.4960498491766968, 0.5275320280601166, 0.5592527816273655, 0.5909845842362916, 0.6224637904475075, 0.6533939417101482, 0.6834528023579372, 0.7123029480037625, 0.7396050596306527, 0.7650324926415055, 0.7882853726428942, 0.8091025156351606, 0.8272698444470284, 0.8426245255663977, 0.8550545908216346, 0.8644942016431678, 0.8709149587695114, 0.8743139449566187, 0.8746999358400596, 0.8720817719437759, 0.8664667258946986, 0.8578627739236057, 0.8462809011736223, 0.8317375822644444, 0.8142571327757436, 0.7938737241583095, 0.7706329546091755, 0.7445929528999539, 0.7158250493365054, 0.6844140750651895, 0.6504583531104533, 0.6140694324655698, 0.5753716023202713, 0.5345012168816066, 0.49160586728528743, 0.44684345526499225, 0.4003812478433476, 0.35239501361765746, 0.30306834704880736, 0.2525922651919798, 0.20116510333613472, 0.14899264483491467, 0.09628831677818153, 0.04327320584874675, -0.00982436004387871, -0.06276984020085194, -0.11532415459469063, -0.16724589223979724, -0.21829384937503146, -0.2682296704216601, -0.3168204013446321, -0.36384082668243484, -0.40907551942085013, -0.45232057503675976, -0.4933850293769903, -0.5320919815808492, -0.5682794626842296, -0.6018011072945929, -0.6325266950321664, -0.6603426245311208, -0.6851523625191441, -0.706876875821724, -0.7254550127186115, -0.7408437633711036, -0.7530183084501648, -0.7619717673900495, -0.76771458149053, -0.7702735023402745, -0.7696901874967988, -0.7660157463341737, -0.7592987407651922, -0.7495874905831251, -0.7369393790003622, -0.7214314402918922, -0.7031699439153919, -0.6822976900176667, -0.658998116593949, -0.6334955667573684, -0.6060514311334207, -0.5769564276947156, -0.5465199159207295, -0.5150576791933271, -0.48287986526310056, -0.45028065616518187, -0.4175307860851859, -0.384873388188587, -0.352523017740556, -0.3206672201729901, -0.2894697603510381, -0.2590745954539769, -0.2296098022541355, -0.201190891088321, -0.1739231958598538]
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
        .outputs(["weights"])             // See which points were downweighted
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
