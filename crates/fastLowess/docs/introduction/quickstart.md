<!-- markdownlint-disable MD024 MD046 -->
# Quick Start

Get up and running with LOWESS in minutes.

## Basic Smoothing

Smooth a noisy sine wave — the kind of signal where LOWESS shines. Each example recovers the underlying trend from 100 points of Gaussian noise.

```rust
use fastLowess::prelude::*;
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
use fastLowess::prelude::*;
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
CI Lower: [0.2948175249759789, 0.3208061922742844, 0.34784232799017045, 0.3758601176114074, 0.40478599046090546, 0.4345334636147747, 0.46499767806973763, 0.4960498491766966, 0.5275320280601163, 0.5592527816273659, 0.5909845842362914, 0.6224637904475077, 0.6533939417101478, 0.6834528023579369, 0.7123029480037627, 0.7396050596306526, 0.7650324926415057, 0.7882853726428939, 0.8091025156351601, 0.8272698444470283, 0.8426245255663977, 0.8550545908216343, 0.8644942016431678, 0.8709149587695117, 0.8743139449566185, 0.8746999358400593, 0.8720817719437762, 0.8664667258946991, 0.857862773923606, 0.8462809011736221, 0.8317375822644445, 0.8142571327757436, 0.7938737241583091, 0.7706329546091757, 0.7445929528999542, 0.715825049336506, 0.6844140750651889, 0.6504583531104536, 0.6140694324655692, 0.5753716023202724, 0.5345012168816068, 0.4916058672852873, 0.4468434552649923, 0.40038124784334717, 0.35239501361765746, 0.3030683470488075, 0.25259226519197964, 0.20116510333613374, 0.14899264483491423, 0.096288316778182, 0.04327320584874658, -0.00982436004387906, -0.06276984020085181, -0.1153241545946906, -0.16724589223979647, -0.21829384937503254, -0.26822967042165957, -0.3168204013446332, -0.36384082668243434, -0.40907551942084996, -0.4523205750367585, -0.4933850293769904, -0.5320919815808481, -0.5682794626842294, -0.6018011072945929, -0.6325266950321665, -0.6603426245311202, -0.6851523625191438, -0.7068768758217239, -0.7254550127186118, -0.7408437633711036, -0.7530183084501649, -0.7619717673900495, -0.7677145814905298, -0.7702735023402747, -0.7696901874967993, -0.7660157463341732, -0.759298740765193, -0.7495874905831253, -0.736939379000362, -0.721431440291893, -0.7031699439153923, -0.6822976900176663, -0.6589981165939484, -0.6334955667573704, -0.6060514311334196, -0.5769564276947162, -0.5465199159207292, -0.515057679193327, -0.48287986526310145, -0.45028065616518276, -0.4175307860851856, -0.3848733881885873, -0.35252301774055467, -0.32066722017298915, -0.28946976035103744, -0.2590745954539768, -0.2296098022541356, -0.20119089108832078, -0.17392319585985294]
```

---

## Handling Outliers

LOWESS can robustly handle outliers through iterative reweighting:

```rust
use fastLowess::prelude::*;
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
use fastLowess::prelude::*;
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
