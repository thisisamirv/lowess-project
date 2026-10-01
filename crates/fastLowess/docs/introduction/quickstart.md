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
First smoothed value: 0.2152  (true: 0.0000)
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
        .intervals(IntervalsBuilder::new()
            .confidence(0.95)
            .prediction(0.95)
        )  // 95% CI and PI
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
CI Lower: [0.2929290861928646, 0.31902466873900626, 0.34618948800269234, 0.37434930090623914, 0.4034217329588277, 0.43331154267005684, 0.46390586262190836, 0.49506958471716156, 0.5266411741824534, 0.5584293387778717, 0.5902111036734498, 0.6217318957694087, 0.652708173262649, 0.6828329168739434, 0.711783937879324, 0.7392345143112518, 0.7648654407319708, 0.7883772795562551, 0.8095015121973228, 0.8280094206808621, 0.8437178282415728, 0.8564911896223116, 0.8662398509295715, 0.8729145551359228, 0.876497521041366, 0.8769908964526969, 0.874404932387371, 0.8687520280070438, 0.8600474931066626, 0.848311495372875, 0.8335710182168554, 0.8158616251848554, 0.795228902738514, 0.7717295236176867, 0.7454319293761383, 0.7164166681709915, 0.684776442040504, 0.6506159202900099, 0.6140513684834911, 0.5752101330134435, 0.5342300154208973, 0.4912585723212819, 0.4464523865364595, 0.3999763698782589, 0.3520031714375028, 0.302712767509633, 0.2522922892146884, 0.2009360928350606, 0.14884599705817758, 0.09623151916741887, 0.043309875297698294, -0.009694491237365113, -0.06255100652698486, -0.11502469737466117, -0.16687842123640823, -0.2178753663149116, -0.26778157899458155, -0.3163683245729471, -0.3634141634180582, -0.4087066964603362, -0.45204398594220774, -0.4932356889067133, -0.5321039579534271, -0.5684841720520716, -0.6022255622257107, -0.633191792338692, -0.6612615426467341, -0.6863291227838814, -0.7083051134346041, -0.7271170066428905, -0.7427097897549283, -0.7550464032654838, -0.7641080015395185, -0.7698939561194125, -0.772421557624088, -0.771725384152916, -0.7678538702476957, -0.7608617479473087, -0.7508125724904473, -0.7377857661885491, -0.7218841289911304, -0.7032401670080535, -0.6820202741714151, -0.658426144081869, -0.6326930770249088, -0.6050852037829486, -0.5758880710303423, -0.5453994398955949, -0.5139194284435123, -0.4817411949614107, -0.4491431909540298, -0.4163836617003211, -0.3836976377445203, -0.3512962509800273, -0.31936790482345556, -0.28808066529015686, -0.2575852128923811, -0.22801777317657768, -0.1995025884647552, -0.17215367238811363]
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
