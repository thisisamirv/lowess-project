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
    println!("First smoothed: {:.4}  (true: {:.4})", result.y[0], x[0].sin());
    Ok(())
}
```

```output
First smoothed: 0.2250  (true: 0.0000)
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
CI Lower: [0.29412988645250643, 0.3183970427530898, 0.3439982438031091, 0.3708562088304487, 0.39888436443844216, 0.42798000814166653, 0.45801759790461655, 0.4888427010289136, 0.5202671343213984, 0.5520658647249862, 0.5839762722157145, 0.6157003386602651, 0.6469101513208138, 0.6772567633016634, 0.7063819633092272, 0.7339319749095413, 0.7595716807943249, 0.7829977894912135, 0.8039494923001689, 0.822215546896859, 0.8376372304568442, 0.8501070667397208, 0.8595635449042679, 0.8659822322681734, 0.8693639240524249, 0.8697211619428963, 0.8670668420824398, 0.8614123225006612, 0.8527695139777194, 0.8411530880326414, 0.8265828507490243, 0.8090859882570998, 0.7886989873637626, 0.7654691325247516, 0.7394555613931489, 0.7107299151212211, 0.6793766439606067, 0.6454930287357777, 0.609188965038981, 0.5705865417601468, 0.5298194390104572, 0.4870321780527755, 0.44237927701524216, 0.39602439471545625, 0.34813957000539286, 0.298904671820252, 0.24850715143393728, 0.19714212381006835, 0.14501270247211587, 0.09233039564032461, 0.039315284710890276, -0.013804302406677502, -0.06679281972691742, -0.11940926835273137, -0.17140956746835218, -0.22254929298729784, -0.2725865235495515, -0.32128457959438306, -0.36841451591494484, -0.4137572972919993, -0.4571056355968832, -0.49826549680357846, -0.5370573059555241, -0.5733168937888119, -0.6068962415098789, -0.6376640865389703, -0.6655064466155437, -0.6903270993136906, -0.712048020340368, -0.7306097442340319, -0.7459715765120848, -0.7581115683960511, -0.7670261707840873, -0.7727295113518667, -0.7752522762374375, -0.7746402084009933, -0.7709484728355122, -0.7642293975040928, -0.7545350146650387, -0.7419266741099153, -0.7264857378420617, -0.7083229672503846, -0.6875852731693471, -0.6644589284411299, -0.6391686514784815, -0.611972415827956, -0.5831524667509869, -0.5530037063100631, -0.5218211309805693, -0.4898881772836776, -0.45746757306756064, -0.4247956798239021, -0.3920805363363705, -0.3595030979056771, -0.32722067564609736, -0.2953713863735007, -0.26407850423493573, -0.2334538864411965, -0.20360004060951584, -0.17461083438332475]
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
        .fraction(0.5)
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
    println!("Smoothed {} points", result.y.len());
    Ok(())
}
```

```output
Smoothed 100 points
```

---

## Next Steps

| Topic | Link |
| --- | --- |
| How LOWESS works | [Concepts](concepts.md) |
| All parameters explained | [Parameters](../user-guide/parameters.md) |
| Batch vs Streaming vs Online | [Execution Modes](../user-guide/adapters.md) |
| Edge handling | [Boundary](../user-guide/boundary.md) |
| Outlier handling in depth | [Robustness](../user-guide/robustness.md) |
| Full API per language | [API Reference](../api/index.md) |
