# Rust Examples

Complete code examples for the `lowess` and `fastLowess` crates.

## Basic Example

```rust
use fastLowess::prelude::*;
use ndarray::Array1;

fn main() -> Result<(), LowessError> {
    // Sample data
    let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let y = Array1::from_vec(vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7]);

    // Build and fit
    let model = Lowess::new()
        .fraction(0.5)
        .iterations(3)
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    
    println!("Smoothed values: {:?}", result.y);
    Ok(())
}
```

---

## With Confidence Intervals

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = x.iter().map(|&xi| (xi).sin() + rand::random::<f64>() * 0.2).collect();

    let model = Lowess::new()
        .fraction(0.3)
        .iterations(3)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .return_diagnostics()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;

    // Access results
    println!("R²: {:.4}", result.diagnostics.unwrap().r_squared);
    
    if let (Some(ci_lower), Some(ci_upper)) = (&result.confidence_lower, &result.confidence_upper) {
        for i in 0..5 {
            println!(
                "x={:.1}: y={:.3} [{:.3}, {:.3}]",
                result.x[i], result.y[i], ci_lower[i], ci_upper[i]
            );
        }
    }

    Ok(())
}
```

---

## Cross-Validation

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x: Vec<f64> = (0..200).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * 0.5 + rand::random::<f64>() * 10.0).collect();

    let model = Lowess::new()
        .cross_validate(KFold(5, &[0.1, 0.2, 0.3, 0.5, 0.7]).seed(42))
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;

    println!("Selected fraction: {}", result.fraction_used);
    
    if let Some(cv) = &result.cv_results {
        println!("CV scores: {:?}", cv.scores);
    }

    Ok(())
}
```

---

## Streaming Processing

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    // Simulate large dataset in chunks
    let mut processor = Lowess::new()
        .fraction(0.1)
        .iterations(2)
        .adapter(Streaming)
        .chunk_size(1000)
        .overlap(100)
        .merge_strategy(Weighted)
        .build()?;

    // Process chunk 1
    let x1: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let y1: Vec<f64> = x1.iter().map(|&x| x.sin()).collect();
    let result1 = processor.process_chunk(&x1, &y1)?;
    println!("Chunk 1: {} points", result1.y.len());

    // Process chunk 2
    let x2: Vec<f64> = (900..2000).map(|i| i as f64).collect();
    let y2: Vec<f64> = x2.iter().map(|&x| x.sin()).collect();
    let result2 = processor.process_chunk(&x2, &y2)?;
    println!("Chunk 2: {} points", result2.y.len());

    // Finalize
    let final_result = processor.finalize()?;
    println!("Final: {} points", final_result.y.len());

    Ok(())
}
```

---

## Online Processing

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let mut processor = Lowess::new()
        .fraction(0.3)
        .iterations(1)
        .adapter(Online)
        .window_capacity(50)
        .min_points(5)
        .update_mode(Incremental)
        .build()?;

    // Simulate sensor data
    for i in 0..100 {
        let x = i as f64;
        let y = 20.0 + 5.0 * (x / 10.0).sin();

        if let Some(output) = processor.add_point(x, y)? {
            println!(
                "t={:3}: raw={:.2}, smoothed={:.2}",
                i, y, output.smoothed
            );
        } else {
            println!("t={:3}: buffering...", i);
        }
    }

    Ok(())
}
```

---

## Outlier Detection

```rust
use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    // Data with outliers
    let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let mut y: Vec<f64> = x.iter().map(|&xi| xi * 2.0).collect();
    
    // Add outliers
    y[10] = 100.0;
    y[25] = -50.0;
    y[40] = 150.0;

    let model = Lowess::new()
        .fraction(0.3)
        .iterations(5)
        .robustness_method(Bisquare)
        .return_robustness_weights()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;

    // Find outliers
    if let Some(weights) = &result.robustness_weights {
        println!("Detected outliers:");
        for (i, &w) in weights.iter().enumerate() {
            if w < 0.5 {
                println!("  Index {}: weight = {:.3}", i, w);
            }
        }
    }

    Ok(())
}
```

---

## no_std Example (lowess crate)

```rust
#![no_std]

use lowess::prelude::*;

fn smooth_data(x: &[f64], y: &[f64]) -> Result<Vec<f64>, LowessError> {
    let model = Lowess::new()
        .fraction(0.5)
        .iterations(2)
        .adapter(Batch)
        .build()?;

    let result = model.fit(x, y)?;
    Ok(result.y)
}
```

---

## Feature Flags

```toml
# Cargo.toml

# Minimal (no_std compatible)
[dependencies]
lowess = "0.99"

# With parallelism
[dependencies]
fastLowess = { version = "0.99", features = ["cpu"] }

# With GPU (beta)
[dependencies]
fastLowess = { version = "0.99", features = ["gpu"] }
```
