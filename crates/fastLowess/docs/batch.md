# Batch Adapter

Standard mode for complete datasets. **Supports all features.**

## When to Use

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

![Gap Handling](../assets/diagrams/gap_handling.svg)

## Example

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
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .return_diagnostics()
        .parallel(true)
        .build()?;

    let result = model.fit(&x, &y)?;

    if let Some(diag) = &result.diagnostics {
        println!("RMSE: {:.4}", diag.rmse);
    }
    Ok(())
}
```

```output
RMSE: 0.1289
```

---
