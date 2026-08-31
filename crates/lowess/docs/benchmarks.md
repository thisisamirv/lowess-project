# Benchmarks

## CPU Benchmarks

Speedup relative to R's `stats::lowess` (higher is better):

| Category | R (baseline) | lowess (Serial) |
| --- | --- | --- |
| **Clustered** | 2.34ms | 2.0× |
| **Constant Y** | 1.81ms | 1.7× |
| **Extreme Outliers** | 5.81ms | 1.5× |
| **Financial** (500–5K) | 0.65ms | 2.0× |
| **Fraction** (0.05–0.67) | 3.8ms | 1.6× |
| **Genomic** (1K–100K) | 11.2ms | 2.2× |
| **High Noise** | 7.08ms | 1.5× |
| **Iterations** (0–10) | 3.0ms | 1.9× |
| **Large** (50K, delta=0) | 5805.90ms | 1.9× |
| **Large** (50K, delta=auto) | 14.46ms | 1.7× |
| **Large** (50K, 10 iter) | 31694.32ms | 3.5× |
| **Large** (20K, fraction=0.67) | 12627.11ms | 3.6× |
| **Scale** (1K–10K) | 1.6ms | 1.5× |
| **Scientific** (500–5K) | 0.9ms | 1.4× |

*The R column shows the average time across scenarios in multi-scenario categories. Speedups are averages across the same range. The `lowess` crate has no `parallel` or `gpu` feature — for CPU-parallel and GPU-accelerated numbers, see the `fastLowess` crate's benchmarks.*

## Reproducing Benchmarks

Use `std::time::Instant` to time fit calls:

```rust
use lowess::prelude::*;
use std::time::Instant;

fn main() -> Result<(), LowessError> {
    let n = 5000usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| xi.sin() + (((i * 7 + 3) % 17) as f64 / 17.0 - 0.5) * 0.6)
        .collect();

    let reps = 10u32;
    let run = || Lowess::new().fraction(0.67).build().unwrap().fit(&x, &y).unwrap();
    run(); // warm-up
    let t0 = Instant::now();
    for _ in 0..reps {
        run();
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(reps);

    println!("Fit: {:.2} ms", elapsed_ms);

    Ok(())
}
```

```output
Fit: 296.82 ms
```
