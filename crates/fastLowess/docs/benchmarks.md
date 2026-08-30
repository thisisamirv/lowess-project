# Benchmarks

## CPU Benchmarks

Speedup relative to R's `stats::lowess` (higher is better):

| Category | R (baseline) | fastLowess (Serial) | fastLowess (Parallel) |
| --- | --- | --- | --- |
| **Clustered** | 2.34ms | 2.0× | **2.5×** |
| **Constant Y** | 1.81ms | 1.7× | **3.2×** |
| **Extreme Outliers** | 5.81ms | 1.5× | **2.6×** |
| **Financial** (500–5K) | 0.65ms | **2.0×** | 1.4× |
| **Fraction** (0.05–0.67) | 3.8ms | 1.6× | **3.2×** |
| **Genomic** (1K–100K) | 11.2ms | 2.2× | **2.4×** |
| **High Noise** | 7.08ms | 1.5× | **3.6×** |
| **Iterations** (0–10) | 3.0ms | 1.9× | **2.7×** |
| **Scale** (1K–10K) | 1.6ms | 1.5× | **1.6×** |
| **Scientific** (500–5K) | 0.9ms | 1.4× | 1.4× |

*The R column shows the average time across scenarios in multi-scenario categories. Speedups are averages across the same range.*

## GPU Backend

For large batch datasets, the GPU backend can outperform CPU-parallel execution. The crossover point is driven by window size (`fraction × n`): at `fraction = 0.5`, GPU overtakes CPU around n ≥ 50K; at smaller fractions, around n ≥ 100K–250K. At n = 1M (`fraction = 0.5`, 3 iterations), GPU is **6.6×** faster than CPU-parallel (1.24s → 187ms). See [benchmarks/README.md](https://github.com/thisisamirv/lowess-project/blob/main/benchmarks/README.md#gpu-benchmarks) for the full sweep and transfer-overhead breakdown.

## Reproducing Benchmarks

Use `std::time::Instant` to time serial vs parallel runs:

```rust
use fastLowess::prelude::*;
use std::time::Instant;

fn main() -> Result<(), LowessError> {
    let n = 5000usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| xi.sin() + (((i * 7 + 3) % 17) as f64 / 17.0 - 0.5) * 0.6)
        .collect();

    let bench_ms = |parallel: bool, reps: u32| -> f64 {
        let run = || Lowess::new().fraction(0.67).parallel(parallel).build().unwrap().fit(&x, &y).unwrap();
        run(); // warm-up
        let t0 = Instant::now();
        for _ in 0..reps {
            run();
        }
        t0.elapsed().as_secs_f64() * 1000.0 / f64::from(reps)
    };

    let serial_ms = bench_ms(false, 10);
    let parallel_ms = bench_ms(true, 10);

    println!("Serial:   {:.2} ms", serial_ms);
    println!("Parallel: {:.2} ms", parallel_ms);
    println!("Speedup:  {:.2}x", serial_ms / parallel_ms);

    Ok(())
}
```
