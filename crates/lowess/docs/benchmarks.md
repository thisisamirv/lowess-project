# Benchmarks

## CPU Benchmarks

Speedup relative to R's `stats::lowess` (higher is better):

| Category | R (baseline) | lowess (Serial) |
| --- | --- | --- |
| **Clustered** | 1.88ms | 0.8× |
| **Constant Y** | 1.50ms | 0.7× |
| **Extreme Outliers** | 5.32ms | 0.9× |
| **Financial** (500–5K) | 0.60ms | 1.0× |
| **Fraction** (0.05–0.67) | 3.24ms | 0.9× |
| **Genomic** (1K–100K) | 8.50ms | 1.0× |
| **High Noise** | 6.98ms | 0.9× |
| **Iterations** (0–10) | 2.28ms | 0.8× |
| **Large** (50K, delta=0) | 10157.00ms | 1.6× |
| **Large** (50K, delta=auto) | 11.48ms | 0.8× |
| **Large** (50K, 10 iter) | 37167.03ms | 2.7× |
| **Large** (20K, fraction=0.67) | 10900.56ms | 2.3× |
| **Scale** (1K–10K) | 1.36ms | 1.0× |
| **Scientific** (500–5K) | 0.87ms | 1.3× |

*The R column shows the average time across scenarios in multi-scenario categories. Speedups are averages across the same range. The `lowess` crate has no `parallel` or `gpu` feature — for CPU-parallel and GPU-accelerated numbers, see the `fastLowess` crate's benchmarks.*
