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
