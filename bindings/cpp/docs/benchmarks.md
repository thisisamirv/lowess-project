\page benchmarks Benchmarks

# Benchmarks

## CPU Benchmarks

Speedup relative to R's `stats::lowess` (higher is better):

| Category | R baseline | Serial | Parallel |
| --- | --- | --- | --- |
| **Clustered** | 2.34 ms | 2.0× | **2.5×** |
| **Constant Y** | 1.81 ms | 1.7× | **3.2×** |
| **Extreme Outliers** | 5.81 ms | 1.5× | **2.6×** |
| **Financial** (500–5K) | 0.65 ms | **2.0×** | 1.4× |
| **Fraction** (0.05–0.67) | 3.8 ms | 1.6× | **3.2×** |
| **Genomic** (1K–100K) | 11.2 ms | 2.2× | **2.4×** |
| **High Noise** | 7.08 ms | 1.5× | **3.6×** |
| **Iterations** (0–10) | 3.0 ms | 1.9× | **2.7×** |
| **Large** (50K, delta=0) | 5805.90 ms | 1.9× | **5.5×** |
| **Large** (50K, delta=auto) | 14.46 ms | 1.7× | **2.2×** |
| **Large** (50K, 10 iter) | 31694.32 ms | 3.5× | **9.7×** |
| **Large** (20K, fraction=0.67) | 12627.11 ms | 3.6× | **17.4×** |
| **Scale** (1K–10K) | 1.6 ms | 1.5× | **1.6×** |
| **Scientific** (500–5K) | 0.9 ms | 1.4× | 1.4× |

*The R column shows average time across scenarios in multi-scenario categories.
Speedups are averages across the same range.*

---

## GPU Backend

For large batch datasets the GPU backend outperforms CPU-parallel execution.
The crossover depends on `fraction × n`:

| Scenario | CPU-Parallel | GPU | Speedup |
| --- | --- | --- | --- |
| n = 1M, fraction = 0.5 | 1.24 s | 187 ms | **6.6×** |

At `fraction = 0.5`, GPU overtakes CPU around n ≥ 50K; at smaller fractions,
around n ≥ 100K–250K. See the benchmarks README in the source repository for
the full sweep and transfer-overhead breakdown.
