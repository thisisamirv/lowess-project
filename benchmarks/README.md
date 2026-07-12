# Benchmarks

Compares `stats::lowess` (base R) against `rfastlowess` (this package) across a set of representative scenarios.

## Scenarios

| Category | Variants | Description |
| --- | --- | --- |
| **Scalability** | n = 1 000 / 5 000 / 10 000 | Sine wave, fraction 0.1, 3 robustness iterations |
| **Fraction** | 0.05 – 0.67 (6 levels) | Effect of smoothing span, n = 5 000 |
| **Iterations** | 0 – 10 (6 levels) | Effect of robustness iterations on outlier data, n = 5 000 |
| **Financial** | n = 500 / 1 000 / 5 000 | Cumulative-return time series, fraction 0.1 |
| **Scientific** | n = 500 / 1 000 / 5 000 | Damped-oscillator signal, fraction 0.15 |
| **Genomic** | n = 1 000 / 5 000 / 100 000 | Step-function expression data, fraction 0.1 |
| **Pathological** | clustered, high-noise | Edge cases: clustered x-values and high-noise signal |

## Running

```sh
# Build and install rfastlowess to system R (required before benchmarking)
make install

# Run benchmarks
make bench-r                    # stats::lowess only
make bench-rfastlowess-serial
make bench-rfastlowess-parallel

# Generate comparison plot (output/benchmark_comparison.svg)
make compare
```

Output JSON files are written to `output/`.
