# Benchmarks

Compares `stats::lowess` (base R) against `rfastlowess` (this package) across a set of representative scenarios, and profiles GPU backend performance.

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

## GPU Benchmarks

### End-to-end GPU fit (`bench-gpu-rust`)

Measures total wall-clock time for a complete GPU LOWESS fit (upload + kernel + download) using the Rust `fastLowess` GPU backend directly. Sine-wave data, fraction 0.3, 3 robustness iterations, 10 timed runs after 2 warm-up runs.

| n | mean | median | min | max |
| ---: | ---: | ---: | ---: | ---: |
| 1 000 | 7.3 ms | 7.1 ms | 5.3 ms | 9.0 ms |
| 5 000 | 6.8 ms | 6.5 ms | 5.3 ms | 9.2 ms |
| 10 000 | 7.1 ms | 6.9 ms | 5.5 ms | 9.1 ms |
| 50 000 | 9.1 ms | 8.9 ms | 8.3 ms | 10.3 ms |
| 100 000 | 15.7 ms | 15.7 ms | 13.6 ms | 17.9 ms |
| 500 000 | 67.0 ms | 66.1 ms | 65.3 ms | 73.2 ms |
| 1 000 000 | 155.3 ms | 154.5 ms | 148.3 ms | 170.6 ms |

The flat ~7 ms cost at small n is dominated by GPU executor initialisation and PCIe round-trip latency, not compute. Compute begins to dominate above n ≈ 50 000.

### Host ↔ device transfer overhead (`bench-gpu-transfer`)

Isolates buffer transfer cost from kernel execution. Upload sends 2 × n f32 (x, y); download retrieves 3 × n f32 (y\_smooth, weights, residuals). 20 timed runs after 5 warm-up runs.

| n | upload | download | round-trip | bandwidth |
| ---: | ---: | ---: | ---: | ---: |
| 1 000 | 0.57 ms | 0.59 ms | 1.16 ms | 0.017 GB/s |
| 5 000 | 0.67 ms | 0.64 ms | 1.30 ms | 0.077 GB/s |
| 10 000 | 0.47 ms | 0.49 ms | 0.96 ms | 0.209 GB/s |
| 50 000 | 0.99 ms | 0.84 ms | 1.83 ms | 0.546 GB/s |
| 100 000 | 0.76 ms | 1.35 ms | 2.11 ms | 0.946 GB/s |
| 500 000 | 3.14 ms | 4.79 ms | 7.93 ms | 1.261 GB/s |
| 1 000 000 | 5.39 ms | 10.31 ms | 15.70 ms | 1.274 GB/s |

Round-trip latency floors at ~0.7 ms regardless of size. Bandwidth saturates near 1.3 GB/s at n ≥ 500 000. Transfer is ≈10–16% of total GPU fit time at n ≥ 500 000, so kernel execution dominates at scale as expected.

## Running

```sh
# Build and install rfastlowess to system R (required before benchmarking)
make install

# Run benchmarks
make bench-r                    # stats::lowess only
make bench-rfastlowess-serial
make bench-rfastlowess-parallel

# GPU benchmarks
make bench-gpu-rust             # end-to-end GPU fit → output/rust_benchmark_gpu.json
make bench-gpu-transfer         # transfer overhead only → output/gpu_transfer.json

# Generate comparison plot (output/benchmark_comparison.svg)
make compare
```

Output JSON files are written to `output/`.
