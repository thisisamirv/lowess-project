# Validation

Validates `fastLowess` (Rust) output against R's `stats::lowess` as the reference implementation across scenarios covering a wide range of inputs and parameter combinations.

## Scenarios

| # | Name | n | Fraction | Iterations | Notes |
| --- | --- | --- | --- | --- | --- |
| 01 | Tiny Linear | 10 | 0.8 | 0 | Minimal dataset |
| 03 | Sine Standard | 100 | 0.3 | 0 | Noisy sine wave |
| 04 | Sine Robust | 100 | 0.3 | 4 | Sine with 5% outliers, bisquare reweighting |
| 06 | Large Scale | 500 | 0.1 | 0 | Narrow bandwidth, 500 points |
| 07 | High Smoothness | 100 | 0.9 | 0 | Very wide bandwidth |
| 08 | Low Smoothness | 100 | 0.05 | 0 | Very narrow bandwidth, direct surface |
| 10 | Constant | 50 | 0.5 | 0 | Constant y signal |
| 11 | Step Function | 100 | 0.4 | 0 | Discontinuous step signal |
| 12 | End-effects Left | 50 | 0.3 | 0 | Left boundary behavior |
| 13 | End-effects Right | 50 | 0.3 | 0 | Right boundary behavior |
| 14 | Sparse Data | 20 | 0.6 | 0 | Wide x-range, only 20 points |
| 15 | Dense Data | 1000 | 0.01 | 0 | Very narrow bandwidth, 1000 points |
| 18 | Iter 2 Check | 100 | 0.4 | 2 | Two robustness iterations |
| 19 | Interpolate Exact | 50 | 0.5 | 0 | Interpolation surface check |
| 20 | Zero Variance | 10 | 0.5 | 0 | Constant y, minimal n |

## Running

```sh
# Generate R reference outputs (writes output/r/)
make r-validate

# Run fastLowess validation (writes output/fastLowess/)
make fastlowess-validate

# Run fastLowess visual output (writes output/fastLowess/)
make fastlowess-visual

# Compare R and fastLowess outputs
make compare

# Generate plots
make plot
```

Output JSON files are written to `output/r/` and `output/fastLowess/`.
