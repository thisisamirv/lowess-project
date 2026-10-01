# Validation

## Numerical Tests

R randomized properties and fixed reference cases live in [`property_tests/`](property_tests/). Golden-output tests and their committed fixtures live in [`fixture_tests/`](fixture_tests/). Run both suites, with lint, from the repository root using `make validate`.

## Visual Validation

[`visual_validation/`](visual_validation/) generates CSV data for explanatory plots. These visual comparisons help inspect kernels, boundary policies, intervals, adapters, and other behaviors; they are not numerical correctness oracles.

From the repository root:

```sh
make -C validation visual
make -C validation plot
```

Generated CSVs and SVGs are kept in [`visual_validation/output/`](visual_validation/output/).

## Reference Sources

[`reference/`](reference/) contains self-contained source references for R's current C LOWESS engine and Cleveland's original 1985 Fortran implementation, including provenance, build instructions, and behavioral differences.
