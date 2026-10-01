# Validation

## Numerical Tests

R randomized properties and fixed reference cases live in [`property_tests/`](property_tests/). Golden-output tests and their committed fixtures live in [`fixture_tests/`](fixture_tests/). Run both suites, with lint, from the repository root using `make validate`.

The property suite compares direct local-linear fits against Locfit for the shared nearest-neighbor span, tricube/Epanechnikov/biweight/triangular kernels, and positive observation weights. It checks zero and Locfit's default three robustness passes (MAR-scaled bisquare), with degree fixed at one. Locfit's Gaussian and rectangular bandwidth conventions differ, so those kernels are not treated as equivalent; boundary padding is validated separately. `make validate` installs `locfit` and `quickcheck` into the local validation library when needed; neither is a package dependency.

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
