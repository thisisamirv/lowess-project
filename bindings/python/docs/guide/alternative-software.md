# Alternative Software

`fastlowess` is presented throughout this package's documentation as a faster, more feature-rich alternative to Python's `statsmodels.nonparametric.smoothers_lowess.lowess()`. This page is for readers who already use `statsmodels.lowess()` and want to know: *can I get the exact same numbers out of `fastlowess`, and if not, why not?*

In short:

- With two options set (`boundary_policy="noboundary"` and `scaling_method="mar"`), `fastlowess` reproduces `statsmodels.lowess()` to within floating-point tolerance (~1e-10) — see [Reproducing `statsmodels.lowess()`](#reproducing-statsmodelslowess) below.
- Outside of those two options, `fastlowess`'s *defaults* intentionally differ from `statsmodels` (see [Why the defaults differ](#why-the-defaults-differ)), and it supports a number of features `statsmodels` doesn't have (see [What this package adds](#what-this-package-adds)).

---

## Reproducing `statsmodels.lowess()`

`statsmodels.nonparametric.smoothers_lowess.lowess()` and `fastlowess`'s `Lowess` implement the same underlying algorithm (both are ports of Cleveland's original Fortran `lowess`), but two defaults differ:

| Option | `statsmodels.lowess()` | `Lowess` default |
| --- | --- | --- |
| Boundary padding | None (`"noboundary"`) | `"extend"` |
| Residual scaling | MAR: `median(\|r\|)` | `"mad"`: `median(\|r - median(r)\|)` |
| `frac` / `fraction` | `2 / 3` | `0.67` |
| `it` / `iterations` | `3` | `3` |
| `delta` | `0.0` | auto (`0.01 * (x.max() - x.min())`) |

Setting the first two options to match `statsmodels`, `Lowess` reproduces `statsmodels.lowess()` exactly (up to floating-point rounding):

:::{jupyter-execute}
import fastlowess as fl
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 60)
y = np.sin(x) + rng.normal(0, 0.2, 60)

reference = sm.nonparametric.lowess(y, x, frac=2 / 3, it=3, delta=0.0)

model = fl.Lowess(
    fraction=2 / 3,
    iterations=3,
    boundary_policy="noboundary",
    scaling_method="mar",
    delta=0.0,
)
result = model.fit(x, y)

print("Max abs difference:", np.max(np.abs(result.y - reference[:, 1])))
:::

If your `x` is not already sorted, add `outputs=["sorted"]` to get results ordered the same way `statsmodels.lowess()` returns them (ascending by `x`):

:::{jupyter-execute}
import fastlowess as fl
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(1)
x = np.linspace(0, 2 * np.pi, 60)
x_unsorted = rng.permutation(x)
y_unsorted = np.sin(x_unsorted) + rng.normal(0, 0.2, 60)

reference = sm.nonparametric.lowess(y_unsorted, x_unsorted, frac=2 / 3, it=3, delta=0.0)

model = fl.Lowess(
    fraction=2 / 3,
    iterations=3,
    boundary_policy="noboundary",
    scaling_method="mar",
    delta=0.0,
    outputs=["sorted"],
)
result = model.fit(x_unsorted, y_unsorted)

print("x is sorted ascending:", np.all(np.diff(result.x) >= 0))
print("Max abs difference:", np.max(np.abs(result.y - reference[:, 1])))
:::

Without `outputs=["sorted"]`, `Lowess` still fits on sorted `x` internally (as `statsmodels.lowess()` requires) but returns values reordered back to match your original input order — useful when you need the fit aligned with other arrays/columns in your own data, but not what you want for a value-by-value comparison against `statsmodels.lowess()`'s own (always sorted) output.

---

## Why the defaults differ

`fastlowess`'s *defaults* (as opposed to what it's capable of reproducing) deliberately depart from `statsmodels.lowess()` in two places:

**Boundary padding** (default `boundary_policy="extend"` vs. no padding). Without padding, the local neighbourhood at the first and last few points is one-sided, which biases the fit toward the interior and increases variance right at the edges. `"extend"` (and the other padding policies — see the [Boundary](../advanced/boundary.md) guide) mitigate this at the cost of no longer being a direct reproduction of Cleveland's original algorithm. `"noboundary"` is kept as an explicit option specifically so reference-matching remains possible.

**Residual scaling** (default `scaling_method="mad"` vs. MAR). MAD (`median(|r - median(r)|)`) centers residuals at their median before taking the median absolute value, which is a breakdown-point-optimal scale estimator; MAR (`median(|r|)`, what `statsmodels` uses) does not center first, so it can be biased when residuals are systematically skewed. See the [Scaling](../weighting/scaling.md) guide for the full comparison, including `"mean"`.

---

## What this package adds

`statsmodels.lowess()` doesn't support:

| Feature | `fastlowess` | `statsmodels.lowess()` |
| --- | :---: | :---: |
| Kernel functions | 7 options | Tricube only |
| Robustness weighting | 3 options | Bisquare only |
| Scale estimation | MAD, MAR, mean | MAR only |
| Boundary padding | 4 policies | none |
| Confidence / prediction intervals | yes | no |
| Cross-validation for `fraction` | K-fold, LOOCV | no |
| Streaming / online modes | yes | no |
| Custom per-observation weights | yes | no |
| Parallel / GPU execution | yes | no |

See the [Concepts](../introduction/concepts.md) page for an overview of these, or the [Benchmarks](../benchmarks.md) page for performance comparisons.
