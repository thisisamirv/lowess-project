<!-- markdownlint-disable MD033 -->
# Custom Weights

Per-observation weights that encode data quality directly into the LOWESS fit.

## How Custom Weights Work

Standard LOWESS assigns equal prior trust to all observations. Custom weights
let you override this assumption point by point — before any distance or
robustness weighting is applied.

The effective weight of observation $j$ in a local fit centred at $x_i$ is:

$$w_{ij} = \text{custom\_weights}[j] \times K\!\left(\frac{d_{ij}}{h_i}\right) \times r_j$$

where $K$ is the distance kernel, $h_i$ is the local bandwidth, and $r_j$ is
the robustness weight from the current iteration.

!!! note "Batch adapter only"
    `custom_weights` applies in **Batch** mode. It is silently ignored in
    Streaming and Online adapters.

---

## When to Use Custom Weights

| Situation | Recommended weight |
| --- | --- |
| Point known to be erroneous | `0.0` — fully excluded |
| Unreliable sensor / low precision | `0.1 – 0.5` |
| Standard observation | `1.0` (default) |
| Carefully calibrated measurement | `> 1.0` |
| Measurement uncertainty $\sigma_i$ | $1 / \sigma_i^2$ |

### Custom Weights vs. Robustness Iterations

Both mechanisms handle unreliable data, but they serve different purposes:

| | Custom Weights | Robustness Iterations |
| --- | --- | --- |
| **When known** | Before fitting | Computed from residuals |
| **Knowledge required** | Prior knowledge of quality | None — data-driven |
| **Effect** | Fixed throughout fit | Adapts each iteration |
| **Use case** | Known bad sensors, calibration | Unknown outlier contamination |

They compose: you can use both simultaneously. Custom weights suppress
*a priori* bad points; robustness iterations then handle any *residual*
outliers that remain.

---

## Basic Usage

### Suppress a Known Outlier

Set the weight to `0` at the bad point — it is excluded from every local fit
that would otherwise include it.

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    std::vector<double> xw(10), yw(10);
    for (std::size_t i = 0; i < 10; ++i) {
        xw[i] = static_cast<double>(i);
        yw[i] = xw[i] * 2.0;
    }
    yw[5] = 100.0; // spike

    fastlowess::LowessOptions opts;
    opts.fraction = 0.5;
    opts.iterations = 0;

    std::vector<double> weights(10, 1.0);
    weights[5] = 0.0; // exclude the spike

    auto result = fastlowess::Lowess(opts).fit(xw, yw, weights).value();

    return 0;
}
```

---

### Emphasize Important Points

Assign high weights to measurements you trust most — calibration standards,
reference instruments, or low-noise observations.

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    std::vector<std::size_t> calibration_indices = {5, 20, 40, 60, 80};
    std::vector<double> weights(x.size(), 1.0);
    for (std::size_t idx : calibration_indices) weights[idx] = 10.0;

    fastlowess::LowessOptions opts;
    opts.fraction = 0.5;
    auto result = fastlowess::Lowess(opts).fit(x, y, weights).value();

    return 0;
}
```

---

### Propagate Measurement Uncertainty

If each observation has a known standard deviation $\sigma_i$, set
$w_i = 1 / \sigma_i^2$ to give the fit information-theoretically optimal
weighting.

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    std::vector<double> sigma(n, 0.1);
    for (int i = 0; i < n; ++i) sigma[i] = 0.1 + (i % 4) * 0.1;
    std::vector<double> weights(sigma.size());
    for (std::size_t i = 0; i < sigma.size(); ++i) weights[i] = 1.0 / (sigma[i] * sigma[i]);

    fastlowess::LowessOptions opts;
    opts.fraction = 0.5;
    auto result = fastlowess::Lowess(opts).fit(x, y, weights).value();

    return 0;
}
```

---

## Combined with Robustness Iterations

Custom weights and robustness iterations compose naturally: use custom weights
for *known* bad points and robustness for *unknown* contamination.

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    std::vector<double> weights(n, 1.0);
    weights[3] = 0.0;

    fastlowess::LowessOptions opts;
    opts.fraction = 0.4;
    opts.iterations = 3;

    auto result = fastlowess::Lowess(opts).fit(x, y, weights).value();

    return 0;
}
```

---

## Validation Rules

| Rule | Effect |
| --- | --- |
| Length must equal `n` | Error at fit time if mismatched |
| All values must be ≥ 0 | Negative weights are rejected |
| All-zero weight vector | Error: no points remain for any local fit |
| Uniform weights (`1.0` everywhere) | Identical result to omitting weights |

!!! warning "Zero-weight windows"
    If a local neighbourhood contains only zero-weight points, the fit at
    that centre point falls back to the behaviour specified by
    `zero_weight_fallback` (default: `"use_local_mean"`).

---

## See Also

- [Robustness](robustness.md) — adaptive outlier downweighting via IRLS
- [Parameters](parameters.md#custom_weights) — full parameter reference
