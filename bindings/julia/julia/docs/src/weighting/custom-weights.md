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

```@example custom-weights
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

x = collect(1.0:10.0)
y = x .* 2.0
y[6] = 100.0               # spike at index 6 (1-indexed)

weights = ones(10)
weights[6] = 0.0           # exclude the spike

model = Lowess(fraction = 0.5, iterations = 0)
result = fit(model, x, y; custom_weights = weights)
println("First smoothed value (outlier excluded with zero weight): ", result.y[1])
```

---

### Emphasize Important Points

Assign high weights to measurements you trust most — calibration standards,
reference instruments, or low-noise observations.

```@example custom-weights
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3
calibration_indices = [5, 20, 40, 60, 80]

weights = ones(length(x))
weights[calibration_indices] .= 10.0   # trust calibration 10× more

model = Lowess(fraction = 0.5)
result = fit(model, x, y; custom_weights = weights)
println("First smoothed value (calibration points upweighted 10×): ", result.y[1])
```

---

### Propagate Measurement Uncertainty

If each observation has a known standard deviation $\sigma_i$, set
$w_i = 1 / \sigma_i^2$ to give the fit information-theoretically optimal
weighting.

```@example custom-weights
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3
sigma = rand(rng, 100) .* 0.4 .+ 0.1

weights = 1.0 ./ sigma .^ 2
model = Lowess(fraction = 0.5)
result = fit(model, x, y; custom_weights = weights)
println("First smoothed value (inverse-variance weights): ", result.y[1])
```

---

## Combined with Robustness Iterations

Custom weights and robustness iterations compose naturally: use custom weights
for *known* bad points and robustness for *unknown* contamination.

```@example custom-weights
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

x = collect(0.0:19.0)
y = x .* 1.5
y[4]  = -50.0   # known bad (1-indexed)
y[13] = 80.0    # unknown outlier (1-indexed)

weights = ones(20)
weights[4] = 0.0

model = Lowess(fraction = 0.4, iterations = 3)
result = fit(model, x, y; custom_weights = weights)
println("First smoothed value (custom weights + robust fitting): ", result.y[1])
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
- [API Reference](../api.md) — full parameter reference
