# Alternative Software

`FastLOWESS.jl` is presented throughout this package's documentation as a faster, more feature-rich alternative to [`Loess.jl`](https://github.com/JuliaStats/Loess.jl), the most widely used local regression package in the Julia ecosystem. This page is for readers who already use `Loess.jl` and want to know: *how does `FastLOWESS.jl` compare, and can I get the same numbers out of it?*

In short:

- `Loess.jl` implements the general LOESS method (Cleveland & Devlin, 1988), which supports quadratic (`degree=2`, its default) as well as linear (`degree=1`) local fits and uses a k-d tree for neighbor lookups. `FastLOWESS.jl` implements the earlier, simpler LOWESS method (Cleveland, 1979) — local *linear* fits only, with iterative robustness reweighting built in — the same algorithm R's `stats::lowess()` and Python's `statsmodels.lowess()` implement. See [How the algorithms differ](#how-the-algorithms-differ) below.
- Because of this, even with `Loess.jl`'s `degree` set to `1` to match, the two packages do **not** reproduce each other's output to floating-point precision the way `FastLOWESS.jl` reproduces `stats::lowess()`/`statsmodels.lowess()` (see the R and Python packages' own "Alternative Software" pages) — they're independent implementations that agree only approximately. See [Comparing the two packages](#comparing-the-two-packages).
- `FastLOWESS.jl` supports a number of features `Loess.jl` doesn't have — see [What this package adds](#what-this-package-adds).

> **Note:** For a **LOESS** implementation, use [`FastLOESS`](https://platform.juliahub.com/ui/Packages/General/FastLOESS).

---

## How the algorithms differ

| Aspect | `FastLOWESS.jl` | `Loess.jl` |
| --- | --- | --- |
| Method | LOWESS (Cleveland, 1979) | LOESS (Cleveland & Devlin, 1988) |
| Local fit degree | Linear only | Linear or quadratic (`degree`, default `2`) |
| Robustness reweighting | Built in (`iterations`) | None |
| Neighbor lookup | Sorted-array window | K-D tree (`cell` bucket parameter) |
| Boundary handling | 4 explicit policies | Implicit (whatever neighbors the tree finds) |
| Dense-data shortcut | `delta` (skip + interpolate nearby points) | `cell` (interpolation nodes added to the K-D tree) |

Both packages call their smoothing parameter something different: `Loess.jl`'s `span` is the same idea as `FastLOWESS.jl`'s `fraction` (the proportion of points used in each local neighbourhood).

---

## Comparing the two packages

Setting `Loess.jl`'s `degree=1` (to match `FastLOWESS.jl`'s linear-only fits) and `FastLOWESS.jl`'s `iterations=0`, `boundary_policy="noboundary"` (to remove the two features `Loess.jl` doesn't have) gets the two packages into their closest possible agreement — but the fits still differ, since they're independent implementations with different windowing and edge-handling internals:

```@example alt-software-compare
using FastLOWESS
using Loess
using Random

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=60))
y = sin.(x) .+ randn(rng, 60) .* 0.2

model = Lowess(; fraction=2 / 3, iterations=0, boundary_policy="noboundary")
result = fit(model, x, y)

lo = Loess.loess(x, y; span=2 / 3, degree=1)
lo_y = Loess.predict(lo, x)

println("Max abs difference: ", maximum(abs.(result.y .- lo_y)))
```

This is a fundamentally different situation from `FastLOWESS.jl`'s R and Python packages, both of which reproduce their respective reference implementation (`stats::lowess()`/`statsmodels.lowess()`) to within floating-point tolerance (~1e-10) once matching options are set — those are ports of the *same* Fortran/C source, whereas `Loess.jl` is an independent implementation of a different (more general) method.

---

## What this package adds

`Loess.jl`'s API is intentionally minimal (`loess()` + `predict()`); it doesn't support:

| Feature | `FastLOWESS.jl` | `Loess.jl` |
| --- | :---: | :---: |
| Robustness weighting | 3 methods | none |
| Kernel functions | 7 options | tricube only |
| Scale estimation | MAD, MAR, mean | n/a (no robustness) |
| Boundary padding | 4 policies | none |
| Confidence / prediction intervals | yes | no |
| Cross-validation for `fraction` | K-fold, LOOCV | no |
| Streaming / online modes | yes | no |
| Custom per-observation weights | yes | no |
| Parallel / GPU execution | yes | no |

See the [Concepts](../introduction/concepts.md) page for an overview of these, or the [Benchmarks](../benchmarks.md) page for performance comparisons.
