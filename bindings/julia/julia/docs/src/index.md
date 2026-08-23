# FastLOWESS.jl

High-performance LOWESS (Locally Weighted Scatterplot Smoothing) for Julia,
backed by a Rust library.

See the [main documentation](https://lowess.readthedocs.io/) and the
[GitHub repository](https://github.com/thisisamirv/lowess-project) for full details.

## Quick Start

```julia
using FastLOWESS

x = collect(1.0:0.1:10.0)
y = sin.(x) .+ 0.1 .* randn(length(x))

result = fit(Lowess(fraction = 0.3), x, y)
println(result.y)
```

## Installation

```julia
using Pkg
Pkg.add("FastLOWESS")
```
