# Julia Examples

Complete Julia examples demonstrating fastlowess.jl with native Julia integration.

## Batch Smoothing

Process complete datasets with confidence intervals and diagnostics.

```julia
--8<-- "../../examples/julia/batch_smoothing.jl"
```

[:material-download: Download batch_smoothing.jl](https://github.com/thisisamirv/lowess-project/blob/main/examples/julia/batch_smoothing.jl)

---

## Streaming Smoothing

Process large datasets in memory-efficient chunks.

```julia
--8<-- "../../examples/julia/streaming_smoothing.jl"
```

[:material-download: Download streaming_smoothing.jl](https://github.com/thisisamirv/lowess-project/blob/main/examples/julia/streaming_smoothing.jl)

---

## Online Smoothing

Real-time smoothing with sliding window for streaming data.

```julia
--8<-- "../../examples/julia/online_smoothing.jl"
```

[:material-download: Download online_smoothing.jl](https://github.com/thisisamirv/lowess-project/blob/main/examples/julia/online_smoothing.jl)

---

## Installation

```julia
using Pkg
Pkg.develop(path="bindings/julia/julia")
```

## Running the Examples

```bash
julia --project=bindings/julia/julia ../../examples/julia/batch_smoothing.jl
julia --project=bindings/julia/julia ../../examples/julia/streaming_smoothing.jl
julia --project=bindings/julia/julia ../../examples/julia/online_smoothing.jl
```

## Quick Start

```julia
using fastlowess

# Generate sample data
x = collect(0.0:0.1:10.0)
y = sin.(x) .+ 0.3 .* randn(length(x))

# Basic smoothing
result = smooth(x, y; fraction=0.3)
println("Smoothed values: ", result.y[1:5])

# With options
result = smooth(x, y;
    fraction=0.3,
    iterations=3,
    confidence_intervals=0.95,
    return_diagnostics=true
)

println("R²: ", result.diagnostics.r_squared)

# Access confidence intervals
lower = result.confidence_lower
upper = result.confidence_upper
```

## Features

The Julia bindings provide:

- **Native Julia types** - Uses Julia arrays directly
- **C FFI integration** - Efficient bindings via `ccall`
- **Multiple dispatch** - Works with Float32 and Float64
- **Full parameter access** - All LOWESS options available
