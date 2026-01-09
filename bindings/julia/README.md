# fastLowess Julia Bindings

High-performance LOWESS (Locally Weighted Scatterplot Smoothing) bindings for Julia.

## Installation

### Building the Rust Library

First, build the shared library:

```bash
cd bindings/julia
cargo build --release
```

This creates `target/release/libfastlowess_jl.so` (Linux), `libfastlowess_jl.dylib` (macOS), or `fastlowess_jl.dll` (Windows).

### Setting Up Julia

From Julia:

```julia
using Pkg
Pkg.develop(path="path/to/lowess-project/bindings/julia/julia")
```

Or set the library path environment variable:

```bash
export FASTLOWESS_LIB=/path/to/lowess-project/bindings/julia/target/release/libfastlowess_jl.so
```

## Usage

```julia
using fastLowess

# Generate sample data
x = collect(1.0:0.1:10.0)
y = sin.(x) .+ 0.1 .* randn(length(x))

# Basic smoothing
result = smooth(x, y)
println("Smoothed values: ", result.y[1:5])

# With more options
result = smooth(x, y,
    fraction = 0.3,
    iterations = 3,
    return_diagnostics = true
)

if result.diagnostics !== nothing
    println("R² = ", result.diagnostics.r_squared)
    println("RMSE = ", result.diagnostics.rmse)
end

# Streaming mode for large datasets
result = smooth_streaming(x, y, chunk_size=1000)

# Online mode with sliding window
result = smooth_online(x, y, window_capacity=50)
```

## API

### `smooth(x, y; kwargs...)`

Batch LOWESS smoothing - processes the entire dataset at once.

**Keyword Arguments:**

- `fraction`: Smoothing fraction (default: 0.67)
- `iterations`: Robustness iterations (default: 3)
- `weight_function`: Kernel function (default: "tricube")
- `robustness_method`: Robustness method (default: "bisquare")
- `return_diagnostics`: Compute diagnostics (default: false)
- `confidence_intervals`: Confidence level (NaN to disable)
- `parallel`: Enable parallel execution (default: true)

### `smooth_streaming(x, y; kwargs...)`

Streaming LOWESS for large datasets - processes data in chunks.

### `smooth_online(x, y; kwargs...)`

Online LOWESS with sliding window - for real-time data.

## License

MIT OR Apache-2.0
