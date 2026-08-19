# StreamingLowess — Julia API Reference

See also: [FastLOWESS Julia API Reference](julia.md)

## Struct

### `StreamingLowess`

The `StreamingLowess` struct processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```julia
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

stream = StreamingLowess()
```

* `kwargs`: Keyword arguments corresponding to `LowessOptions` and `StreamingOptions` fields.

**Methods:**

```julia
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

stream = StreamingLowess()
partial_result = process_chunk(stream, x, y)
```

* Processes a chunk of data. Returns partial results.

```julia
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

stream = StreamingLowess()
process_chunk(stream, x, y)
final_result = finalize(stream)
```

* Finalizes the smoothing process and returns any remaining buffered results.

## Result Structure

### `LowessResult`

Returned by `process_chunk()` and `finalize()`.

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Vector{Float64}` | Sorted x values |
| `y` | `Vector{Float64}` | Smoothed y values |
| `fraction_used` | `Float64` | Fraction used |
| `iterations_used` | `Int` | Robustness iterations actually performed (-1 = N/A) |
| `residuals` | `Union{Vector{Float64}, Nothing}` | Residuals (if `return_residuals`) |
| `robustness_weights` | `Union{Vector{Float64}, Nothing}` | Robustness weights (if `return_robustness_weights`) |
| `diagnostics` | `Union{Diagnostics, Nothing}` | Fit metrics (if `return_diagnostics`) |
| `dimensions` | `Int` | Number of predictor dimensions |

See [julia.md](julia.md) for the full `LowessResult` field reference.

## Options Structure

### `StreamingOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `Int` | `5000` | Data chunk size |
| `overlap` | `Int` | `500` | Overlap between chunks |
| `merge_strategy` | `String` | `"weighted_average"` | Strategy for blending overlap regions |

## Options

### merge_strategy

*See: [Merge Strategies](../user-guide/merge.md)*

* `"weighted_average"` (default; alias: `"weighted"`)
* `"average"` (alias: `"mean"`)
* `"take_first"` (alias: `"first"`)
* `"take_last"` (alias: `"last"`)
