# StreamingLowess — Julia API Reference

See also: [FastLOWESS Julia API Reference](julia.md)

## Struct

### `StreamingLowess`

The `StreamingLowess` struct processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```julia
using FastLOWESS

stream = StreamingLowess(chunk_size=50, overlap=10)
```

**Methods:**

```julia
using FastLOWESS

x = collect(range(0, 2π, length=100))
y = sin.(x) .+ 0.1

stream = StreamingLowess(fraction=0.5, chunk_size=50, overlap=10)
partial_result = process_chunk(stream, x[1:50], y[1:50])
println(partial_result.fraction_used)
# 0.5
```

* Processes a chunk of data. Returns partial results.

```julia
using FastLOWESS

x = collect(range(0, 2π, length=100))
y = sin.(x) .+ 0.1

stream = StreamingLowess(fraction=0.3, chunk_size=50, overlap=10)
process_chunk(stream, x[1:50], y[1:50])
process_chunk(stream, x[51:end], y[51:end])
final_result = finalize(stream)
println(final_result.fraction_used)
# 0.3
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
