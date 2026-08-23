# Merge Strategies

## Overview

Streaming LOWESS processes data in fixed-size chunks with a configurable
overlap. Points inside the overlap zone are fitted twice — once by the
left chunk and once by the right chunk. The `merge_strategy` decides how
those two estimates are combined.

``` text
Chunk A:   [=========|=====]
Chunk B:            [=====|=========]
Overlap:            [=====]
                      ↑
                 merge_strategy
                 applied here
```

| Strategy | Method | Best For |
|----|----|----|
| `"average"` | Simple mean of both estimates | Uniform data density |
| `"take_first"` | Left-chunk estimate only | Left chunk is more accurate |
| `"take_last"` | Right-chunk estimate only | Right chunk is more accurate |
| `"weighted_average"` | Distance-weighted mean (default) | Most situations |

![Merge strategy comparison](../reference/figures/merge_comparison.svg)

Merge strategy comparison

------------------------------------------------------------------------

## Weighted Average (Default)

Blends left- and right-chunk estimates using distance-based weights.
Points closer to the centre of each chunk get higher weight.

**Use when**: General-purpose streaming; most reliable option.

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)
x_chunk <- x[seq_len(50)]
y_chunk <- y[seq_len(50)]

model <- StreamingLowess(
    merge_strategy = "weighted_average",
    chunk_size = 5000,
    overlap = 500
)
result <- process_chunk(model, x_chunk, y_chunk)
```

------------------------------------------------------------------------

## Average

Takes the arithmetic mean of the left-chunk and right-chunk estimates in
the overlap region.

**Use when**: Chunks are large and the overlap region has uniform data
density.

``` r

model <- StreamingLowess(
    merge_strategy = "average",
    chunk_size = 5000,
    overlap = 500
)
result <- process_chunk(model, x_chunk, y_chunk)
```

------------------------------------------------------------------------

## Take First

Uses only the left-chunk estimate in the overlap region.

**Use when**: The left chunk has a larger context window and is expected
to be more accurate at the overlap boundary.

``` r

model <- StreamingLowess(
    merge_strategy = "take_first",
    chunk_size = 5000,
    overlap = 500
)
result <- process_chunk(model, x_chunk, y_chunk)
```

------------------------------------------------------------------------

## Take Last

Uses only the right-chunk estimate in the overlap region.

**Use when**: The right chunk is expected to provide better context
(e.g., more data follows).

``` r

model <- StreamingLowess(
    merge_strategy = "take_last",
    chunk_size = 5000,
    overlap = 500
)
result <- process_chunk(model, x_chunk, y_chunk)
```

------------------------------------------------------------------------

## Choosing Chunk Size and Overlap

| Consideration | Recommendation                                     |
|---------------|----------------------------------------------------|
| Memory limit  | `chunk_size` = largest chunk that fits comfortably |
| Edge accuracy | `overlap` ≥ `fraction × chunk_size`                |
| Speed         | Larger chunks → fewer merges → faster              |

``` r

# Rule of thumb: overlap = fraction * chunk_size
fraction  <- 0.3
chunk_size <- 5000
overlap    <- ceiling(fraction * chunk_size)  # 1500

model <- StreamingLowess(
    fraction       = fraction,
    chunk_size     = chunk_size,
    overlap        = overlap,
    merge_strategy = "weighted_average"
)
```
