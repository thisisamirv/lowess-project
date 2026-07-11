<!-- markdownlint-disable MD024 MD033 MD046 -->
# Merge Strategies

How overlapping chunk boundaries are reconciled in Streaming mode.

## Overview

Streaming LOWESS processes data in fixed-size chunks with a configurable overlap. Points inside the overlap zone are fitted twice — once by the left chunk and once by the right chunk. The `merge_strategy` decides how those two estimates are combined into a single output value.

```text
Chunk A:   [=========|=====]
Chunk B:            [=====|=========]
Overlap:            [=====]
                      ↑
                 merge_strategy
                 applied here
```

| Strategy | Method | Robustness | Speed |
| --- | --- | --- | --- |
| `"average"` | Simple mean of both estimates | Low | Fastest |
| `"take_first"` | Left-chunk estimate only | Low | Fastest |
| `"take_last"` | Right-chunk estimate only | Low | Fastest |
| `"weighted_average"` | Distance-weighted mean | High | Moderate |

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

---

## Average

Takes the arithmetic mean of the left-chunk and right-chunk estimates in the overlap region. Fast and sufficient when both chunks have similar smoothing quality.

**Use when**: Chunks are large and the overlap region has uniform data density.

=== "R"
    ```r
    library(rfastlowess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)
    x_chunk <- x[seq_len(50)]
    y_chunk <- y[seq_len(50)]

    model <- StreamingLowess(
        merge_strategy = "average",
        chunk_size = 5000,
        overlap = 500
    )
    result <- model$process_chunk(x_chunk, y_chunk)
    ```

=== "Python"
    ```python
    from fastlowess import StreamingLowess
    import numpy as np

    rng = np.random.default_rng(42)
    x_chunk = np.linspace(0, np.pi, 50)
    y_chunk = np.sin(x_chunk) + rng.normal(0, 0.1, 50)

    model = StreamingLowess(merge_strategy="average", chunk_size=5000, overlap=500)
    result = model.process_chunk(x_chunk, y_chunk)
    ```

=== "Rust"
    ```rust
    use fastLowess::prelude::*;

    fn main() -> Result<(), LowessError> {
        let model = StreamingLowess::new()
            .merge_strategy("average")
            .chunk_size(5000)
            .overlap(500)
            .build()?;

        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using FastLOWESS
    using Random, Statistics

    rng = MersenneTwister(42)
    x = collect(range(0, 2π, length=100))
    y = sin.(x) .+ randn(rng, 100) .* 0.3
    x_chunk = x[1:50]; y_chunk = y[1:50]

    model = StreamingLowess(; merge_strategy="average", chunk_size=5000, overlap=500)
    result = process_chunk(model, x_chunk, y_chunk)
    ```

=== "Node.js"
    ```javascript
    const { StreamingLowess } = require('fastlowess');

    const n = 50;
    const xChunk = Float64Array.from({ length: n }, (_, i) => i * Math.PI / (n - 1));
    const yChunk = Float64Array.from(xChunk, xi => Math.sin(xi));

    const processor = new StreamingLowess(
        {},
        { merge_strategy: "average", chunk_size: 5000, overlap: 500 }
    );
    const result = processor.process_chunk(xChunk, yChunk);
    ```

=== "WebAssembly"
    ```javascript
    import init, { StreamingLowess } from 'fastlowess-wasm';
    await init();

    const n = 100;
    const xChunk = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const yChunk = Float64Array.from(xChunk, xi => Math.sin(xi) + 0.1);

    const processor = new StreamingLowess(
        {},
        { merge_strategy: "average", chunk_size: 5000, overlap: 500 }
    );
    const result = processor.process_chunk(xChunk, yChunk);
    ```

=== "C++"
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

        fastlowess::StreamingOptions opts;
        opts.merge_strategy = "average";
        opts.chunk_size = 5000;
        opts.overlap = 500;
        fastlowess::StreamingLowess stream(opts);
        (void)stream.process_chunk(x, y);
        auto result = stream.finalize().value();

        return 0;
    }
    ```

---

## Take First

Keeps only the left-chunk estimate in the overlap zone and discards the right-chunk estimate. Produces a definitive, non-revised output as soon as the right boundary of each chunk is reached.

**Use when**: You need final output values immediately after each chunk (no look-ahead revision); left-chunk data quality is higher.

=== "R"
    ```r
    library(rfastlowess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    model <- StreamingLowess(merge_strategy = "take_first")
    ```

=== "Python"
    ```python
    from fastlowess import StreamingLowess
    model = StreamingLowess(merge_strategy="take_first")
    ```

=== "Rust"
    ```rust
    use fastLowess::prelude::*;

    fn main() -> Result<(), LowessError> {
        let mut processor = StreamingLowess::new()
            .merge_strategy("take_first")
            .build()?;

        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using FastLOWESS
    using Random, Statistics

    rng = MersenneTwister(42)
    x = collect(range(0, 2π, length=100))
    y = sin.(x) .+ randn(rng, 100) .* 0.3

    model = StreamingLowess(; merge_strategy="take_first")
    ```

=== "Node.js"
    ```javascript
    const { StreamingLowess } = require('fastlowess');

    const processor = new StreamingLowess(
        {},
        { merge_strategy: "take_first" }
    );
    ```

=== "WebAssembly"
    ```javascript
    import init, { StreamingLowess } from 'fastlowess-wasm';
    await init();

    const stream = new StreamingLowess({}, { merge_strategy: 'take_first' });
    ```

=== "C++"
    ```cpp
    #include <fastlowess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        fastlowess::StreamingOptions s_opts;
        s_opts.merge_strategy = "take_first";
        fastlowess::StreamingLowess model(s_opts);

        return 0;
    }
    ```

---

## Take Last

Keeps only the right-chunk estimate in the overlap zone. The right chunk sees more of the surrounding data, so its fit can be more accurate near the left boundary of the new chunk.

**Use when**: Right-chunk context improves overlap quality; you are post-processing complete data rather than streaming live.

=== "R"
    ```r
    library(rfastlowess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    model <- StreamingLowess(merge_strategy = "take_last")
    ```

=== "Python"
    ```python
    from fastlowess import StreamingLowess
    model = StreamingLowess(merge_strategy="take_last")
    ```

=== "Rust"
    ```rust
    use fastLowess::prelude::*;

    fn main() -> Result<(), LowessError> {
        let mut processor = StreamingLowess::new()
            .merge_strategy("take_last")
            .build()?;

        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using FastLOWESS
    using Random, Statistics

    rng = MersenneTwister(42)
    x = collect(range(0, 2π, length=100))
    y = sin.(x) .+ randn(rng, 100) .* 0.3

    model = StreamingLowess(; merge_strategy="take_last")
    ```

=== "Node.js"
    ```javascript
    const { StreamingLowess } = require('fastlowess');

    const processor = new StreamingLowess(
        {},
        { merge_strategy: "take_last" }
    );
    ```

=== "WebAssembly"
    ```javascript
    import init, { StreamingLowess } from 'fastlowess-wasm';
    await init();

    const stream = new StreamingLowess({}, { merge_strategy: 'take_last' });
    ```

=== "C++"
    ```cpp
    #include <fastlowess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        fastlowess::StreamingOptions s_opts;
        s_opts.merge_strategy = "take_last";
        fastlowess::StreamingLowess model(s_opts);

        return 0;
    }
    ```

---

## Weighted Average

Assigns each overlap point a weight proportional to its proximity to the centre of its respective chunk: points near the left-chunk centre get higher left weight; points near the right-chunk centre get higher right weight. This produces the smoothest transition across chunk boundaries.

$$\hat{y} = \frac{w_L \hat{y}_L + w_R \hat{y}_R}{w_L + w_R}$$

where $w_L$ and $w_R$ are linear distance weights from the chunk centres.

**Use when**: Minimising boundary artefacts is more important than speed; moderate overlap (10–20 % of chunk size).

=== "R"
    ```r
    library(rfastlowess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    model <- StreamingLowess(
        merge_strategy = "weighted_average",
        chunk_size = 5000,
        overlap = 500
    )
    ```

=== "Python"
    ```python
    from fastlowess import StreamingLowess
    model = StreamingLowess(
        merge_strategy="weighted_average",
        chunk_size=5000,
        overlap=500
    )
    ```

=== "Rust"
    ```rust
    use fastLowess::prelude::*;

    fn main() -> Result<(), LowessError> {
        let model = StreamingLowess::new()
        .merge_strategy("weighted_average")
        .chunk_size(5000)
        .overlap(500)
        .build()?;

        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using FastLOWESS
    using Random, Statistics

    rng = MersenneTwister(42)
    x = collect(range(0, 2π, length=100))
    y = sin.(x) .+ randn(rng, 100) .* 0.3

    model = StreamingLowess(;
        merge_strategy="weighted_average",
        chunk_size=5000,
        overlap=500
    )
    ```

=== "Node.js"
    ```javascript
    const { StreamingLowess } = require('fastlowess');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new StreamingLowess(
        {},
        { merge_strategy: "weighted_average", chunk_size: 5000, overlap: 500 }
    );
    ```

=== "WebAssembly"
    ```javascript
    import init, { StreamingLowess } from 'fastlowess-wasm';
    await init();

    const processor = new StreamingLowess(
        {},
        { merge_strategy: "weighted_average", chunk_size: 5000, overlap: 500 }
    );
    ```

=== "C++"
    ```cpp
    #include <fastlowess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        fastlowess::StreamingOptions s_opts;
        s_opts.merge_strategy = "weighted_average";
        fastlowess::StreamingLowess model(s_opts);

        return 0;
    }
    ```

---

## Choosing a Strategy

| Situation | Recommended Strategy |
| --- | --- |
| General purpose | `"weighted_average"` |
| Maximum throughput | `"average"` |
| Immediate finalised output | `"take_first"` |
| Post-processing, right context better | `"take_last"` |
| Minimising boundary artefacts | `"weighted_average"` |

!!! tip "Overlap size matters"
    A larger overlap gives the merge strategy more room to blend, reducing boundary artefacts regardless of the strategy chosen. A good starting point is 10 % of `chunk_size`.
