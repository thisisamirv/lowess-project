<!-- markdownlint-disable MD033 -->
# Examples

Complete working examples demonstrating fastlowess capabilities across all supported languages.

## Language-Specific Examples

<div class="grid cards" markdown>

- :fontawesome-brands-r-project: **[R](r.md)**

    ---

    Native R package with formula interface, ggplot2 visualization, and comprehensive statistics.

- :fontawesome-brands-python: **[Python](python.md)**

    ---

    PyO3-based bindings with NumPy integration, matplotlib visualization, and full API access.

- :fontawesome-brands-rust: **[Rust](rust.md)**

    ---

    Native Rust with builder pattern, type safety, and zero-cost abstractions.

- :simple-julia: **[Julia](julia.md)**

    ---

    Native Julia package with C FFI for high-performance scientific computing.

- :fontawesome-brands-node-js: **[Node.js](nodejs.md)**

    ---

    Native N-API bindings for server-side JavaScript with TypeScript support.

- :simple-webassembly: **[WebAssembly](wasm.md)**

    ---

    Browser-ready WASM package for client-side smoothing applications.

- :material-language-cpp: **[C++](cpp.md)**

    ---

    Modern C++ wrapper with RAII, STL containers, and exception-based error handling.

</div>

## Example Categories

Each language includes three complete examples:

| Example                 | Description                      | Key Features                     |
| ----------------------- | -------------------------------- | -------------------------------- |
| **Batch Smoothing**     | Complete dataset processing      | All features                     |
| **Streaming Smoothing** | Large dataset chunked processing | Memory efficiency, chunk merging |
| **Online Smoothing**    | Real-time point-by-point updates | Incremental updates              |

## Running Examples

=== "Python"
    ```bash
    cd examples/python
    pip install fastlowess matplotlib numpy
    python batch_smoothing.py
    ```

=== "R"
    ```bash
    Rscript examples/r/batch_smoothing.R
    ```

=== "Rust"
    ```bash
    cargo run --example fast_batch_smoothing -p examples
    ```

=== "Julia"
    ```bash
    julia --project=bindings/julia/julia examples/julia/batch_smoothing.jl
    ```

=== "Node.js"
    ```bash
    cd bindings/nodejs && npm install
    node examples/nodejs/batch_smoothing.js
    ```

=== "C++"
    ```bash
    make cpp
    ./examples/cpp/batch_smoothing
    ```
