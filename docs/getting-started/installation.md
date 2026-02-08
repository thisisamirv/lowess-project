<!-- markdownlint-disable MD024 MD046 -->
# Installation

Install the LOWESS library for your preferred language.

=== "R"

    === "From R-universe (recommended)"

    ```r
    install.packages("rfastlowess", repos = "https://thisisamirv.r-universe.dev")
    ```

    === "From conda-forge"

    ```r
    conda install -c conda-forge r-rfastlowess
    ```

    === "From Source"

    ```r
    # Install Rust first: https://rustup.rs/
    devtools::install_github("thisisamirv/lowess-project", subdir = "bindings/r")
    ```

=== "Python"

    === "From PyPI (recommended)"

    ```bash
    pip install fastlowess
    ```

    === "From conda-forge"

    ```bash
    conda install -c conda-forge fastlowess
    ```

    === "From Source"

    ```bash
    # Install Rust first: https://rustup.rs/
    git clone https://github.com/thisisamirv/lowess-project
    cd lowess-project/bindings/python
    pip install maturin
    maturin develop --release
    ```

=== "Rust"

    === "From crates.io"

    ```toml
    # lowess (no_std compatible)
    [dependencies]
    lowess = "1.1"

    # fastLowess (parallel + GPU)
    [dependencies]
    fastLowess = { version = "1.1", features = ["cpu"] }
    ```

    === "Feature Flags"

    | Crate        | Feature | Description                             |
    |--------------|---------|-----------------------------------------|
    | `lowess`     | `std`   | Enable standard library (default)       |
    | `fastLowess` | `cpu`   | Enable CPU parallelism via Rayon        |
    | `fastLowess` | `gpu`   | Enable GPU acceleration via wgpu (beta) |

=== "Julia"

    === "From General Registry (recommended)"

    ```julia
    Pkg.add("FastLOWESS")
    ```

    === "From Source"

    ```julia
    using Pkg
    Pkg.develop(url="https://github.com/thisisamirv/lowess-project", subdir="bindings/julia/julia")
    ```

=== "Node.js"

    === "From NPM (recommended)"

    ```bash
    npm install fastlowess
    ```

    === "From Source"

    ```bash
    git clone https://github.com/thisisamirv/lowess-project
    cd lowess-project/bindings/nodejs
    npm install
    npm run build
    ```

=== "WebAssembly"

    === "From NPM (recommended)"

    ```bash
    npm install fastlowess-wasm
    ```

    === "From CDN"

    ```html
    <script type="module">
      import { smooth } from "https://cdn.jsdelivr.net/npm/fastlowess-wasm@0.99/index.js";
    </script>
    ```

    === "From Source"

    ```bash
    # Install Rust first: https://rustup.rs/
    # Install wasm-pack: https://rustwasm.github.io/wasm-pack/installer/
    git clone https://github.com/thisisamirv/lowess-project
    cd lowess-project/bindings/wasm
    # For bundlers (Webpack, Vite, etc.)
    wasm-pack build --target bundler
    # For Node.js
    wasm-pack build --target nodejs
    # For browser (no bundler)
    wasm-pack build --target web
    ```

=== "C++"

    === "Pre-built Binaries (Linux (x64))"

    ```bash
    wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-x64.so
    wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
    g++ -o myapp myapp.cpp -L. -lfastlowess-linux-x64
    ```

    === "Pre-built Binaries (macOS (x64))"

    ```bash
    curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-macos-x64.dylib
    curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
    clang++ -o myapp myapp.cpp -L. -lfastlowess-macos-x64
    ```

    === "Pre-built Binaries (Windows (x64))"

    ```powershell
    wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess-win32-x64.dll
    wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
    cl myapp.cpp /link fastlowess-win32-x64.lib
    ```

    === "From Source"

    ```bash
    # Install Rust first: https://rustup.rs/
    git clone https://github.com/thisisamirv/lowess-project
    cd lowess-project/bindings/cpp

    # Build the library
    cargo build --release

    # Headers are at: include/fastlowess.hpp (C++)
    # Library is at: target/release/libfastlowess_cpp.so (Linux)
    #                target/release/libfastlowess_cpp.dylib (macOS)
    #                target/release/fastlowess_cpp.dll (Windows)
    ```

    === "From conda-forge"

    ```bash
    conda install -c conda-forge libfastlowess
    ```

---

## Verify Installation

=== "R"

    ```r
    library(rfastlowess)
    
    x <- c(1, 2, 3)
    y <- c(2, 4, 6)
    
    result <- fastlowess(x, y)
    print("Installed successfully!")
    ```

=== "Python"

    ```python
    import fastlowess as fl
    import numpy as np
    
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    
    result = fl.smooth(x, y)
    print("Installed successfully!")
    ```

=== "Rust"

    ```rust
    use lowess::prelude::*;
    
    fn main() -> Result<(), LowessError> {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        
        let model = Lowess::new().adapter(Batch).build()?;
        let result = model.fit(&x, &y)?;
        
        println!("Installed successfully!");
        Ok(())
    }
    ```

=== "Julia"

    ```julia
    using FastLOWESS
    
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0, 6.0]
    
    result = smooth(x, y)
    println("Installed successfully!")
    ```

=== "Node.js"

    ```javascript
    const fl = require('fastlowess');
    
    const x = new Float64Array([1.0, 2.0, 3.0]);
    const y = new Float64Array([2.0, 4.0, 6.0]);
    
    const result = fl.smooth(x, y);
    console.log("Installed successfully!");
    ```

=== "WebAssembly"

    ```javascript
    import init, { smooth } from 'fastlowess-wasm';

    async function verify() {
        await init();
        const x = new Float64Array([1.0, 2.0, 3.0]);
        const y = new Float64Array([2.0, 4.0, 6.0]);
        const result = smooth(x, y);
        console.log("Installed successfully!");
    }
    verify();
    ```

=== "C++"

    ```cpp
    #include <fastlowess.hpp>
    #include <iostream>
    #include <vector>

    int main() {
        std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<double> y = {2.0, 4.1, 5.9, 8.2, 9.8};

        fastlowess::Lowess model;
        auto result = model.fit(x, y);

        std::cout << "Installed successfully!" << std::endl;
        return 0;
    }
    ```
