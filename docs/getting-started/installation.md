<!-- markdownlint-disable MD024 MD046 -->
# Installation

Install the LOWESS library for your preferred language.

## R

**From R-universe (recommended):**

Pre-built binaries, no Rust toolchain required:

```r
install.packages("rfastlowess", repos = "https://thisisamirv.r-universe.dev")
```

**From conda-forge:**

```r
conda install -c conda-forge r-rfastlowess
```

**From Source:**

Requires Rust toolchain:

```r
# Install Rust first: https://rustup.rs/
devtools::install_github("thisisamirv/lowess-project", subdir = "bindings/r")
```

---

## Python

**From PyPI (recommended):**

```bash
pip install fastlowess
```

**From conda-forge:**

```bash
conda install -c conda-forge fastlowess
```

**From Source:**

```bash
git clone https://github.com/thisisamirv/lowess-project
cd lowess-project/bindings/python
pip install maturin
maturin develop --release
```

---

## Rust

**From crates.io:**

=== "lowess (no_std compatible)"

    ```toml
    [dependencies]
    lowess = "0.99"
    ```

=== "fastLowess (parallel + GPU)"

    ```toml
    [dependencies]
    fastLowess = { version = "0.99", features = ["cpu"] }
    ```

---

## Julia

**From General Registry (recommended):**

```julia
using Pkg
Pkg.add("fastlowess")
```

**From Source:**

```julia
using Pkg
Pkg.develop(url="https://github.com/thisisamirv/lowess-project", subdir="bindings/julia/julia")
```

---

## Node.js

**From NPM (recommended):**

```bash
npm install fastlowess
```

**From Source:**

```bash
git clone https://github.com/thisisamirv/lowess-project
cd lowess-project/bindings/nodejs
npm install
npm run build
```

---

## WebAssembly

**From NPM (recommended):**

```bash
npm install fastlowess-wasm
```

**From CDN:**

```html
<script type="module">
  import { smooth } from "https://cdn.jsdelivr.net/npm/fastlowess-wasm@0.99/index.js";
</script>
```

**From Source:**

Requires Rust toolchain and [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/).

```bash
git clone https://github.com/thisisamirv/lowess-project
cd lowess-project/bindings/wasm
# For bundlers (Webpack, Vite, etc.)
wasm-pack build --target bundler
# For Node.js
wasm-pack build --target nodejs
# For browser (no bundler)
wasm-pack build --target web
```

---

## C++

**Pre-built Binaries (recommended):**

Download pre-built libraries from [GitHub Releases](https://github.com/thisisamirv/lowess-project/releases):

*Linux (x64):*

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-x64.so
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
```

*macOS (x64):*

```bash
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-macos-x64.dylib
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
```

*Windows (x64):*

```powershell
# Download from: https://github.com/thisisamirv/lowess-project/releases/latest
# Files: fastlowess-win32-x64.dll, fastlowess.hpp
```

Then link against the library in your build system:

```bash
# Linux
g++ -o myapp myapp.cpp -L. -lfastlowess-linux-x64

# macOS
clang++ -o myapp myapp.cpp -L. -lfastlowess-macos-x64

# Windows (MSVC)
cl myapp.cpp /link fastlowess-win32-x64.lib
```

**From Source:**

Requires Rust toolchain.

```bash
git clone https://github.com/thisisamirv/lowess-project
cd lowess-project/bindings/cpp

# Build the library
cargo build --release

# Headers are at: include/fastlowess.hpp (C++)
# Library is at: target/release/libfastlowess_cpp.so (Linux)
#                target/release/libfastlowess_cpp.dylib (macOS)
#                target/release/fastlowess_cpp.dll (Windows)
```

**From conda-forge:**

```bash
conda install -c conda-forge libfastlowess
```

---

### Feature Flags

| Crate        | Feature | Description                             |
|--------------|---------|-----------------------------------------|
| `lowess`     | `std`   | Enable standard library (default)       |
| `fastLowess` | `cpu`   | Enable CPU parallelism via Rayon        |
| `fastLowess` | `gpu`   | Enable GPU acceleration via wgpu (beta) |

### Minimum Supported Rust Version (MSRV)

Both crates require **Rust 1.85.0** or later.

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
    using fastlowess
    
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

    See [Node.js API](../api/nodejs.md) for full reference.

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

    See [WebAssembly API](../api/wasm.md) for full reference.

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

    See [C++ API](../api/cpp.md) for full reference.
