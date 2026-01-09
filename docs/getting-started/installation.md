# Installation

Install the LOWESS library for your preferred language.

## R

**From R-universe (recommended):**

Pre-built binaries, no Rust toolchain required:

```r
install.packages("rfastlowess", repos = "https://thisisamirv.r-universe.dev")
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
Pkg.add("fastLowess")
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
    using fastLowess
    
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

    See [WebAssembly API](../api/webassembly.md) for full reference.
