# Installation

Install the LOWESS library for your preferred language.

## Rust

Add the crate to your `Cargo.toml`:

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

### Feature Flags

| Crate        | Feature | Description                             |
|--------------|---------|-----------------------------------------|
| `lowess`     | `std`   | Enable standard library (default)       |
| `fastLowess` | `cpu`   | Enable CPU parallelism via Rayon        |
| `fastLowess` | `gpu`   | Enable GPU acceleration via wgpu (beta) |

### Minimum Supported Rust Version (MSRV)

Both crates require **Rust 1.85.0** or later.

---

## Python

Install from PyPI:

```bash
pip install fastlowess
```

Or install from conda-forge:

```bash
conda install -c conda-forge fastlowess
```

### Optional: Install from source

```bash
git clone https://github.com/thisisamirv/lowess-project
cd lowess-project/bindings/python
pip install maturin
maturin develop --release
```

---

## R

### From R-universe (recommended)

Pre-built binaries, no Rust toolchain required:

```r
install.packages("rfastlowess", repos = "https://thisisamirv.r-universe.dev")
```

### From source

Requires Rust toolchain:

```r
# Install Rust first: https://rustup.rs/
devtools::install_github("thisisamirv/lowess-project", subdir = "bindings/r")
```

### Requirements

- R 4.2+
- Rust 1.85+ (for source installation)

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
