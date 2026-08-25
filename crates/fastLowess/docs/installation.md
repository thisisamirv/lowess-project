<!-- markdownlint-disable MD024 MD046 -->
# Installation

Install the LOWESS library for your preferred language.

```toml
# lowess (no_std compatible)
[dependencies]
lowess = "*"

# fastLowess (parallel + GPU)
[dependencies]
fastLowess = { version = "*", features = ["cpu"] }
```

## Feature Flags

| Crate | Feature | Description |
| --- | --- | --- |
| `lowess` | `std` | Enable standard library (default) |
| `fastLowess` | `cpu` | Enable CPU parallelism via Rayon |
| `fastLowess` | `gpu` | Enable GPU acceleration via wgpu (beta) |

---

## Verify Installation

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];
    
    let model = Lowess::new().build()?;
    let result = model.fit(&x, &y)?;
    
    println!("Installed successfully!");
    Ok(())
}
```

```output
Installed successfully!
```
