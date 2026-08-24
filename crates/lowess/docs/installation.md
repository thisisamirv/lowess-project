<!-- markdownlint-disable MD024 MD046 -->
# Installation

Install the LOWESS library for your preferred language.

```toml
# lowess (no_std compatible)
[dependencies]
lowess = "1.3"

# lowess (parallel + GPU)
[dependencies]
lowess = { version = "1.3", features = ["cpu"] }
```

## Feature Flags

| Crate | Feature | Description |
| --- | --- | --- |
| `lowess` | `std` | Enable standard library (default) |
| `lowess` | `cpu` | Enable CPU parallelism via Rayon |
| `lowess` | `gpu` | Enable GPU acceleration via wgpu (beta) |

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
