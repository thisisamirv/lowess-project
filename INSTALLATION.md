<!-- markdownlint-disable MD025 MD041 -->

> [!NOTE]
>
> Installation instructions are available for:
>
> - [R](#r)
> - [Python](#python)
> - [Rust (lowess)](#rust-lowess-no_std-compatible)
> - [Rust (fastLowess)](#rust-fastlowess-parallel--gpu)
> - [Julia](#julia)
> - [Node.js](#nodejs)
> - [WebAssembly](#webassembly)
> - [C++](#c)

---

# R

**From R-universe:**

```r
install.packages("rfastlowess", repos = "https://thisisamirv.r-universe.dev")
```

**Or from conda-forge:**

```r
conda install -c conda-forge r-rfastlowess
```

# Python

**From PyPI:**

```bash
pip install fastlowess
```

**Or from conda-forge:**

```bash
conda install -c conda-forge fastlowess
```

# Rust (lowess, no_std compatible)

**From crates.io:**

```toml
[dependencies]
lowess = "1.1"
```

# Rust (fastLowess, parallel + GPU)

**From crates.io:**

```toml
[dependencies]
fastLowess = { version = "1.1", features = ["cpu"] }
```

# Julia

**From General Registry:**

```julia
using Pkg
Pkg.add("fastLowess")
```

# Node.js

**From npm:**

```bash
npm install fastlowess
```

# WebAssembly

**From npm:**

```bash
npm install fastlowess-wasm
```

**Or via CDN:**

```html
<script type="module">
  import init, { smooth } from 'https://unpkg.com/fastlowess-wasm@latest';
  await init();
</script>
```

# C++

**From source:**

```bash
make cpp
# Links against libfastlowess_cpp.so
```

**Or from conda-forge:**

```bash
conda install -c conda-forge libfastlowess
```
