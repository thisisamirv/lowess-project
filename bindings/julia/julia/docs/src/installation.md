# Installation

Install the LOWESS library for your preferred language.

=== "From General Registry (recommended)"

```julia
Pkg.add("FastLOWESS")
```

=== "From Source"

```julia
using Pkg
Pkg.develop(url="https://github.com/thisisamirv/lowess-project", subdir="bindings/julia/julia")
```

---

## Verify Installation

```julia
using FastLOWESS

x = [1.0, 2.0, 3.0]
y = [2.0, 4.0, 6.0]

model = Lowess()
result = fit(model, x, y)
println("Installed successfully!")
```
