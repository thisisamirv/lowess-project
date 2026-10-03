# Installation

Install the LOWESS library for your preferred language.

## From General Registry (recommended)

```julia
Pkg.add("FastLOWESS")
```

## From Source

```julia
using Pkg
Pkg.develop(url="https://github.com/thisisamirv/lowess-project", subdir="bindings/julia/julia")
```

---

## Verify Installation

```@example installation
using FastLOWESS

x = [1.0, 2.0, 3.0]
y = [2.0, 4.0, 6.0]

model = Lowess()
result = fit(model, x, y)
println("Installed successfully!")
```

## Check the Package Version

Read the Julia binding version from its installed package metadata:

```@example package_version
using FastLOWESS
println(FastLOWESS.version())
```

This reports the Julia package version, not the underlying Rust library version.
