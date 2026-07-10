using Pkg
Pkg.activate(temp = true)
Pkg.add("JuliaFormatter")
using JuliaFormatter
result = format(["bindings/julia/julia", "tests/julia", "examples/julia"], verbose = true, overwrite = true)
println("Format result: ", result)
