---
title: "Benchmarks"
weight: 90
---

The Go binding calls the same `fastLowess` Rust core as every other binding (C++, Python, R, Julia, Node.js, WASM) through a thin `cgo` FFI layer, so its raw fit performance is governed by the same underlying engine — see the [C++ binding's benchmarks](https://thisisamirv.github.io/lowess-project/cpp/benchmarks.html) for representative numbers (R baseline vs. serial vs. parallel, across scale/fraction/iteration/pathological-data scenarios).

A dedicated Go benchmark harness (using `testing.B`, wired into `benchmarks/compare.py` alongside the R/Rust comparisons) has not been added yet — this is a good follow-up once the binding has seen some real-world use. In the meantime, `go test -bench=.` can be used for local relative comparisons between option configurations (e.g. serial vs. `Parallel: true`, different `Fraction`/`Iterations` values).
