# Getting Started

**LOWESS** (Locally Weighted Scatterplot Smoothing) fits smooth curves through noisy data by performing weighted local regressions — no global functional form assumed. This project provides a high-performance Rust core with bindings for R, Python, Julia, Node.js, WebAssembly, and C++.

Key capabilities:

- **Batch** — full dataset smoothing with confidence/prediction intervals, diagnostics, and cross-validation
- **Streaming** — chunk-based processing for datasets that exceed memory
- **Online** — sliding-window smoothing for live sensor or time-series data

:::{toctree}
:maxdepth: 1
:caption: Getting Started

installation
quickstart
concepts
:::
