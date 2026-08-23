# Package index

## Core LOWESS Interface

Main R6 classes for LOWESS smoothing.

- [`rfastlowess`](https://thisisamirv.github.io/lowess-project/r/reference/rfastlowess-package.md)
  [`rfastlowess-package`](https://thisisamirv.github.io/lowess-project/r/reference/rfastlowess-package.md)
  : rfastlowess: High-performance LOWESS Smoothing for R
- [`Lowess()`](https://thisisamirv.github.io/lowess-project/r/reference/Lowess.md)
  : LOWESS Batch Smoothing
- [`StreamingLowess()`](https://thisisamirv.github.io/lowess-project/r/reference/StreamingLowess.md)
  : LOWESS Streaming Smoothing
- [`OnlineLowess()`](https://thisisamirv.github.io/lowess-project/r/reference/OnlineLowess.md)
  : LOWESS Online Smoothing

## Results and Utilities

Objects returned by fit methods and helper functions.

- [`fit()`](https://thisisamirv.github.io/lowess-project/r/reference/fit.md)
  : Fit a LOWESS model to data
- [`process_chunk()`](https://thisisamirv.github.io/lowess-project/r/reference/process_chunk.md)
  : Process a data chunk through a streaming LOWESS model
- [`finalize()`](https://thisisamirv.github.io/lowess-project/r/reference/finalize.md)
  : Finalize a streaming LOWESS model
- [`add_point()`](https://thisisamirv.github.io/lowess-project/r/reference/add_point.md)
  : Add a single point to an online LOWESS model
- [`plot(`*`<LowessResult>`*`)`](https://thisisamirv.github.io/lowess-project/r/reference/plot.LowessResult.md)
  : Plot Lowess Result
- [`print(`*`<Lowess>`*`)`](https://thisisamirv.github.io/lowess-project/r/reference/print.Lowess.md)
  : Print Lowess Model
- [`print(`*`<LowessResult>`*`)`](https://thisisamirv.github.io/lowess-project/r/reference/print.LowessResult.md)
  : Print Lowess Result
- [`print(`*`<OnlineLowess>`*`)`](https://thisisamirv.github.io/lowess-project/r/reference/print.OnlineLowess.md)
  : Print OnlineLowess Model
- [`print(`*`<StreamingLowess>`*`)`](https://thisisamirv.github.io/lowess-project/r/reference/print.StreamingLowess.md)
  : Print StreamingLowess Model

## Types

Helper types for documentation.

- [`Nullable()`](https://thisisamirv.github.io/lowess-project/r/reference/Nullable.md)
  : Nullable Value Wrapper

## GPU Backend

Check for and install the optional GPU-accelerated backend.

- [`gpu_available()`](https://thisisamirv.github.io/lowess-project/r/reference/gpu_available.md)
  : Check GPU Backend Availability
- [`install_gpu()`](https://thisisamirv.github.io/lowess-project/r/reference/install_gpu.md)
  : Download and Install the GPU-Enabled Backend
