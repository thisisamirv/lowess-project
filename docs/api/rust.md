# Rust API

API reference for the `lowess` and `fastLowess` Rust crates.

## Crate Overview

| Crate                                      | Features                          | Use Case         |
|--------------------------------------------|-----------------------------------|------------------|
| [`lowess`](https://docs.rs/lowess)         | `no_std` compatible, minimal deps | Embedded, WASM   |
| [`fastLowess`](https://docs.rs/fastLowess) | Parallel, GPU, ndarray            | High performance |

---

## Builder Pattern

Both crates use a fluent builder pattern:

```rust
use fastLowess::prelude::*;

let model = Lowess::new()
    .fraction(0.5)
    .iterations(3)
    .adapter(Batch)
    .build()?;

let result = model.fit(&x, &y)?;
```

---

## Core Types

### Lowess

The main builder struct.

```rust
impl Lowess {
    pub fn new() -> Self;
    
    // Core parameters
    pub fn fraction(self, f: f64) -> Self;
    pub fn iterations(self, n: usize) -> Self;
    pub fn delta(self, d: f64) -> Self;
    pub fn parallel(self, enabled: bool) -> Self;
    
    // Weight and robustness
    pub fn weight_function(self, wf: WeightFunction) -> Self;
    pub fn robustness_method(self, rm: RobustnessMethod) -> Self;
    pub fn scaling_method(self, sm: ScalingMethod) -> Self;
    pub fn zero_weight_fallback(self, zf: ZeroWeightFallback) -> Self;
    pub fn boundary_policy(self, bp: BoundaryPolicy) -> Self;
    pub fn auto_converge(self, tol: f64) -> Self;
    
    // Output options
    pub fn return_residuals(self) -> Self;
    pub fn return_diagnostics(self) -> Self;
    pub fn return_robustness_weights(self) -> Self;
    pub fn confidence_intervals(self, level: f64) -> Self;
    pub fn prediction_intervals(self, level: f64) -> Self;
    
    // Cross-validation
    pub fn cross_validate<CV: CrossValidation>(self, cv: CV) -> Self;
    
    // Adapter selection
    pub fn adapter<A: Adapter>(self, adapter: A) -> LowessBuilder<A>;
}
```

### LowessResult

The result of fitting a model.

```rust
pub struct LowessResult<T> {
    pub x: Array1<T>,
    pub y: Array1<T>,
    pub fraction_used: T,
    
    // Optional outputs
    pub residuals: Option<Array1<T>>,
    pub std_err: Option<Array1<T>>,
    pub confidence_lower: Option<Array1<T>>,
    pub confidence_upper: Option<Array1<T>>,
    pub prediction_lower: Option<Array1<T>>,
    pub prediction_upper: Option<Array1<T>>,
    pub robustness_weights: Option<Array1<T>>,
    pub diagnostics: Option<Diagnostics<T>>,
    pub cv_results: Option<CVResults<T>>,
}
```

### LowessError

Error type for LOWESS operations.

```rust
pub enum LowessError {
    InvalidFraction(f64),
    InvalidIterations(usize),
    InsufficientData { required: usize, provided: usize },
    MismatchedLengths { x_len: usize, y_len: usize },
    EmptyInput,
    // ... more variants
}
```

---

## Enums

### WeightFunction

```rust
pub enum WeightFunction {
    Tricube,      // Default
    Epanechnikov,
    Gaussian,
    Biweight,
    Cosine,
    Triangle,
    Uniform,
}
```

### RobustnessMethod

```rust
pub enum RobustnessMethod {
    Bisquare,  // Default
    Huber,
    Talwar,
}
```

### BoundaryPolicy

```rust
pub enum BoundaryPolicy {
    Extend,      // Default
    Reflect,
    Zero,
    NoBoundary,
}
```

### Backend (fastLowess only)

```rust
pub enum Backend {
    CPU,  // Default
    GPU,  // Beta
}
```

---

## Adapters

### Batch

```rust
use fastLowess::prelude::*;

let model = Lowess::new()
    .adapter(Batch)
    .backend(CPU)  // or GPU
    .build()?;

let result = model.fit(&x, &y)?;
```

### Streaming

```rust
let mut processor = Lowess::new()
    .adapter(Streaming)
    .chunk_size(5000)
    .overlap(500)
    .merge_strategy(Average)
    .build()?;

let result = processor.process_chunk(&x, &y)?;
let final_result = processor.finalize()?;
```

### Online

```rust
let mut processor = Lowess::new()
    .adapter(Online)
    .window_capacity(100)
    .min_points(5)
    .update_mode(Incremental)
    .build()?;

if let Some(output) = processor.add_point(x, y)? {
    println!("Smoothed: {}", output.smoothed);
}
```

---

## Prelude

Import common types with:

```rust
use lowess::prelude::*;
// or
use fastLowess::prelude::*;
```

This imports:

- `Lowess` — Builder
- `LowessResult` — Result type
- `LowessError` — Error type
- `Batch`, `Streaming`, `Online` — Adapters
- All enums (`WeightFunction`, `RobustnessMethod`, etc.)
- `KFold`, `LOOCV` — Cross-validation types

---

## Full Documentation

For complete API documentation including all methods and types:

- [`lowess` on docs.rs](https://docs.rs/lowess)
- [`fastLowess` on docs.rs](https://docs.rs/fastLowess)
