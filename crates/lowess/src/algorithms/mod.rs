//! Layer 3: Algorithms
//!
//! # Purpose
//!
//! This layer implements the core logic for local weighted regression, robustness
//! iterations, and interpolation. It contains the "business logic" of LOWESS
//! but is orchestrated by the engine layer.
//!
//! # Architecture
//!
//! ```text
//! Layer 7: API
//!   ↓
//! Layer 6: Adapters
//!   ↓
//! Layer 5: Engine
//!   ↓
//! Layer 4: Evaluation
//!   ↓
//! Layer 3: Algorithms ← You are here
//!   ↓
//! Layer 2: Math
//!   ↓
//! Layer 1: Primitives
//! ```

/// Local weighted regression implementations.
pub mod regression;

/// Robustness weight updates for outlier downweighting.
pub mod robustness;

/// Interpolation and delta optimization utilities.
pub mod interpolation;
