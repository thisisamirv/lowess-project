//! Layer 3: Algorithms
//!
//! This layer implements the core logic for local weighted regression, robustness
//! iterations, and interpolation. It contains the "business logic" of LOWESS
//! but is orchestrated by the engine layer.

// Local weighted regression implementations.
pub mod regression;

// Robustness weight updates for outlier downweighting.
pub mod robustness;

// Interpolation and delta optimization utilities.
pub mod interpolation;
