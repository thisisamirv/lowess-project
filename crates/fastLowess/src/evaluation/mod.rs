//! Layer 4: Evaluation
//!
//! This layer provides parallel implementations of statistical evaluation tools:
//! - Parallel cross-validation for bandwidth selection
//! - Parallel estimation of confidence and prediction intervals

// Parallel cross-validation for bandwidth selection
pub mod cv;

// Parallel estimation of confidence and prediction intervals
pub mod intervals;
