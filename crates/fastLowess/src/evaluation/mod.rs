//! Layer 4: Evaluation
//!
//! ## Purpose
//!
//! This layer provides parallel implementations of statistical evaluation tools:
//! - Parallel cross-validation for bandwidth selection
//! - Parallel estimation of confidence and prediction intervals
//!
//! ## Architecture
//!
//! ```text
//! Layer 7: API
//!   ↓
//! Layer 6: Adapters
//!   ↓
//! Layer 5: Engine
//!   ↓
//! Layer 4: Evaluation ← You are here
//!   ↓
//! lowess
//! ```

/// Parallel cross-validation for bandwidth selection
pub mod cv;

/// Parallel estimation of confidence and prediction intervals
pub mod intervals;
