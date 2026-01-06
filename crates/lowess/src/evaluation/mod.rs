//! Layer 4: Evaluation
//!
//! # Purpose
//!
//! This layer calculates high-level statistical metrics based on the smoothing results:
//! - Cross-validation for parameter selection
//! - Diagnostic metrics for fit quality
//! - Confidence and prediction intervals
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
//! Layer 4: Evaluation ← You are here
//!   ↓
//! Layer 3: Algorithms
//!   ↓
//! Layer 2: Math
//!   ↓
//! Layer 1: Primitives
//! ```

/// Cross-validation for bandwidth selection.
pub mod cv;

/// Diagnostic metrics for fit quality assessment.
pub mod diagnostics;

/// Confidence and prediction interval computation.
pub mod intervals;
