//! Layer 5: Engine
//!
//! # Purpose
//!
//! This layer orchestrates the smoothing process by coordinating between
//! primitives (traits, utilities) and algorithms (kernels, regression, robustness).
//! It provides the main iteration loops and convergence detection.
//!
//! # Architecture
//!
//! ```text
//! Layer 7: API
//!   ↓
//! Layer 6: Adapters
//!   ↓
//! Layer 5: Engine ← You are here
//!   ↓
//! Layer 4: Evaluation
//!   ↓
//! Layer 3: Algorithms
//!   ↓
//! Layer 2: Math
//!   ↓
//! Layer 1: Primitives
//! ```

/// Unified execution engine for LOWESS smoothing.
pub mod executor;

/// Validation utilities.
pub mod validator;

/// Output types for LOWESS operations.
pub mod output;
