//! Layer 5: Engine
//!
//! This layer orchestrates the smoothing process by coordinating between
//! primitives (traits, utilities) and algorithms (kernels, regression, robustness).
//! It provides the main iteration loops and convergence detection.

// Unified execution engine for LOWESS smoothing.
pub mod executor;

// Validation utilities.
pub mod validator;

// Output types for LOWESS operations.
pub mod output;
