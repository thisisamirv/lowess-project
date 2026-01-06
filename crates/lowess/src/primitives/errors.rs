//! Error types for LOWESS operations.
//!
//! ## Purpose
//!
//! This module defines error conditions that can occur during LOWESS smoothing,
//! including input validation, parameter constraints, and adapter limitations.
//!
//! ## Design notes
//!
//! * **Contextual**: Errors include relevant values (e.g., actual vs. expected lengths).
//! * **Deferred**: Errors are often caught and stored during builder configuration.
//! * **No-std**: Supports `no_std` environments by using `alloc` for dynamic messages.
//! * **Trait Implementation**: Implements `Display` and `std::error::Error` (when `std` is enabled).
//!
//! ## Key concepts
//!
//! 1. **Input validation**: Empty arrays, mismatched lengths, non-finite values.
//! 2. **Parameter validation**: Invalid fraction, delta, iterations, or interval levels.
//! 3. **Adapter constraints**: Invalid chunk size, overlap, or window capacity.
//! 4. **Feature support**: Features not supported by the selected execution adapter.
//!
//! ## Invariants
//!
//! * All variants provide sufficient context for diagnosis.
//! * Error messages are consistent in tone and formatting.
//! * Numeric values in errors use the same types as the public API.
//!
//! ## Non-goals
//!
//! * This module does not perform the validation logic itself.
//! * This module does not provide error recovery or fallback strategies.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(feature = "std")]
use std::error::Error;
#[cfg(feature = "std")]
use std::string::String;

// External dependencies
use core::fmt::{Display, Formatter, Result};

// ============================================================================
// Error Type
// ============================================================================

/// Error type for LOWESS operations.
#[derive(Debug, Clone, PartialEq)]
pub enum LowessError {
    /// Input arrays are empty; LOWESS requires at least 2 points.
    EmptyInput,

    /// Generic invalid input error with a descriptive message.
    InvalidInput(String),

    /// `x` and `y` arrays must have the same number of elements.
    MismatchedInputs {
        /// Number of elements in the `x` array.
        x_len: usize,
        /// Number of elements in the `y` array.
        y_len: usize,
    },

    /// Input data contains NaN or infinite values.
    InvalidNumericValue(String),

    /// Number of points is below the minimum requirement for the selected parameters.
    TooFewPoints {
        /// Number of points provided.
        got: usize,
        /// Minimum required points.
        min: usize,
    },

    /// Smoothing fraction must be in the range (0, 1].
    InvalidFraction(f64),

    /// Delta controls interpolation optimization and must be non-negative.
    InvalidDelta(f64),

    /// Local regression requires at least 1 iteration.
    InvalidIterations(usize),

    /// Interval coverage level must be strictly between 0 and 1.
    InvalidIntervals(f64),

    /// Convergence tolerance must be positive and finite.
    InvalidTolerance(f64),

    /// Chunk size must be large enough to accommodate the minimum window.
    InvalidChunkSize {
        /// The chunk size provided.
        got: usize,
        /// Minimum required chunk size.
        min: usize,
    },

    /// Overlap must be strictly less than the chunk size to ensure progress.
    InvalidOverlap {
        /// The overlap provided.
        overlap: usize,
        /// The chunk size.
        chunk_size: usize,
    },

    /// Window capacity must be large enough for the requested smoothing parameters.
    InvalidWindowCapacity {
        /// The window capacity provided.
        got: usize,
        /// Minimum required window capacity.
        min: usize,
    },

    /// Minimum points must be at least 2 and at most the window capacity.
    InvalidMinPoints {
        /// The min_points provided.
        got: usize,
        /// The window capacity.
        window_capacity: usize,
    },

    /// Selected adapter does not support the requested feature (e.g., cross-validation).
    UnsupportedFeature {
        /// Name of the adapter (e.g., "Streaming", "Online").
        adapter: &'static str,
        /// Name of the unsupported feature.
        feature: &'static str,
    },

    /// Parameter was set multiple times in the builder.
    DuplicateParameter {
        /// Name of the parameter that was set multiple times.
        parameter: &'static str,
    },
}

// ============================================================================
// Display Implementation
// ============================================================================

impl Display for LowessError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::EmptyInput => write!(f, "Input arrays are empty"),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Self::MismatchedInputs { x_len, y_len } => {
                write!(f, "Length mismatch: x has {x_len} points, y has {y_len}")
            }
            Self::InvalidNumericValue(s) => write!(f, "Invalid numeric value: {s}"),
            Self::TooFewPoints { got, min } => {
                write!(f, "Too few points: got {got}, need at least {min}")
            }
            Self::InvalidFraction(frac) => {
                write!(f, "Invalid fraction: {frac} (must be > 0 and <= 1)")
            }
            Self::InvalidDelta(delta) => write!(f, "Invalid delta: {delta} (must be >= 0)"),
            Self::InvalidIterations(iter) => {
                write!(f, "Invalid iterations: {iter} (must be in [0, 1000])")
            }
            Self::InvalidIntervals(level) => {
                write!(f, "Invalid interval level: {level} (must be > 0 and < 1)")
            }
            Self::InvalidTolerance(tol) => {
                write!(f, "Invalid tolerance: {tol} (must be > 0 and finite)")
            }
            Self::InvalidChunkSize { got, min } => {
                write!(f, "Invalid chunk_size: {got} (must be at least {min})")
            }
            Self::InvalidOverlap {
                overlap,
                chunk_size,
            } => {
                write!(
                    f,
                    "Invalid overlap: {overlap} (must be less than chunk_size {chunk_size})"
                )
            }
            Self::InvalidWindowCapacity { got, min } => {
                write!(f, "Invalid window_capacity: {got} (must be at least {min})")
            }
            Self::InvalidMinPoints {
                got,
                window_capacity,
            } => {
                write!(
                    f,
                    "Invalid min_points: {got} (must be between 2 and window_capacity {window_capacity})"
                )
            }
            Self::UnsupportedFeature { adapter, feature } => {
                write!(f, "Adapter '{adapter}' does not support feature: {feature}")
            }
            Self::DuplicateParameter { parameter } => {
                write!(
                    f,
                    "Parameter '{parameter}' was set multiple times. Each parameter can only be configured once."
                )
            }
        }
    }
}

// ============================================================================
// Standard Error Trait
// ============================================================================

#[cfg(feature = "std")]
impl Error for LowessError {}
