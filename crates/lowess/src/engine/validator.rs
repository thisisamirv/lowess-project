//! Input validation for LOWESS configuration and data.
//!
//! ## Purpose
//!
//! This module provides comprehensive validation functions for LOWESS
//! configuration parameters and input data. It checks requirements
//! such as input lengths, finite values, and parameter bounds.
//!
//! ## Design notes
//!
//! * **Fail-Fast**: Validation stops at the first error encountered.
//! * **Efficiency**: Checks are ordered from cheap to expensive.
//! * **Generics**: Validation is generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Parameter Bounds**: Enforces constraints like fraction in (0, 1].
//! * **Finite Checks**: Ensures all inputs are finite (no NaN/Inf).
//! * **Regression Requirements**: Ensures at least 2 points for linear regression.
//!
//! ## Invariants
//!
//! * All validated inputs satisfy their respective mathematical constraints.
//! * Validation logic is deterministic and side-effect free.
//!
//! ## Non-goals
//!
//! * This module does not sort, transform, or filter input data.
//! * This module does not provide automatic correction of invalid inputs.
//! * This module does not perform the smoothing or optimization itself.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::format;

// External dependencies
use num_traits::Float;

// Internal dependencies
use crate::primitives::errors::LowessError;

// ============================================================================
// Validator
// ============================================================================

/// Validation utility for LOWESS configuration and input data.
///
/// Provides static methods for validating various LOWESS parameters and
/// input data. All methods return `Result<(), LowessError>` and fail fast
/// upon identifying the first violation.
pub struct Validator;

impl Validator {
    // ========================================================================
    // Core Input Validation
    // ========================================================================

    /// Validate input arrays for LOWESS smoothing.
    pub fn validate_inputs<T: Float>(x: &[T], y: &[T]) -> Result<(), LowessError> {
        // Check 1: Non-empty arrays
        if x.is_empty() || y.is_empty() {
            return Err(LowessError::EmptyInput);
        }

        // Check 2: Matching lengths
        let n = x.len();
        if n != y.len() {
            return Err(LowessError::MismatchedInputs {
                x_len: n,
                y_len: y.len(),
            });
        }

        // Check 3: Sufficient points for regression
        if n < 2 {
            return Err(LowessError::TooFewPoints { got: n, min: 2 });
        }

        // Check 4: All values finite (combined loop for cache locality)
        for i in 0..n {
            if !x[i].is_finite() {
                return Err(LowessError::InvalidNumericValue(format!(
                    "x[{}]={}",
                    i,
                    x[i].to_f64().unwrap_or(f64::NAN)
                )));
            }
            if !y[i].is_finite() {
                return Err(LowessError::InvalidNumericValue(format!(
                    "y[{}]={}",
                    i,
                    y[i].to_f64().unwrap_or(f64::NAN)
                )));
            }
        }

        Ok(())
    }

    /// Validate a single numeric value for finiteness.
    pub fn validate_scalar<T: Float>(val: T, name: &str) -> Result<(), LowessError> {
        if !val.is_finite() {
            return Err(LowessError::InvalidNumericValue(format!(
                "{}={}",
                name,
                val.to_f64().unwrap_or(f64::NAN)
            )));
        }
        Ok(())
    }

    // ========================================================================
    // Parameter Validation
    // ========================================================================

    /// Validate the smoothing fraction (bandwidth) parameter.
    pub fn validate_fraction<T: Float>(fraction: T) -> Result<(), LowessError> {
        if !fraction.is_finite() || fraction <= T::zero() || fraction > T::one() {
            return Err(LowessError::InvalidFraction(
                fraction.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    /// Validate the number of robustness iterations.
    ///
    /// # Notes
    ///
    /// * 0 iterations means initial fit only (no robustness).
    /// * Maximum of 1000 iterations to prevent excessive computation.
    pub fn validate_iterations(iterations: usize) -> Result<(), LowessError> {
        const MAX_ITERATIONS: usize = 1000;
        if iterations > MAX_ITERATIONS {
            return Err(LowessError::InvalidIterations(iterations));
        }
        Ok(())
    }

    /// Validate the confidence/prediction interval level.
    pub fn validate_interval_level<T: Float>(level: T) -> Result<(), LowessError> {
        if !level.is_finite() || level <= T::zero() || level >= T::one() {
            return Err(LowessError::InvalidIntervals(
                level.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    /// Validate a collection of candidate fractions for cross-validation.
    pub fn validate_cv_fractions<T: Float>(fracs: &[T]) -> Result<(), LowessError> {
        if fracs.is_empty() {
            return Err(LowessError::InvalidFraction(0.0));
        }

        for &f in fracs {
            Self::validate_fraction(f)?;
        }

        Ok(())
    }

    /// Validate the number of folds for k-fold cross-validation.
    pub fn validate_kfold(k: usize) -> Result<(), LowessError> {
        if k < 2 {
            return Err(LowessError::InvalidNumericValue(format!(
                "k-fold must be at least 2, got {}",
                k
            )));
        }
        Ok(())
    }

    /// Validate the auto-convergence tolerance.
    pub fn validate_tolerance<T: Float>(tol: T) -> Result<(), LowessError> {
        if !tol.is_finite() || tol <= T::zero() {
            return Err(LowessError::InvalidTolerance(
                tol.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    /// Validate delta optimization parameter (equivalent to cell size in some contexts).
    pub fn validate_delta<T: Float>(delta: T) -> Result<(), LowessError> {
        if !delta.is_finite() || delta < T::zero() {
            return Err(LowessError::InvalidDelta(
                delta.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    // ========================================================================
    // Adapter-Specific Validation
    // ========================================================================

    /// Validate the chunk size for shared processing in streaming mode.
    pub fn validate_chunk_size(chunk_size: usize, min: usize) -> Result<(), LowessError> {
        if chunk_size < min {
            return Err(LowessError::InvalidChunkSize {
                got: chunk_size,
                min,
            });
        }
        Ok(())
    }

    /// Validate the overlap between consecutive chunks in streaming mode.
    pub fn validate_overlap(overlap: usize, chunk_size: usize) -> Result<(), LowessError> {
        if overlap >= chunk_size {
            return Err(LowessError::InvalidOverlap {
                overlap,
                chunk_size,
            });
        }
        Ok(())
    }

    /// Validate the maximum capacity of the sliding window in online mode.
    pub fn validate_window_capacity(window_capacity: usize, min: usize) -> Result<(), LowessError> {
        if window_capacity < min {
            return Err(LowessError::InvalidWindowCapacity {
                got: window_capacity,
                min,
            });
        }
        Ok(())
    }

    /// Validate the activation threshold for online smoothing.
    pub fn validate_min_points(
        min_points: usize,
        window_capacity: usize,
    ) -> Result<(), LowessError> {
        if min_points < 2 || min_points > window_capacity {
            return Err(LowessError::InvalidMinPoints {
                got: min_points,
                window_capacity,
            });
        }
        Ok(())
    }

    /// Validate that no parameters were set multiple times in the builder.
    pub fn validate_no_duplicates(
        duplicate_param: Option<&'static str>,
    ) -> Result<(), LowessError> {
        if let Some(param) = duplicate_param {
            return Err(LowessError::DuplicateParameter { parameter: param });
        }
        Ok(())
    }
}
