#![cfg(feature = "dev")]
//! Tests for input validation utilities.
//!
//! These tests verify the validation functions used in LOWESS for:
//! - Input array validation (length, emptiness, numeric validity)
//! - Parameter validation (fraction, tolerance, interval levels)
//! - Error handling and error messages
//!
//! ## Test Organization
//!
//! 1. **Input Validation** - Array validation, length checks
//! 2. **Parameter Validation** - Fraction, tolerance, interval levels
//! 3. **Error Messages** - Proper error reporting

use lowess::internals::engine::validator::Validator;
use lowess::internals::primitives::errors::LowessError;

// ============================================================================
// Helper Functions
// ============================================================================

fn make_valid_xy() -> (Vec<f64>, Vec<f64>) {
    (vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0])
}

// ============================================================================
// Input Validation Tests
// ============================================================================

/// Test validation rejects empty input.
///
/// Verifies that empty arrays produce EmptyInput error.
#[test]
fn test_validate_empty_input() {
    let x: Vec<f64> = vec![];
    let y: Vec<f64> = vec![];
    let res = Validator::validate_inputs(&x, &y);

    assert!(
        matches!(res, Err(LowessError::EmptyInput)),
        "Empty input should error"
    );
}

/// Test validation rejects length mismatch.
///
/// Verifies that mismatched x and y lengths produce error.
#[test]
fn test_validate_length_mismatch() {
    let x = vec![0.0, 1.0];
    let y = vec![1.0];
    let res = Validator::validate_inputs(&x, &y);

    assert!(
        matches!(
            res,
            Err(LowessError::MismatchedInputs { x_len: 2, y_len: 1 })
        ),
        "Length mismatch should error"
    );
}

/// Test validation rejects too few points.
///
/// Verifies that single point produces TooFewPoints error.
#[test]
fn test_validate_too_few_points() {
    let x = vec![0.0];
    let y = vec![1.0];
    let res = Validator::validate_inputs(&x, &y);

    assert!(
        matches!(res, Err(LowessError::TooFewPoints { got: 1, min: 2 })),
        "Single point should error"
    );
}

/// Test validation rejects non-finite values in x.
///
/// Verifies that NaN and Infinity in x produce errors.
#[test]
fn test_validate_nonfinite_x() {
    let x = vec![0.0, f64::NAN];
    let y = vec![1.0, 2.0];
    let res_x = Validator::validate_inputs(&x, &y);

    if let Err(LowessError::InvalidNumericValue(s)) = res_x {
        assert!(
            s.contains("x[1]") || s.contains("x[0]"),
            "Error should mention x array"
        );
    } else {
        panic!("Expected InvalidNumericValue for x");
    }
}

/// Test validation rejects non-finite values in y.
///
/// Verifies that NaN and Infinity in y produce errors.
#[test]
fn test_validate_nonfinite_y() {
    let x = vec![0.0, 1.0];
    let y = vec![1.0, f64::INFINITY];
    let res_y = Validator::validate_inputs(&x, &y);

    if let Err(LowessError::InvalidNumericValue(s)) = res_y {
        assert!(
            s.contains("y[1]") || s.contains("y[0]"),
            "Error should mention y array"
        );
    } else {
        panic!("Expected InvalidNumericValue for y");
    }
}

/// Test validation accepts valid input.
///
/// Verifies that valid arrays pass validation.
#[test]
fn test_validate_valid_input() {
    let (x, y) = make_valid_xy();
    let res = Validator::validate_inputs(&x, &y);

    assert!(res.is_ok(), "Valid input should pass");

    // Edge case: fraction=1.0 is valid
    assert!(
        Validator::validate_inputs(&x, &y).is_ok(),
        "Fraction 1.0 should be valid"
    );
}

// ============================================================================
// Parameter Validation Tests
// ============================================================================

/// Test validation rejects invalid fractions.
///
/// Verifies that fraction <= 0 or > 1 produces errors.
#[test]
fn test_validate_invalid_fraction() {
    // Fraction = 0
    assert!(
        matches!(
            Validator::validate_fraction(0.0),
            Err(LowessError::InvalidFraction(_))
        ),
        "Fraction 0 should error"
    );

    // Fraction > 1
    assert!(
        matches!(
            Validator::validate_fraction(1.5),
            Err(LowessError::InvalidFraction(_))
        ),
        "Fraction > 1 should error"
    );

    // Fraction = NaN
    assert!(
        matches!(
            Validator::validate_fraction(f64::NAN),
            Err(LowessError::InvalidFraction(_))
        ),
        "Fraction NaN should error"
    );
}

/// Test tolerance validation.
///
/// Verifies that invalid tolerance values produce errors.
#[test]
fn test_validate_tolerance() {
    // Zero tolerance
    assert!(
        matches!(
            Validator::validate_tolerance(0.0),
            Err(LowessError::InvalidTolerance(t)) if t == 0.0
        ),
        "Zero tolerance should error"
    );

    // Negative tolerance
    assert!(
        matches!(
            Validator::validate_tolerance(-1.0),
            Err(LowessError::InvalidTolerance(t)) if t == -1.0
        ),
        "Negative tolerance should error"
    );

    // NaN tolerance
    assert!(
        matches!(
            Validator::validate_tolerance(f64::NAN),
            Err(LowessError::InvalidTolerance(t)) if t.is_nan()
        ),
        "NaN tolerance should error"
    );

    // Valid tolerance
    assert!(
        Validator::validate_tolerance(1e-6).is_ok(),
        "Valid tolerance should pass"
    );
}

/// Test interval level validation.
///
/// Verifies that invalid confidence levels produce errors.
#[test]
fn test_validate_interval_level() {
    // Valid level
    assert!(
        Validator::validate_interval_level(0.95).is_ok(),
        "Valid level should pass"
    );

    // Level > 1
    match Validator::validate_interval_level(1.5) {
        Err(LowessError::InvalidIntervals(v)) => {
            assert!((v - 1.5).abs() < 1e-10, "Error should contain value")
        }
        _ => panic!("Expected InvalidIntervals error"),
    }

    // Level < 0
    match Validator::validate_interval_level(-0.1) {
        Err(LowessError::InvalidIntervals(v)) => {
            assert!((v + 0.1).abs() < 1e-10, "Error should contain value")
        }
        _ => panic!("Expected InvalidIntervals error"),
    }

    // Level = NaN
    match Validator::validate_interval_level(f64::NAN) {
        Err(LowessError::InvalidIntervals(v)) => {
            assert!(v.is_nan(), "Error should contain NaN")
        }
        _ => panic!("Expected InvalidIntervals error"),
    }
}

/// Test fraction boundary values.
///
/// Verifies correct handling of boundary fraction values.
#[test]
fn test_validate_fraction_boundaries() {
    // Fraction = 1.0 is valid (upper boundary)
    assert!(
        Validator::validate_fraction(1.0).is_ok(),
        "Fraction 1.0 should be valid"
    );

    // Fraction slightly above 0 is valid
    assert!(
        Validator::validate_fraction(0.001).is_ok(),
        "Small positive fraction should be valid"
    );

    // Fraction slightly above 1 is invalid
    assert!(
        matches!(
            Validator::validate_fraction(1.001),
            Err(LowessError::InvalidFraction(_))
        ),
        "Fraction > 1 should error"
    );
}

/// Test tolerance boundary values.
///
/// Verifies correct handling of boundary tolerance values.
#[test]
fn test_validate_tolerance_boundaries() {
    // Very small positive tolerance is valid
    assert!(
        Validator::validate_tolerance(1e-12).is_ok(),
        "Very small positive tolerance should be valid"
    );

    // Exactly zero is invalid
    assert!(
        matches!(
            Validator::validate_tolerance(0.0),
            Err(LowessError::InvalidTolerance(_))
        ),
        "Zero tolerance should error"
    );

    // Large tolerance is valid
    assert!(
        Validator::validate_tolerance(1.0).is_ok(),
        "Large tolerance should be valid"
    );
}

/// Test interval level boundary values.
///
/// Verifies correct handling of boundary interval levels.
#[test]
fn test_validate_interval_level_boundaries() {
    // Common valid levels
    assert!(Validator::validate_interval_level(0.90).is_ok());
    assert!(Validator::validate_interval_level(0.95).is_ok());
    assert!(Validator::validate_interval_level(0.99).is_ok());

    // Exactly 0 is invalid
    assert!(matches!(
        Validator::validate_interval_level(0.0),
        Err(LowessError::InvalidIntervals(_))
    ));

    // Exactly 1 is invalid
    assert!(matches!(
        Validator::validate_interval_level(1.0),
        Err(LowessError::InvalidIntervals(_))
    ));

    // Very close to boundaries
    assert!(Validator::validate_interval_level(0.001).is_ok());
    assert!(Validator::validate_interval_level(0.999).is_ok());
}

// ============================================================================
// Additional Validator Edge Cases
// ============================================================================

/// Test iterations validation at MAX_ITERATIONS boundary.
#[test]
fn test_validate_iterations_boundary() {
    // Exactly at 1000 should pass
    assert!(Validator::validate_iterations(1000).is_ok());

    // 1001 should fail
    assert!(Validator::validate_iterations(1001).is_err());
}

/// Test overlap validation when equal to chunk size.
#[test]
fn test_validate_chunk_overlap_equal() {
    // Overlap < chunk_size should pass
    assert!(Validator::validate_overlap(99, 100).is_ok());

    // Overlap == chunk_size should fail
    assert!(Validator::validate_overlap(100, 100).is_err());

    // Overlap > chunk_size should also fail
    assert!(Validator::validate_overlap(101, 100).is_err());
}

/// Test CV fractions validation with single fraction.
#[test]
fn test_validate_cv_fractions_single() {
    // Single valid fraction should pass
    assert!(Validator::validate_cv_fractions(&[0.5]).is_ok());

    // Single invalid fraction should fail
    assert!(Validator::validate_cv_fractions(&[0.0]).is_err());
    assert!(Validator::validate_cv_fractions(&[1.5]).is_err());
}
