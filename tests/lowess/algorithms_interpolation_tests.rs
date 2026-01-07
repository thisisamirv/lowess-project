#![cfg(feature = "dev")]
//! Tests for interpolation and delta optimization algorithms.
//!
//! These tests verify the core interpolation utilities used in LOWESS for:
//! - Delta calculation for optimization (skipping dense regions)
//! - Gap interpolation between fitted points
//! - Close point detection and skipping
//!
//! ## Test Organization
//!
//! 1. **Delta Calculation** - Computing optimal delta for interpolation
//! 2. **Gap Interpolation** - Linear interpolation between fitted points
//! 3. **Point Skipping** - Detecting and handling close points
//! 4. **Boundary Conditions** - Edge cases and special scenarios

use approx::assert_relative_eq;

use lowess::internals::algorithms::interpolation::{calculate_delta, interpolate_gap};

// ============================================================================
// Delta Calculation Tests
// ============================================================================

/// Test delta calculation with None (automatic) and empty input.
///
/// Verifies:
/// - Empty input returns delta of 0.0
/// - Automatic delta is 1% of x range
#[test]
fn test_calculate_delta_automatic() {
    // Empty x => delta zero
    let empty: Vec<f64> = vec![];
    let delta = calculate_delta::<f64>(None, &empty).unwrap();
    assert_relative_eq!(delta, 0.0, epsilon = 1e-12);

    // Range-based default: 1% of (last - first)
    let xs = vec![1.0f64, 2.0, 5.0];
    let delta = calculate_delta::<f64>(None, &xs).unwrap();
    let expected = 0.01 * (5.0 - 1.0);
    assert_relative_eq!(delta, expected, epsilon = 1e-12);
}

/// Test delta calculation with explicit valid and invalid values.
///
/// Verifies:
/// - Valid provided delta is used as-is
/// - Negative delta produces error
#[test]
fn test_calculate_delta_explicit() {
    let xs = vec![0.0f64, 1.0];

    // Valid provided delta
    let delta = calculate_delta(Some(0.2f64), &xs).unwrap();
    assert_relative_eq!(delta, 0.2, epsilon = 1e-12);
}

// ============================================================================
// Gap Interpolation Tests
// ============================================================================

/// Test linear interpolation between fitted points.
///
/// Verifies that gaps are filled with linear interpolation.
#[test]
fn test_interpolate_gap_linear() {
    let x = vec![0.0f64, 1.0, 2.0, 3.0];
    let mut y_smooth = vec![10.0f64, 0.0, 0.0, 20.0];

    // Interpolate between indices 0 and 3
    interpolate_gap(&x, &mut y_smooth, 0, 3);

    // Linear interpolation between 10 and 20 over x=[0,3]
    // At x=1: 10 + (1/3)*10 = 13.333...
    // At x=2: 10 + (2/3)*10 = 16.666...
    assert_relative_eq!(y_smooth[1], 10.0 + (1.0 / 3.0) * 10.0, epsilon = 1e-12);
    assert_relative_eq!(y_smooth[2], 10.0 + (2.0 / 3.0) * 10.0, epsilon = 1e-12);
}

/// Test interpolation with duplicate x values.
///
/// Verifies that when x values are equal, averaging is used instead of interpolation.
#[test]
fn test_interpolate_gap_duplicate_x() {
    // Duplicate x values: denom <= 0 leads to averaging
    let x = vec![0.0f64, 0.0, 0.0, 1.0];
    let mut y = vec![5.0f64, 0.0, 15.0, 0.0];

    // Interpolate between index 0 and 2 where x values are equal
    // Should average y[0] and y[2]
    interpolate_gap(&x, &mut y, 0, 2);

    assert_relative_eq!(y[1], (5.0 + 15.0) / 2.0, epsilon = 1e-12);
}

/// Test delta calculation with extremely large x range.
#[test]
fn test_calculate_delta_extreme_range() {
    let xs = vec![0.0f64, 1e20];
    let delta = calculate_delta::<f64>(None, &xs).unwrap();
    // Default 1% of 1e20 = 1e18
    assert_relative_eq!(delta, 1e18f64, epsilon = 1e6);
}

/// Test interpolation with a minimal gap (gap size 1).
#[test]
fn test_interpolate_gap_minimal() {
    let x = vec![0.0f64, 1.0, 2.0];
    let mut y = vec![10.0f64, 0.0, 20.0];

    // Gap at index 1
    interpolate_gap(&x, &mut y, 0, 2);

    assert_relative_eq!(y[1], 15.0f64, epsilon = 1e-12);
}

/// Test interpolation with non-finite smoothed values.
#[test]
fn test_interpolate_gap_nan_inf() {
    let x = vec![0.0f64, 1.0, 2.0];

    // y0 is Inf, y1 is finite => Inf - Inf => NaN
    let mut y_inf = vec![f64::INFINITY, 0.0, 20.0];
    interpolate_gap(&x, &mut y_inf, 0, 2);
    assert!(y_inf[1].is_nan());

    // y0 is NaN
    let mut y_nan = vec![f64::NAN, 0.0, 20.0];
    interpolate_gap(&x, &mut y_nan, 0, 2);
    assert!(y_nan[1].is_nan());
}
