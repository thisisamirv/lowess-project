#![cfg(feature = "dev")]
//! Tests for Median Absolute Deviation (MAD) computation.
//!
//! These tests verify the MAD calculation used in LOWESS for:
//! - Robust scale estimation
//! - Outlier detection in robustness iterations
//! - Residual normalization
//!
//! ## Test Organization
//!
//! 1. **Basic Computation** - MAD calculation for various data sizes
//! 2. **Edge Cases** - Empty, single, and special inputs
//! 3. **Statistical Properties** - Correctness and robustness

use approx::assert_relative_eq;

use lowess::internals::math::scaling::ScalingMethod;

// ============================================================================
// Basic MAD Computation Tests
// ============================================================================

/// Test MAD computation with even-length input.
///
/// Verifies correct median and MAD calculation.
#[test]
fn test_mad_even_length() {
    // Even-length: [1, 2, 3, 4]
    // Median = (2 + 3) / 2 = 2.5
    // Deviations: [1.5, 0.5, 0.5, 1.5]
    // MAD = median([0.5, 0.5, 1.5, 1.5]) = 1.0
    let mut residuals = vec![1.0f64, 2.0, 3.0, 4.0];
    let mad = ScalingMethod::MAD.compute(&mut residuals);

    assert_relative_eq!(mad, 1.0, epsilon = 1e-12);
}

/// Test MAD computation with odd-length input.
///
/// Verifies correct median and MAD calculation.
#[test]
fn test_mad_odd_length() {
    // Odd-length: [1, 2, 3]
    // Median = 2
    // Deviations: [1, 0, 1]
    // MAD = median([0, 1, 1]) = 1
    let mut residuals = vec![1.0f64, 2.0, 3.0];
    let mad = ScalingMethod::MAD.compute(&mut residuals);

    assert_relative_eq!(mad, 1.0, epsilon = 1e-12);
}

/// Test MAD with identical values.
///
/// Verifies that MAD is zero when all values are the same.
#[test]
fn test_mad_identical_values() {
    let mut residuals = vec![5.0f64; 10];
    let mad = ScalingMethod::MAD.compute(&mut residuals);

    // MAD should be zero for identical values
    assert_relative_eq!(mad, 0.0, epsilon = 1e-12);
}

/// Test MAD with symmetric distribution.
///
/// Verifies MAD calculation for symmetric data.
#[test]
fn test_mad_symmetric_distribution() {
    // Symmetric around 0: [-2, -1, 0, 1, 2]
    // Median = 0
    // Deviations: [2, 1, 0, 1, 2]
    // MAD = median([0, 1, 1, 2, 2]) = 1
    let mut residuals = vec![-2.0f64, -1.0, 0.0, 1.0, 2.0];
    let mad = ScalingMethod::MAD.compute(&mut residuals);

    assert_relative_eq!(mad, 1.0, epsilon = 1e-12);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

/// Test MAD with two values.
///
/// Verifies correct calculation for minimal non-trivial case.
#[test]
fn test_mad_two_values() {
    // Two values: [1, 3]
    // Median = (1 + 3) / 2 = 2
    // Deviations: [1, 1]
    // MAD = median([1, 1]) = 1
    let mut residuals = vec![1.0f64, 3.0];
    let mad = ScalingMethod::MAD.compute(&mut residuals);

    assert_relative_eq!(mad, 1.0, epsilon = 1e-12);
}

// ============================================================================
// Statistical Properties Tests
// ============================================================================

/// Test MAD with outliers.
///
/// Verifies that MAD is robust to outliers.
#[test]
fn test_mad_with_outliers() {
    // Data with outlier: [1, 2, 3, 4, 100]
    // Median = 3
    // Deviations: [2, 1, 0, 1, 97]
    // MAD = median([0, 1, 1, 2, 97]) = 1
    let mut residuals = vec![1.0f64, 2.0, 3.0, 4.0, 100.0];
    let mad = ScalingMethod::MAD.compute(&mut residuals);

    // MAD should be robust to the outlier
    assert_relative_eq!(mad, 1.0, epsilon = 1e-12);
}

/// Test MAD with negative values.
///
/// Verifies that MAD handles negative values correctly.
#[test]
fn test_mad_negative_values() {
    // All negative: [-4, -3, -2, -1]
    // Median = (-3 + -2) / 2 = -2.5
    // Deviations: [1.5, 0.5, 0.5, 1.5]
    // MAD = 1.0
    let mut residuals = vec![-4.0f64, -3.0, -2.0, -1.0];
    let mad = ScalingMethod::MAD.compute(&mut residuals);

    assert_relative_eq!(mad, 1.0, epsilon = 1e-12);
}

/// Test MAD with mixed positive and negative values.
///
/// Verifies correct handling of mixed signs.
#[test]
fn test_mad_mixed_signs() {
    // Mixed: [-3, -1, 1, 3]
    // Median = (-1 + 1) / 2 = 0
    // Deviations: [3, 1, 1, 3]
    // MAD = median([1, 1, 3, 3]) = 2
    let mut residuals = vec![-3.0f64, -1.0, 1.0, 3.0];
    let mad = ScalingMethod::MAD.compute(&mut residuals);

    assert_relative_eq!(mad, 2.0, epsilon = 1e-12);
}

/// Test MAD with very small values.
///
/// Verifies numerical stability with small numbers.
#[test]
fn test_mad_small_values() {
    let mut residuals = vec![1e-10f64, 2e-10, 3e-10, 4e-10];
    let mad = ScalingMethod::MAD.compute(&mut residuals);

    // Should be approximately 1e-10
    assert!(mad > 0.0, "MAD should be positive");
    assert!(mad < 1e-9, "MAD should be small");
    assert!(mad.is_finite(), "MAD should be finite");
}

/// Test MAD with large values.
///
/// Verifies numerical stability with large numbers.
#[test]
fn test_mad_large_values() {
    let mut residuals = vec![1e10f64, 2e10, 3e10, 4e10];
    let mad = ScalingMethod::MAD.compute(&mut residuals);

    // Should be approximately 1e10
    assert!(mad > 0.0, "MAD should be positive");
    assert!(mad.is_finite(), "MAD should be finite");
    assert_relative_eq!(mad, 1e10, epsilon = 1e8);
}

/// Test MAD is scale-invariant.
///
/// Verifies that MAD(k*x) = k*MAD(x) for positive k.
#[test]
fn test_mad_scale_invariance() {
    let residuals = vec![1.0f64, 2.0, 3.0, 4.0];
    let mad1 = ScalingMethod::MAD.compute(&mut residuals.clone());

    let mut scaled: Vec<f64> = residuals.iter().map(|&x| x * 10.0).collect();
    let mad2 = ScalingMethod::MAD.compute(&mut scaled);

    assert_relative_eq!(mad2, mad1 * 10.0, epsilon = 1e-10);
}

/// Test MAD with unsorted input.
///
/// Verifies that MAD works correctly regardless of input order.
#[test]
fn test_mad_unsorted_input() {
    // Unsorted: [4, 1, 3, 2]
    // Should give same result as sorted [1, 2, 3, 4]
    let mut unsorted = vec![4.0f64, 1.0, 3.0, 2.0];
    let mad_unsorted = ScalingMethod::MAD.compute(&mut unsorted);

    let mut sorted = vec![1.0f64, 2.0, 3.0, 4.0];
    let mad_sorted = ScalingMethod::MAD.compute(&mut sorted);

    assert_relative_eq!(mad_unsorted, mad_sorted, epsilon = 1e-12);
}

/// Test MAD with zeros.
///
/// Verifies handling of zero values.
#[test]
fn test_mad_with_zeros() {
    // Data with zeros: [0, 0, 1, 2]
    // Median = (0 + 1) / 2 = 0.5
    // Deviations: [0.5, 0.5, 0.5, 1.5]
    // MAD = median([0.5, 0.5, 0.5, 1.5]) = 0.5
    let mut residuals = vec![0.0f64, 0.0, 1.0, 2.0];
    let mad = ScalingMethod::MAD.compute(&mut residuals);

    assert_relative_eq!(mad, 0.5, epsilon = 1e-12);
}
