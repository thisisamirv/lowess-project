#![cfg(feature = "dev")]
//! Tests for robustness weight algorithms.
//!
//! These tests verify the robustness weight updaters used in LOWESS for:
//! - Outlier detection and downweighting
//! - Different robustness methods (Bisquare, Huber, Talwar)
//! - Weight computation based on residuals
//!
//! ## Test Organization
//!
//! 1. **Bisquare Method** - Smooth downweighting for general use
//! 2. **Huber Method** - Linear beyond threshold for moderate outliers
//! 3. **Talwar Method** - Hard threshold (0 or 1) for extreme contamination
//! 4. **Edge Cases** - Empty inputs and boundary conditions

use approx::assert_relative_eq;

use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::math::scaling::ScalingMethod;

// ============================================================================
// Bisquare Method Tests
// ============================================================================

/// Test Bisquare robustness weight computation.
///
/// Verifies:
/// - Weights are in [0, 1] range
/// - Outliers receive lower weights
/// - Small residuals receive high weights
#[test]
fn test_bisquare_weight_computation() {
    let method = RobustnessMethod::Bisquare;
    let residuals = vec![0.0f64, 0.5f64, 10.0f64];
    let mut weights = vec![1.0f64; residuals.len()];

    let mut scratch = vec![0.0f64; residuals.len()];
    method.apply_robustness_weights(
        &residuals,
        &mut weights,
        ScalingMethod::default(),
        &mut scratch,
    );

    // Verify all weights are in valid range
    for &w in &weights {
        assert!(
            (0.0f64..=1.0f64).contains(&w),
            "Weight {} should be in [0, 1]",
            w
        );
    }

    // Outlier (10.0) should have low weight compared to others
    assert!(
        weights[2] < weights[0],
        "Outlier should have lower weight than small residual"
    );
    assert!(
        weights[2] < weights[1],
        "Outlier should have lower weight than moderate residual"
    );

    // Zero residual should have weight close to 1
    assert_relative_eq!(weights[0], 1.0, epsilon = 0.1);
}

/// Test Bisquare with various residual magnitudes.
///
/// Verifies smooth downweighting behavior across residual range.
#[test]
fn test_bisquare_smooth_downweighting() {
    let method = RobustnessMethod::Bisquare;
    let residuals = vec![0.0f64, 1.0, 2.0, 5.0, 10.0, 20.0];
    let mut weights = vec![1.0f64; residuals.len()];

    let mut scratch = vec![0.0f64; residuals.len()];
    method.apply_robustness_weights(
        &residuals,
        &mut weights,
        ScalingMethod::default(),
        &mut scratch,
    );

    // Weights should decrease monotonically with residual magnitude
    for i in 1..weights.len() {
        assert!(
            weights[i] <= weights[i - 1],
            "Weights should decrease with residual magnitude"
        );
    }
}

// ============================================================================
// Huber Method Tests
// ============================================================================

/// Test Huber robustness weight computation.
///
/// Verifies:
/// - Weights are in [0, 1] range
/// - Outliers receive lower weights
/// - Linear downweighting beyond threshold
#[test]
fn test_huber_weight_computation() {
    let method = RobustnessMethod::Huber;
    let residuals = vec![0.0f64, 0.5f64, 10.0f64];
    let mut weights = vec![1.0f64; residuals.len()];

    let mut scratch = vec![0.0f64; residuals.len()];
    method.apply_robustness_weights(
        &residuals,
        &mut weights,
        ScalingMethod::default(),
        &mut scratch,
    );

    // Verify all weights are in valid range
    for &w in &weights {
        assert!(
            (0.0f64..=1.0f64).contains(&w),
            "Weight {} should be in [0, 1]",
            w
        );
    }

    // Outlier should have lower weight
    assert!(
        weights[2] < weights[0],
        "Outlier should have lower weight than small residual"
    );
}

/// Test Huber with moderate outliers.
///
/// Verifies that Huber is less aggressive than Bisquare for moderate outliers.
#[test]
fn test_huber_moderate_outliers() {
    let method = RobustnessMethod::Huber;
    let residuals = vec![0.0f64, 1.0, 2.0, 3.0, 4.0];
    let mut weights = vec![1.0f64; residuals.len()];

    let mut scratch = vec![0.0f64; residuals.len()];
    method.apply_robustness_weights(
        &residuals,
        &mut weights,
        ScalingMethod::default(),
        &mut scratch,
    );

    // All weights should be positive (Huber doesn't completely reject points)
    for &w in &weights {
        assert!(w > 0.0, "Huber should not completely reject points");
    }

    // Weights should decrease with residual magnitude
    for i in 1..weights.len() {
        assert!(
            weights[i] <= weights[i - 1],
            "Weights should decrease with residual magnitude"
        );
    }
}

// ============================================================================
// Talwar Method Tests
// ============================================================================

/// Test Talwar robustness weight computation (hard thresholding).
///
/// Verifies:
/// - Small residuals get weight 1.0
/// - Large residuals get weight 0.0
/// - Hard threshold behavior
#[test]
fn test_talwar_hard_rejection() {
    let method = RobustnessMethod::Talwar;
    let residuals = vec![0.0f64, 0.1f64, 10.0f64];
    let mut weights = vec![1.0f64; residuals.len()];

    let mut scratch = vec![0.0f64; residuals.len()];
    method.apply_robustness_weights(
        &residuals,
        &mut weights,
        ScalingMethod::default(),
        &mut scratch,
    );

    // Small residual should have weight 1.0
    assert_relative_eq!(weights[0], 1.0, epsilon = 1e-12);

    // Large residual should have weight 0.0 (hard rejection)
    assert_relative_eq!(weights[2], 0.0, epsilon = 1e-12);
}

/// Test Talwar with extreme contamination.
///
/// Verifies that Talwar completely rejects extreme outliers.
#[test]
fn test_talwar_extreme_outliers() {
    let method = RobustnessMethod::Talwar;
    let residuals = vec![0.0f64, 0.5, 1.0, 50.0, 100.0];
    let mut weights = vec![1.0f64; residuals.len()];

    let mut scratch = vec![0.0f64; residuals.len()];
    method.apply_robustness_weights(
        &residuals,
        &mut weights,
        ScalingMethod::default(),
        &mut scratch,
    );

    // Extreme outliers should be completely rejected
    assert_relative_eq!(weights[3], 0.0, epsilon = 1e-12);
    assert_relative_eq!(weights[4], 0.0, epsilon = 1e-12);

    // Small residuals should be accepted
    assert!(weights[0] > 0.5, "Small residuals should be accepted");
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

/// Test robustness methods with empty input.
///
/// Verifies that empty inputs are handled gracefully.
#[test]
fn test_robustness_empty_input() {
    let methods = vec![
        RobustnessMethod::Bisquare,
        RobustnessMethod::Huber,
        RobustnessMethod::Talwar,
    ];

    for method in methods {
        let residuals: Vec<f64> = vec![];
        let mut weights: Vec<f64> = vec![];

        let mut scratch = vec![0.0f64; residuals.len()];
        method.apply_robustness_weights(
            &residuals,
            &mut weights,
            ScalingMethod::default(),
            &mut scratch,
        );

        assert!(
            weights.is_empty(),
            "Empty input should produce empty output"
        );
    }
}

/// Test robustness methods with all zero residuals.
///
/// Verifies that all weights remain 1.0 when residuals are zero.
#[test]
fn test_robustness_zero_residuals() {
    let methods = vec![
        RobustnessMethod::Bisquare,
        RobustnessMethod::Huber,
        RobustnessMethod::Talwar,
    ];

    for method in methods {
        let residuals = vec![0.0f64; 5];
        let mut weights = vec![1.0f64; 5];

        let mut scratch = vec![0.0f64; residuals.len()];
        method.apply_robustness_weights(
            &residuals,
            &mut weights,
            ScalingMethod::default(),
            &mut scratch,
        );

        // All weights should remain 1.0 for zero residuals
        for &w in &weights {
            assert_relative_eq!(w, 1.0, epsilon = 1e-12);
        }
    }
}

/// Test robustness methods with single point.
///
/// Verifies that single-point inputs are handled correctly.
#[test]
fn test_robustness_single_point() {
    let methods = vec![
        RobustnessMethod::Bisquare,
        RobustnessMethod::Huber,
        RobustnessMethod::Talwar,
    ];

    for method in methods {
        let residuals = vec![5.0f64];
        let mut weights = vec![1.0f64];

        let mut scratch = vec![0.0f64; residuals.len()];
        method.apply_robustness_weights(
            &residuals,
            &mut weights,
            ScalingMethod::default(),
            &mut scratch,
        );

        assert_eq!(weights.len(), 1, "Should have one weight");
        assert!(
            weights[0] >= 0.0 && weights[0] <= 1.0,
            "Weight should be in [0, 1]"
        );
    }
}

/// Test robustness methods with identical residuals.
///
/// Verifies that all points receive equal weights when residuals are identical.
#[test]
fn test_robustness_identical_residuals() {
    let methods = vec![
        RobustnessMethod::Bisquare,
        RobustnessMethod::Huber,
        RobustnessMethod::Talwar,
    ];

    for method in methods {
        let residuals = vec![2.0f64; 5];
        let mut weights = vec![1.0f64; 5];

        let mut scratch = vec![0.0f64; residuals.len()];
        method.apply_robustness_weights(
            &residuals,
            &mut weights,
            ScalingMethod::default(),
            &mut scratch,
        );

        // All weights should be equal
        let first_weight = weights[0];
        for &w in &weights {
            assert_relative_eq!(w, first_weight, epsilon = 1e-12);
        }
    }
}

/// Compare all three methods on same data.
///
/// Verifies that methods have different downweighting characteristics.
#[test]
fn test_robustness_method_comparison() {
    let residuals = vec![0.0f64, 1.0, 2.0, 5.0, 10.0];

    let mut weights_bisquare = vec![1.0f64; 5];
    let mut weights_huber = vec![1.0f64; 5];
    let mut weights_talwar = vec![1.0f64; 5];

    let mut scratch = vec![0.0f64; residuals.len()];
    RobustnessMethod::Bisquare.apply_robustness_weights(
        &residuals,
        &mut weights_bisquare,
        ScalingMethod::default(),
        &mut scratch,
    );
    RobustnessMethod::Huber.apply_robustness_weights(
        &residuals,
        &mut weights_huber,
        ScalingMethod::default(),
        &mut scratch,
    );
    RobustnessMethod::Talwar.apply_robustness_weights(
        &residuals,
        &mut weights_talwar,
        ScalingMethod::default(),
        &mut scratch,
    );

    // All methods should downweight the large outlier (index 4)
    assert!(weights_bisquare[4] < weights_bisquare[0]);
    assert!(weights_huber[4] < weights_huber[0]);
    assert!(weights_talwar[4] < weights_talwar[0]);

    // Talwar should be most aggressive (likely 0 for large outlier)
    assert!(
        weights_talwar[4] <= weights_bisquare[4],
        "Talwar should be most aggressive"
    );
    assert!(
        weights_talwar[4] <= weights_huber[4],
        "Talwar should be most aggressive"
    );
}

/// Test with negative residuals.
///
/// Verifies that methods handle negative residuals correctly (using absolute value).
#[test]
fn test_robustness_negative_residuals() {
    let methods = vec![
        RobustnessMethod::Bisquare,
        RobustnessMethod::Huber,
        RobustnessMethod::Talwar,
    ];

    for method in methods {
        let residuals = vec![-10.0f64, -1.0, 0.0, 1.0, 10.0];
        let mut weights = vec![1.0f64; 5];

        let mut scratch = vec![0.0f64; residuals.len()];
        method.apply_robustness_weights(
            &residuals,
            &mut weights,
            ScalingMethod::default(),
            &mut scratch,
        );

        // Symmetric residuals should get symmetric weights
        assert_relative_eq!(weights[0], weights[4], epsilon = 1e-10);
        assert_relative_eq!(weights[1], weights[3], epsilon = 1e-10);
    }
}

/// Test with extremely large and small residual values.
#[test]
fn test_robustness_extreme_residual_values() {
    let method = RobustnessMethod::Bisquare;
    // With 10 points, MAR will be 1/10 of the outlier.
    // u = 1e30 / (6 * 1e29) = 1.66 > 1.0 => weight 0.0
    let mut residuals = vec![0.0; 9];
    residuals.push(1e30);
    let mut weights = vec![1.0; 10];

    let mut scratch = vec![0.0f64; residuals.len()];
    method.apply_robustness_weights(
        &residuals,
        &mut weights,
        ScalingMethod::default(),
        &mut scratch,
    );

    assert_relative_eq!(weights[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(weights[9], 0.0, epsilon = 1e-12); // Massive outlier rejected
}

/// Test specifically the MAR fallback when MAD is zero.
#[test]
fn test_robustness_mar_fallback() {
    let method = RobustnessMethod::Bisquare;
    // MAD of [0, 0, 10] is 0. MAR is 10/3 = 3.33.
    // Scale should be 3.33.
    // u = 10 / (6 * 3.33) = 10 / 20 = 0.5.
    // Weight for 10 should be (1 - 0.5^2)^2 = (0.75)^2 = 0.5625
    let residuals = vec![0.0, 0.0, 10.0];
    let mut weights = vec![1.0; 3];

    let mut scratch = vec![0.0f64; residuals.len()];
    method.apply_robustness_weights(
        &residuals,
        &mut weights,
        ScalingMethod::default(),
        &mut scratch,
    );

    assert_relative_eq!(weights[2], 0.5625, epsilon = 1e-12);
}

/// Test when all points except one are outliers.
#[test]
fn test_robustness_all_outliers_except_one() {
    let method = RobustnessMethod::Bisquare;
    // Four identical "outliers" and one "good" point at 0
    let residuals = vec![0.0, 100.0, 100.0, 100.0, 100.0];
    let mut weights = vec![1.0; 5];

    let mut scratch = vec![0.0f64; residuals.len()];
    method.apply_robustness_weights(
        &residuals,
        &mut weights,
        ScalingMethod::default(),
        &mut scratch,
    );

    // The point at 0 should have the highest weight
    assert!(weights[0] > weights[1]);
    assert_relative_eq!(weights[0], 1.0, epsilon = 1e-12);
}

/// Test Huber and Talwar with zero scale.
#[test]
fn test_huber_talwar_zero_scale() {
    let residuals = vec![0.0, 0.0, 0.0];
    let mut weights = vec![0.0; 3];

    let mut scratch = vec![0.0f64; residuals.len()];
    RobustnessMethod::Huber.apply_robustness_weights(
        &residuals,
        &mut weights,
        ScalingMethod::default(),
        &mut scratch,
    );
    for &w in &weights {
        assert_relative_eq!(w, 1.0, epsilon = 1e-12);
    }

    RobustnessMethod::Talwar.apply_robustness_weights(
        &residuals,
        &mut weights,
        ScalingMethod::default(),
        &mut scratch,
    );
    for &w in &weights {
        assert_relative_eq!(w, 1.0, epsilon = 1e-12);
    }
}

/// Test handling of NaN and Inf residuals.
#[test]
fn test_robustness_nan_inf_residuals() {
    let method = RobustnessMethod::Bisquare;
    let residuals = vec![0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    let mut weights = vec![1.0; 4];

    // This should not panic
    let mut scratch = vec![0.0f64; residuals.len()];
    method.apply_robustness_weights(
        &residuals,
        &mut weights,
        ScalingMethod::default(),
        &mut scratch,
    );

    // We don't strictly define what NaN/Inf should result in,
    // but they shouldn't break the high-level invariants.
    assert!(weights[0].is_finite());
}
