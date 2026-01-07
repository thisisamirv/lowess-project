#![cfg(feature = "dev")]
//! Tests for local regression algorithms.
//!
//! These tests verify the core regression utilities used in LOWESS for:
//! - Local weighted least squares (WLS) fitting
//! - Weight parameter computation
//! - Zero-weight fallback strategies
//! - Boundary condition handling
//!
//! ## Test Organization
//!
//! 1. **Weight Parameters** - WeightParams construction and validation
//! 2. **Local WLS** - Weighted least squares fitting
//! 3. **Fit Point** - Complete point fitting with various scenarios
//! 4. **Zero Weight Fallbacks** - Handling degenerate cases
//! 5. **Boundary Conditions** - Edge cases and invalid inputs

use approx::assert_relative_eq;
use num_traits::Float;

use lowess::internals::algorithms::regression::{
    LinearFit, RegressionContext, WLSSolver, WeightParams, ZeroWeightFallback,
};

use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::primitives::window::Window;

// ============================================================================
// Helper Functions
// ============================================================================

fn compute_weighted_sum<T: Float>(values: &[T], weights: &[T], left: usize, right: usize) -> T {
    let mut sum = T::zero();
    for j in left..=right {
        sum = sum + values[j] * weights[j];
    }
    sum
}

fn local_wls_helper<T: Float + WLSSolver>(
    x: &[T],
    y: &[T],
    weights: &[T],
    left: usize,
    right: usize,
    x_current: T,
    window_radius: T,
) -> T {
    let window_x = &x[left..=right];
    let window_y = &y[left..=right];
    let window_weights = &weights[left..=right];

    let model = LinearFit::fit_wls(window_x, window_y, window_weights, window_radius);
    model.predict(x_current)
}

// ============================================================================
// Weight Parameters Tests
// ============================================================================

/// Test WeightParams construction and computation.
///
/// Verifies that h1 and h9 are correctly computed from window radius.
#[test]
fn test_weight_params_construction() {
    let wp = WeightParams::new(1.0f64, 2.0f64, true);

    assert_relative_eq!(wp.window_radius, 2.0f64, epsilon = 1e-12);
    // h1 = 0.001 * radius
    assert_relative_eq!(wp.h1, 0.001f64 * 2.0f64, epsilon = 1e-12);
    // h9 = 0.999 * radius
    assert_relative_eq!(wp.h9, 0.999f64 * 2.0f64, epsilon = 1e-12);
}

/// Test WeightParams with non-positive bandwidth.
///
/// Verifies behavior differs between debug and release builds:
/// - Debug: panics on assertion
/// - Release: clamps to small positive value
#[test]
fn test_weight_params_nonpositive_bandwidth() {
    if cfg!(debug_assertions) {
        // In debug, should panic
        let result = std::panic::catch_unwind(|| {
            let _ = WeightParams::new(1.0f64, 0.0f64, false);
        });
        assert!(
            result.is_err(),
            "Expected panic in debug build for non-positive bandwidth"
        );
    } else {
        // In release, should clamp to small positive value
        let wp = WeightParams::new(1.0f64, 0.0f64, false);
        assert!(
            wp.window_radius > 0.0f64,
            "Radius should be clamped to positive value"
        );
        assert_relative_eq!(wp.h1, 0.001f64 * wp.window_radius, epsilon = 1e-12);
        assert_relative_eq!(wp.h9, 0.999f64 * wp.window_radius, epsilon = 1e-12);
    }
}

// ============================================================================
// Local WLS Tests
// ============================================================================

/// Test local_wls with degenerate bandwidth.
///
/// Verifies that when bandwidth <= 0, weighted average is returned.
#[test]
fn test_local_wls_degenerate_bandwidth() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![10.0f64, 20.0, 30.0];
    let weights = vec![1.0f64, 1.0, 1.0];

    // bandwidth <= 0 triggers early return of weighted average
    let result = local_wls_helper(&x, &y, &weights, 0, 2, 1.0f64, 0.0f64);
    let sum_w = weights.iter().fold(0.0, |acc, &w| acc + w);
    let expected = compute_weighted_sum(&y, &weights, 0, 2) / sum_w;

    assert_relative_eq!(result, expected, epsilon = 1e-12);
}

/// Test local_wls with small denominator (identical x values).
///
/// Verifies fallback to weighted average when denominator is too small.
#[test]
fn test_local_wls_small_denominator() {
    // All x values identical => denom zero => fallback to average
    let x = vec![1.0f64, 1.0, 1.0];
    let y = vec![2.0f64, 3.0, 4.0];
    let weights = vec![1.0f64, 1.0, 1.0];

    let result = local_wls_helper(&x, &y, &weights, 0, 2, 1.0f64, 1.0f64);

    // Average of y = 3.0
    assert_relative_eq!(result, 3.0f64, epsilon = 1e-12);
}

/// Test local_wls recovers correct linear slope.
///
/// Verifies that WLS correctly fits a simple linear relationship.
#[test]
fn test_local_wls_linear_fit() {
    // y = 2 * x, expect fitted at x_current = 1.0 => y = 2.0
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![0.0f64, 2.0, 4.0];
    let weights = vec![1.0f64, 1.0, 1.0];

    let result = local_wls_helper(&x, &y, &weights, 0, 2, 1.0f64, 1.0f64);

    assert_relative_eq!(result, 2.0f64, epsilon = 1e-12);
}

// ============================================================================
// Fit Point Tests
// ============================================================================

/// Test fit_point with degenerate bandwidth.
///
/// Verifies that when all x values are identical, weighted average is returned.
#[test]
fn test_fit_point_degenerate_bandwidth() {
    // All x identical => bandwidth computed will be zero
    let x = vec![1.0f64, 1.0, 1.0];
    let y = vec![10.0f64, 20.0, 30.0];
    let mut weights = vec![1.0f64, 1.0, 1.0];
    let robustness = vec![1.0f64; 3];
    let window = Window { left: 0, right: 2 };

    let mut ctx = RegressionContext {
        x: &x,
        y: &y,
        weights: &mut weights,
        idx: 1usize,
        window,
        use_robustness: false,
        robustness_weights: &robustness,
        weight_function: WeightFunction::Tricube,
        zero_weight_fallback: ZeroWeightFallback::UseLocalMean,
    };

    let result = ctx.fit().expect("Should return weighted average");
    let weights_ones = vec![1.0f64; 3];
    let sum_w = 3.0f64;
    let expected = compute_weighted_sum(&y, &weights_ones, 0, 2) / sum_w;

    assert_relative_eq!(result, expected, epsilon = 1e-12);
}

// ============================================================================
// Zero Weight Fallback Tests
// ============================================================================

/// Test zero weight fallback: UseLocalMean.
///
/// Verifies that when all weights are zero, local mean is returned.
#[test]
fn test_zero_weight_fallback_local_mean() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![10.0f64, 20.0, 30.0];
    let mut weights = vec![0.0f64; 3];
    let robustness = vec![0.0f64; 3]; // Zero robustness => zero total weight
    let window = Window { left: 0, right: 2 };

    let mut ctx = RegressionContext {
        x: &x,
        y: &y,
        weights: &mut weights,
        idx: 1usize,
        window,
        use_robustness: true,
        robustness_weights: &robustness,
        weight_function: WeightFunction::Tricube,
        zero_weight_fallback: ZeroWeightFallback::UseLocalMean,
    };

    let result = ctx.fit().expect("Should return local mean");

    // Mean of [10, 20, 30] = 20
    assert_relative_eq!(result, 20.0f64, epsilon = 1e-12);
}

/// Test zero weight fallback: ReturnOriginal.
///
/// Verifies that when all weights are zero, original y[idx] is returned.
#[test]
fn test_zero_weight_fallback_return_original() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![10.0f64, 20.0, 30.0];
    let mut weights = vec![0.0f64; 3];
    let robustness = vec![0.0f64; 3];
    let window = Window { left: 0, right: 2 };

    let mut ctx = RegressionContext {
        x: &x,
        y: &y,
        weights: &mut weights,
        idx: 2usize,
        window,
        use_robustness: true,
        robustness_weights: &robustness,
        weight_function: WeightFunction::Tricube,
        zero_weight_fallback: ZeroWeightFallback::ReturnOriginal,
    };

    let result = ctx.fit().expect("Should return original y");

    assert_relative_eq!(result, 30.0f64, epsilon = 1e-12);
}

/// Test zero weight fallback: ReturnNone.
///
/// Verifies that when all weights are zero, None is returned.
#[test]
fn test_zero_weight_fallback_return_none() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![10.0f64, 20.0, 30.0];
    let mut weights = vec![0.0f64; 3];
    let robustness = vec![0.0f64; 3];
    let window = Window { left: 0, right: 2 };

    let mut ctx = RegressionContext {
        x: &x,
        y: &y,
        weights: &mut weights,
        idx: 0usize,
        window,
        use_robustness: true,
        robustness_weights: &robustness,
        weight_function: WeightFunction::Tricube,
        zero_weight_fallback: ZeroWeightFallback::ReturnNone,
    };

    let result = ctx.fit();

    assert!(result.is_none(), "Should return None for zero weights");
}

/// Test degenerate bandwidth with zero weights returns original y.
///
/// Verifies combined degenerate case handling.
#[test]
fn test_degenerate_bandwidth_zero_weights() {
    // Identical x within window => bandwidth == 0
    let x = vec![1.0f64, 1.0, 1.0];
    let y = vec![5.0f64, 6.0, 7.0];
    let mut weights = vec![0.0f64, 0.0f64, 0.0f64];
    let robustness = vec![1.0f64; 3];
    let window = Window { left: 0, right: 2 };

    let mut ctx = RegressionContext {
        x: &x,
        y: &y,
        weights: &mut weights,
        idx: 1usize,
        window,
        use_robustness: false,
        robustness_weights: &robustness,
        weight_function: WeightFunction::Tricube,
        zero_weight_fallback: ZeroWeightFallback::ReturnOriginal,
    };

    let result = ctx
        .fit()
        .expect("Should return original y when weights sum to zero in degenerate bandwidth");

    assert_relative_eq!(result, 6.0f64, epsilon = 1e-12);
}

// ============================================================================
// Boundary Conditions Tests
// ============================================================================

/// Test fit_point with invalid index.
///
/// Verifies that out-of-bounds index returns None.
#[test]
fn test_fit_point_invalid_index() {
    let x = vec![0.0f64, 1.0];
    let y = vec![10.0f64, 20.0];
    let mut weights = vec![1.0f64, 1.0f64];
    let robustness = vec![1.0f64, 1.0f64];
    let window = Window { left: 0, right: 1 };

    // idx out of bounds (equal to n) => None
    let mut ctx = RegressionContext {
        x: &x,
        y: &y,
        weights: &mut weights,
        idx: 2usize,
        window,
        use_robustness: false,
        robustness_weights: &robustness,
        weight_function: WeightFunction::Tricube,
        zero_weight_fallback: ZeroWeightFallback::ReturnNone,
    };

    let result = ctx.fit();

    assert!(result.is_none(), "Out-of-bounds index should return None");
}

/// Test fit_point with invalid window bounds.
///
/// Verifies that invalid window boundaries return None.
#[test]
fn test_fit_point_invalid_window() {
    let x = vec![0.0f64, 1.0];
    let y = vec![10.0f64, 20.0];
    let mut weights = vec![1.0f64, 1.0f64];
    let robustness = vec![1.0f64, 1.0f64];

    // left >= n
    let window_bad_left = Window { left: 2, right: 2 };
    let mut ctx_left = RegressionContext {
        x: &x,
        y: &y,
        weights: &mut weights,
        idx: 1usize,
        window: window_bad_left,
        use_robustness: false,
        robustness_weights: &robustness,
        weight_function: WeightFunction::Tricube,
        zero_weight_fallback: ZeroWeightFallback::ReturnNone,
    };

    assert!(
        ctx_left.fit().is_none(),
        "Invalid left bound should return None"
    );

    // right >= n
    let window_bad_right = Window { left: 0, right: 2 };
    let mut ctx_right = RegressionContext {
        x: &x,
        y: &y,
        weights: &mut weights,
        idx: 0usize,
        window: window_bad_right,
        use_robustness: false,
        robustness_weights: &robustness,
        weight_function: WeightFunction::Tricube,
        zero_weight_fallback: ZeroWeightFallback::ReturnNone,
    };

    assert!(
        ctx_right.fit().is_none(),
        "Invalid right bound should return None"
    );
}

/// Test with various weight functions.
///
/// Verifies that different kernel functions work correctly.
#[test]
fn test_fit_point_various_kernels() {
    let kernels = vec![
        WeightFunction::Tricube,
        WeightFunction::Epanechnikov,
        WeightFunction::Gaussian,
        WeightFunction::Biweight,
    ];

    for kernel in kernels {
        let x = vec![0.0f64, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mut weights = vec![1.0f64; 5];
        let robustness = vec![1.0f64; 5];
        let window = Window { left: 0, right: 4 };

        let mut ctx = RegressionContext {
            x: &x,
            y: &y,
            weights: &mut weights,
            idx: 2usize,
            window,
            use_robustness: false,
            robustness_weights: &robustness,
            weight_function: kernel,
            zero_weight_fallback: ZeroWeightFallback::UseLocalMean,
        };

        let result = ctx.fit();
        assert!(
            result.is_some(),
            "Kernel {:?} should produce valid result",
            kernel
        );
        assert!(
            result.unwrap().is_finite(),
            "Result should be finite for kernel {:?}",
            kernel
        );
    }
}

/// Test with robustness weights enabled.
///
/// Verifies that robustness weights are correctly applied.
#[test]
fn test_fit_point_with_robustness() {
    let x = vec![0.0f64, 1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0f64, 2.0, 100.0, 4.0, 5.0]; // Point 2 is outlier
    let mut weights = vec![1.0f64; 5];
    let robustness = vec![1.0f64, 1.0, 0.1, 1.0, 1.0]; // Downweight outlier
    let window = Window { left: 0, right: 4 };

    let mut ctx = RegressionContext {
        x: &x,
        y: &y,
        weights: &mut weights,
        idx: 2usize,
        window,
        use_robustness: true,
        robustness_weights: &robustness,
        weight_function: WeightFunction::Tricube,
        zero_weight_fallback: ZeroWeightFallback::UseLocalMean,
    };

    let result = ctx.fit();
    assert!(result.is_some(), "Should produce valid result");

    // Result should be closer to linear trend than to outlier
    let fitted = result.unwrap();
    assert!(
        (fitted - 3.0).abs() < 50.0,
        "Fitted value should be closer to trend than outlier"
    );
}

/// Test local_wls when only one point has a non-zero weight.
#[test]
fn test_local_wls_single_weight() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![10.0f64, 20.0, 30.0];
    let weights = vec![1.0f64, 0.0, 0.0];

    // With only one point having weight, denominator for slope is zero.
    // Should fallback to weighted average (which is y[0] since others are weighted 0).
    let result = local_wls_helper(&x, &y, &weights, 0, 2, 0.0f64, 1.0f64);
    assert_relative_eq!(result, 10.0f64, epsilon = 1e-12);
}

/// Test local_wls with exactly two points.
#[test]
fn test_local_wls_two_points() {
    let x = vec![0.0f64, 2.0];
    let y = vec![10.0f64, 20.0];
    let weights = vec![1.0f64, 1.0];

    // Fit at x = 1.0. Linear interpolation between (0,10) and (2,20) is 15.0
    let result = local_wls_helper(&x, &y, &weights, 0, 1, 1.0f64, 1.0f64);
    assert_relative_eq!(result, 15.0f64, epsilon = 1e-12);
}

/// Test local_wls with perfectly correlated data.
#[test]
fn test_local_wls_perfect_correlation() {
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..10).map(|i| i as f64 * 3.0 + 5.0).collect();
    let weights = vec![1.0f64; 10];

    // Fit at x = 4.5. Expect 3.0 * 4.5 + 5.0 = 13.5 + 5.0 = 18.5
    let result = local_wls_helper(&x, &y, &weights, 0, 9, 4.5f64, 5.0f64);
    assert_relative_eq!(result, 18.5f64, epsilon = 1e-12);
}

/// Test local_wls with extreme values.
#[test]
fn test_local_wls_extreme_values() {
    let x = vec![0.0f64, 1e10];
    let y = vec![0.0f64, 1e10];
    let weights = vec![1.0f64, 1.0];

    // Slope is 1.0. Fit at 5e9. Expect 5e9
    let result = local_wls_helper(&x, &y, &weights, 0, 1, 5e9f64, 1e10f64);
    assert_relative_eq!(result, 5e9f64, epsilon = 1e-2); // Relaxed epsilon for large values
}
