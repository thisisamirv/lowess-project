#![cfg(feature = "dev")]
//! Tests for kernel weight functions.
//!
//! These tests verify the mathematical kernel functions used in LOWESS for:
//! - Distance-based weighting in local regression
//! - Different smoothness profiles (Tricube, Epanechnikov, Gaussian, etc.)
//! - Bounded vs unbounded support
//! - Weight computation and window operations
//!
//! ## Test Organization
//!
//! 1. **Kernel Properties** - Names, variance, efficiency, boundedness
//! 2. **Weight Computation** - Value tests at specific points
//! 3. **Mathematical Properties** - Symmetry, boundary behavior
//! 4. **Support and Bounds** - Bounded vs unbounded kernels
//! 5. **Window Weight Computation** - Batch weight calculation

use approx::assert_relative_eq;

use lowess::internals::algorithms::regression::WeightParams;
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::math::scaling::ScalingMethod;
use lowess::internals::primitives::window::Window;

// ============================================================================
// Kernel Properties Tests
// ============================================================================

/// Test kernel properties: name, variance, efficiency.
///
/// Verifies that all kernels have valid properties.
#[test]
fn test_kernel_basic_properties() {
    let kernel = WeightFunction::Tricube;

    // Check name
    assert_eq!(kernel.name(), "Tricube");
    assert!(!kernel.name().is_empty());

    // Check numeric properties are finite
    assert!(kernel.variance().is_finite(), "Variance should be finite");
    assert!(kernel.roughness().is_finite(), "Roughness should be finite");
    assert!(
        kernel.efficiency().is_finite(),
        "Efficiency should be finite"
    );

    // Test Default trait
    let _ = WeightFunction::default();
}

/// Test kernel metadata: name, integrator, variance, roughness, efficiency, support.
#[test]
fn test_kernel_metadata_expanded() {
    let kernels = [
        WeightFunction::Tricube,
        WeightFunction::Epanechnikov,
        WeightFunction::Gaussian,
        WeightFunction::Biweight,
        WeightFunction::Triangle,
        WeightFunction::Cosine,
        WeightFunction::Uniform,
    ];

    for k in kernels {
        assert!(!k.name().is_empty());
        assert!(k.integrator() > 0.0);
        assert!(k.variance() > 0.0);
        assert!(k.roughness() > 0.0);
        assert!(k.efficiency() > 0.0);

        // Test support
        if k == WeightFunction::Gaussian {
            assert!(k.support().is_none());
        } else {
            assert_eq!(k.support(), Some((-1.0, 1.0)));
        }

        // Test weight computation at boundaries
        assert!(k.compute_weight(0.0) >= 0.0);
        if k != WeightFunction::Gaussian {
            assert_eq!(k.compute_weight(1.1), 0.0);
            assert_eq!(k.compute_weight(1.0), 0.0);
        }
    }
}

/// Test that all kernels have valid variance and efficiency.
///
/// Verifies invariants across all kernel types.
#[test]
fn test_all_kernels_valid_properties() {
    let kernels = [
        WeightFunction::Cosine,
        WeightFunction::Epanechnikov,
        WeightFunction::Gaussian,
        WeightFunction::Biweight,
        WeightFunction::Triangle,
        WeightFunction::Tricube,
        WeightFunction::Uniform,
    ];

    for &kernel in kernels.iter() {
        let var = kernel.variance();
        assert!(
            var > 0.0 && var.is_finite(),
            "{} variance should be positive and finite",
            kernel.name()
        );

        let eff = kernel.efficiency();
        assert!(
            eff > 0.0 && eff.is_finite(),
            "{} efficiency should be positive and finite",
            kernel.name()
        );
    }
}

// ============================================================================
// Weight Computation Tests
// ============================================================================

/// Test weight computation at specific points.
///
/// Verifies correct values at u=0, midpoints, and boundaries.
#[test]
fn test_weight_computation_values() {
    // Tricube at 0 => 1
    let tricube = WeightFunction::Tricube;
    assert_relative_eq!(tricube.compute_weight(0.0f64), 1.0f64, epsilon = 1e-12);

    // Epanechnikov at 0.5 => 1 - 0.25 = 0.75
    let epan = WeightFunction::Epanechnikov;
    assert_relative_eq!(epan.compute_weight(0.5f64), 0.75f64, epsilon = 1e-12);

    // Triangle at 0.2 => 0.8
    let triangle = WeightFunction::Triangle;
    assert_relative_eq!(triangle.compute_weight(0.2f64), 0.8f64, epsilon = 1e-12);

    // Bounded kernels return zero at |u| >= 1
    assert_eq!(tricube.compute_weight(1.0f64), 0.0f64);
    assert_eq!(epan.compute_weight(-1.0f64), 0.0f64);
}

/// Test Gaussian kernel values.
///
/// Verifies that Gaussian returns positive finite values.
#[test]
fn test_gaussian_weight_values() {
    let gaussian = WeightFunction::Gaussian;

    // At u=0, should return positive value
    let val = gaussian.compute_weight(0.0f64);
    assert!(
        val > 0.0 && val.is_finite(),
        "Gaussian at 0 should be positive"
    );

    // At large u, should still return small positive value
    let val_large = gaussian.compute_weight(1000.0f64);
    assert!(
        val_large > 0.0 && val_large.is_finite(),
        "Gaussian at large u should be positive and finite"
    );
}

/// Test Cosine kernel formula.
///
/// Verifies that Cosine kernel uses correct formula.
#[test]
fn test_cosine_formula() {
    let cosine = WeightFunction::Cosine;
    let u = 0.5f64;
    let computed = cosine.compute_weight(u);
    let expected = (std::f64::consts::FRAC_PI_2 * u).cos();

    assert_relative_eq!(computed, expected, epsilon = 1e-12);
}

// ============================================================================
// Mathematical Properties Tests
// ============================================================================

/// Test that all kernels are even functions.
///
/// Verifies K(u) = K(-u) for all kernels.
#[test]
fn test_kernels_symmetry() {
    let kernels = [
        WeightFunction::Cosine,
        WeightFunction::Epanechnikov,
        WeightFunction::Gaussian,
        WeightFunction::Biweight,
        WeightFunction::Triangle,
        WeightFunction::Tricube,
        WeightFunction::Uniform,
    ];

    let u = 0.37f64;
    for &kernel in kernels.iter() {
        let pos = kernel.compute_weight(u);
        let neg = kernel.compute_weight(-u);

        // Kernels should be symmetric: K(u) = K(-u)
        assert_relative_eq!(pos, neg, epsilon = 1e-12);
    }
}

/// Test bounded kernels return zero at boundary.
///
/// Verifies that K(1) = 0 for bounded kernels.
#[test]
fn test_bounded_kernels_boundary() {
    let bounded_kernels = [
        WeightFunction::Tricube,
        WeightFunction::Biweight,
        WeightFunction::Epanechnikov,
        WeightFunction::Triangle,
        WeightFunction::Cosine,
        WeightFunction::Uniform,
    ];

    for &kernel in bounded_kernels.iter() {
        let val = kernel.compute_weight(1.0f64);
        // Bounded kernels should be zero at boundary
        assert_relative_eq!(val, 0.0f64, epsilon = 1e-12);
    }
}

/// Test kernels work with f32 generics.
///
/// Verifies that kernels support both f32 and f64.
#[test]
fn test_kernel_generic_floats() {
    let kernel = WeightFunction::Biweight;
    let val_f32 = kernel.compute_weight(0.3f32);
    let val_f64 = kernel.compute_weight(0.3f64);

    // Cast f32 to f64 for comparison with relaxed tolerance
    assert_relative_eq!(val_f32 as f64, val_f64, epsilon = 1e-6);
}

// ============================================================================
// Support and Bounds Tests
// ============================================================================

/// Test support and boundedness properties.
///
/// Verifies that bounded kernels have finite support.
#[test]
fn test_kernel_support() {
    let tricube = WeightFunction::Tricube;
    assert!(
        tricube.support().is_some(),
        "Tricube should have finite support"
    );

    let gaussian = WeightFunction::Gaussian;
    assert!(
        gaussian.support().is_none(),
        "Gaussian should have infinite support"
    );
}

// ============================================================================
// Window Weight Computation Tests
// ============================================================================

/// Test compute_window_weights with no points within h9.
///
/// Verifies that when all points are outside the window, zero sum is returned.
#[test]
fn test_window_weights_no_points_in_range() {
    let x = vec![1.0, 2.0, 3.0];
    let weights = &mut [0.0f64; 3];
    let params = WeightParams::new(5.0, 1.0, false);

    let kernel = WeightFunction::Tricube;
    let (sum, rightmost) = kernel.compute_window_weights(
        &x,
        0,
        2,
        params.x_current,
        params.window_radius,
        params.h1,
        params.h9,
        weights,
    );

    assert_eq!(sum, 0.0, "Sum should be zero when no points in range");
    assert_eq!(rightmost, 0, "Rightmost should be left index");
    assert_eq!(weights[0], 0.0, "Weights should be zero");
}

/// Test compute_window_weights respects bandwidth cutoff.
///
/// Verifies that points beyond h9 are excluded.
#[test]
fn test_window_weights_bandwidth_cutoff() {
    let x = vec![0.0, 1.0, 10.0];
    let weights = &mut [0.0f64; 3];
    let params = WeightParams::new(0.0, 2.0, false);

    let kernel = WeightFunction::Tricube;
    let (_sum, rightmost) = kernel.compute_window_weights(
        &x,
        0,
        2,
        params.x_current,
        params.window_radius,
        params.h1,
        params.h9,
        weights,
    );

    // Points within bandwidth should have positive weights
    assert_eq!(weights[0], 1.0, "Point at center should have weight 1");
    assert!(
        weights[1] > 0.0,
        "Point within bandwidth should have positive weight"
    );

    // Point far outside bandwidth should have zero weight
    assert_eq!(
        weights[2], 0.0,
        "Point outside bandwidth should have zero weight"
    );
    assert_eq!(rightmost, 1, "Rightmost should be last included point");
}

/// Test compute_window_weights with degenerate bandwidth.
///
/// Verifies that zero bandwidth returns zero sum.
#[test]
fn test_window_weights_degenerate_bandwidth() {
    let x = vec![0.0f64, 1.0];
    let window = Window { left: 0, right: 1 };

    // Zero bandwidth (degenerate case)
    let params = WeightParams {
        x_current: 0.0f64,
        window_radius: 0.0f64,
        h1: 0.0f64,
        h9: 0.0f64,
    };

    let mut weights = vec![1.0f64; 2];
    let kernel = WeightFunction::Tricube;

    let (sum, rightmost) = kernel.compute_window_weights(
        &x,
        window.left,
        window.right,
        params.x_current,
        params.window_radius,
        params.h1,
        params.h9,
        &mut weights,
    );

    // Sum should be zero for degenerate bandwidth
    assert_relative_eq!(sum, 0.0f64, epsilon = 1e-12);
    assert_eq!(rightmost, window.left, "Rightmost should equal left");

    // Weights should be zeroed
    assert_relative_eq!(weights[0], 0.0f64, epsilon = 1e-12);
    assert_relative_eq!(weights[1], 0.0f64, epsilon = 1e-12);
}

/// Test compute_window_weights skips left points and respects h1.
///
/// Verifies that points outside the lower bound are skipped.
#[test]
fn test_window_weights_skip_left_points() {
    let x = vec![-10.0f64, 0.0f64, 1.0f64, 2.0f64];
    let window = Window { left: 0, right: 3 };

    // x_current = 1.0, bandwidth = 1.0
    let mut params = WeightParams::new(1.0f64, 1.0f64, true);
    params.h9 = 0.5f64; // lower_bound = 0.5 => skip indices 0 and 1
    params.h1 = 0.01f64; // Center point gets weight 1

    let mut weights = vec![0.0f64; 4];
    let kernel = WeightFunction::Tricube;

    let (sum, rightmost) = kernel.compute_window_weights(
        &x,
        window.left,
        window.right,
        params.x_current,
        params.window_radius,
        params.h1,
        params.h9,
        &mut weights,
    );

    // Left points should be skipped
    assert_eq!(weights[0], 0.0, "Far left point should be skipped");
    assert_eq!(weights[1], 0.0, "Left point should be skipped");

    // Center point should have weight 1 (distance < h1)
    assert_relative_eq!(weights[2], 1.0f64, epsilon = 1e-12);

    // Rightmost should be at least index 2
    assert!(rightmost >= 2, "Rightmost should include center point");

    // Sum should equal manual sum
    let manual_sum: f64 = weights.iter().copied().sum();
    assert_relative_eq!(sum, manual_sum, epsilon = 1e-12);
}

/// Test window weights with various kernels.
///
/// Verifies that different kernels produce valid weights.
#[test]
fn test_window_weights_various_kernels() {
    let kernels = [
        WeightFunction::Tricube,
        WeightFunction::Epanechnikov,
        WeightFunction::Biweight,
        WeightFunction::Gaussian,
    ];

    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let params = WeightParams::new(2.0, 2.0, false);

    for kernel in kernels {
        let mut weights = vec![0.0f64; 5];
        let (sum, _rightmost) = kernel.compute_window_weights(
            &x,
            0,
            4,
            params.x_current,
            params.window_radius,
            params.h1,
            params.h9,
            &mut weights,
        );

        assert!(sum > 0.0, "{} should produce positive sum", kernel.name());
        assert!(sum.is_finite(), "{} sum should be finite", kernel.name());
    }
}

// ============================================================================
// MAD and Additional Math Edge Cases
// ============================================================================

/// Test explicit MAD computation for various distributions.
#[test]
fn test_compute_mad_explicit() {
    // 1. Identical values => MAD=0
    let mut residuals = vec![10.0, 10.0, 10.0, 10.0];
    assert_eq!(ScalingMethod::MAD.compute(&mut residuals), 0.0);

    // 2. Uniformly spaced => [1, 2, 3, 4, 5]
    // Median is 3. Deviations: [2, 1, 0, 1, 2]. Sorted: [0, 1, 1, 2, 2]. MAD median = 1.
    let mut residuals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert_relative_eq!(
        ScalingMethod::MAD.compute(&mut residuals),
        1.0,
        epsilon = 1e-12
    );

    // 3. Even length: [1, 2, 3, 4]
    // Median: (2+3)/2 = 2.5.
    // Deviations: [1.5, 0.5, 0.5, 1.5]. Sorted: [0.5, 0.5, 1.5, 1.5]. MAD median = (0.5+1.5)/2 = 1.0.
    let mut residuals = vec![1.0, 2.0, 3.0, 4.0];
    assert_relative_eq!(
        ScalingMethod::MAD.compute(&mut residuals),
        1.0,
        epsilon = 1e-12
    );

    // 4. Extreme outlier: [1, 2, 3, 1000]
    // Median: (2+3)/2 = 2.5.
    // Deviations: [1.5, 0.5, 0.5, 997.5]. Sorted: [0.5, 0.5, 1.5, 997.5]. MAD median = (0.5+1.5)/2 = 1.0.
    let mut residuals = vec![1.0, 2.0, 3.0, 1000.0];
    assert_relative_eq!(
        ScalingMethod::MAD.compute(&mut residuals),
        1.0,
        epsilon = 1e-12
    );
}

/// Test kernel weights exactly at the 1.0 boundary.
#[test]
fn test_kernel_exact_boundaries() {
    let bounded = [
        WeightFunction::Tricube,
        WeightFunction::Biweight,
        WeightFunction::Epanechnikov,
        WeightFunction::Triangle,
        WeightFunction::Cosine,
    ];

    for kernel in bounded {
        // u=1.0 should be exactly 0.0
        assert_eq!(
            kernel.compute_weight(1.0),
            0.0,
            "{} should be 0 at u=1.0",
            kernel.name()
        );
    }
}

/// Test Gaussian kernel cutoff behavior.
#[test]
fn test_gaussian_cutoff_behavior() {
    let gaussian = WeightFunction::Gaussian;

    // GAUSSIAN_CUTOFF is 6.0
    let at_cutoff = gaussian.compute_weight(6.0);
    assert!(at_cutoff > 0.0);

    let just_past = gaussian.compute_weight(6.0001);
    // Should return MIN_POSITIVE or similar small value
    assert!(just_past > 0.0);
    assert!(just_past <= at_cutoff);
}
