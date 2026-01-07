#![cfg(feature = "dev")]

use lowess::internals::api::{Adapter, BoundaryPolicy, LowessBuilder as Lowess, WeightFunction};
use lowess::internals::math::boundary::apply_boundary_policy;

#[test]
fn test_boundary_policy_comparison() {
    // Dataset where edge points are sensitive to padding
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0];

    // Use fraction(0.8) to get q=4, so that h is large enough to include padded points
    let base_builder = Lowess::new()
        .fraction(0.8)
        .iterations(0)
        .weight_function(WeightFunction::Uniform);

    // Fit with standard Extend (default)
    let res_extend = base_builder
        .clone()
        .boundary_policy(BoundaryPolicy::Extend)
        .adapter(Adapter::Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Fit with Reflect
    let res_reflect = base_builder
        .clone()
        .boundary_policy(BoundaryPolicy::Reflect)
        .adapter(Adapter::Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Fit with Zero
    let res_zero = base_builder
        .clone()
        .boundary_policy(BoundaryPolicy::Zero)
        .adapter(Adapter::Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    println!(
        "Edge values (q=4, Uniform): Extend={}, Reflect={}, Zero={}",
        res_extend.y[0], res_reflect.y[0], res_zero.y[0]
    );

    assert_ne!(
        res_extend.y[0], res_reflect.y[0],
        "Extend vs Reflect at edge"
    );
    assert_ne!(res_extend.y[0], res_zero.y[0], "Extend vs Zero at edge");
    assert_ne!(res_reflect.y[0], res_zero.y[0], "Reflect vs Zero at edge");
}

#[test]
fn test_boundary_policy_zero_effect() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![100.0, 100.0, 100.0, 100.0, 100.0, 100.0];

    // Zero padding should pull the edge down
    let res_zero = Lowess::new()
        .fraction(0.8)
        .iterations(0)
        .weight_function(WeightFunction::Uniform)
        .boundary_policy(BoundaryPolicy::Zero)
        .adapter(Adapter::Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // With q=4, res_zero.y[0] should be around 50 (if window includes two 0s and two 100s)
    // res_zero.y[2] should be 100 (if window only includes 100s)
    println!(
        "Zero padding effect: edge={}, middle={}",
        res_zero.y[0], res_zero.y[2]
    );
    assert!(
        res_zero.y[2] > res_zero.y[0],
        "Center should be higher than Zero-padded edge"
    );
}

// ============================================================================
// Internal Boundary Edge Cases
// ============================================================================

/// Test boundary policy with a minimal dataset (2 points).
#[test]
fn test_boundary_minimal_dataset() {
    let x = vec![0.0, 1.0];
    let y = vec![10.0, 20.0];

    // window_size=2 => pad_len=1. min(1, n-1) = 1.
    let (px, py) = apply_boundary_policy(&x, &y, 2, BoundaryPolicy::Extend);

    // Should result in [x0-dx, x0, x1, x1+dx] => [-1.0, 0.0, 1.0, 2.0]
    assert_eq!(px.len(), 4);
    assert_eq!(py.len(), 4);
    assert_eq!(px[0], -1.0);
    assert_eq!(px[3], 2.0);
    assert_eq!(py[0], 10.0);
    assert_eq!(py[3], 20.0);
}

/// Test boundary policy with a window size larger than the data length.
#[test]
fn test_boundary_large_window() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![10.0, 20.0, 30.0];

    // window_size=10 => pad_len=5. min(5, 3-1) = 2.
    let (px, _) = apply_boundary_policy(&x, &y, 10, BoundaryPolicy::Extend);

    // pad_len is capped at n-1 = 2.
    // total_len = 3 + 2*2 = 7.
    assert_eq!(px.len(), 7);
}

/// Test boundary policy with zero dx (identical x values).
#[test]
fn test_boundary_zero_dx() {
    let x = vec![1.0, 1.0, 2.0, 2.0];
    let y = vec![10.0, 20.0, 30.0, 40.0];

    // x[1] - x[0] = 0.
    let (px, _) = apply_boundary_policy(&x, &y, 4, BoundaryPolicy::Extend);

    // Padding should all have x=1.0 on the left
    assert_eq!(px[0], 1.0);
    assert_eq!(px[1], 1.0);
}

/// Test boundary policy with a small window (size 2).
#[test]
fn test_boundary_small_window() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    // window_size=2 => pad_len=1.
    let (px, _) = apply_boundary_policy(&x, &y, 2, BoundaryPolicy::Reflect);

    assert_eq!(px.len(), 7); // 5 + 2*1
    assert_eq!(px[0], -1.0); // x0 - (x1 - x0) = 0 - (1 - 0) = -1
    assert_eq!(px[6], 5.0); // x4 + (x4 - x3) = 4 + (4 - 3) = 5
}
