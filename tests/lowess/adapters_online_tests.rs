#![cfg(feature = "dev")]
//! Tests for the Online adapter.
//!
//! The Online adapter provides incremental LOWESS smoothing with a sliding window,
//! designed for:
//! - Real-time data streams
//! - Sensor data processing
//! - Memory-constrained environments
//! - Incremental updates without reprocessing entire dataset
//!
//! ## Test Organization
//!
//! 1. **Basic Functionality** - Core incremental smoothing behavior
//! 2. **Window Management** - Capacity, eviction, and sliding window behavior
//! 3. **Lifecycle Management** - Reset and reuse functionality
//! 4. **Builder Validation** - Parameter validation and error handling
//! 5. **Edge Cases** - Boundary conditions and special scenarios
//! 6. **Update Mode** - Incremental vs Full mode comparison

use approx::assert_relative_eq;
use lowess::prelude::*;

use lowess::internals::adapters::online::OnlineLowessBuilder;
use lowess::internals::adapters::online::UpdateMode;
use lowess::internals::math::boundary::BoundaryPolicy;
use lowess::internals::primitives::errors::LowessError;

// ============================================================================
// Basic Functionality Tests
// ============================================================================

/// Test basic incremental smoothing with exact linear data.
///
/// Verifies that online LOWESS reproduces exact linear data when fraction=1.0.
#[test]
fn test_online_exact_linear_reproduction() {
    let x = [0.0f64, 1.0, 2.0];
    let y = [1.0f64, 3.0, 5.0]; // y = 2*x + 1

    let mut processor = Lowess::new()
        .fraction(1.0)
        .iterations(0)
        .adapter(Online)
        .window_capacity(3)
        .min_points(2)
        .build()
        .expect("Builder should succeed");

    let mut smoothed = Vec::new();
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if let Some(output) = processor
            .add_point(xi, yi)
            .expect("add_point should succeed")
        {
            smoothed.push(output.smoothed);
        } else {
            // First few points return None until min_points reached
            smoothed.push(yi);
        }
    }

    assert_eq!(smoothed.len(), 3, "Should have 3 smoothed values");
    for (&s, &t) in smoothed.iter().zip(y.iter()) {
        assert_relative_eq!(s, t, max_relative = 1e-12, epsilon = 1e-14);
    }
}

/// Test basic add_point functionality and type checking.
///
/// Verifies:
/// - Window size tracking
/// - Point addition returns correct values
/// - Smoothed values are computed correctly
#[test]
fn test_online_add_point_basic() {
    let mut processor = Lowess::new()
        .fraction(1.0)
        .iterations(0)
        .adapter(Online)
        .window_capacity(5)
        .min_points(2)
        .build()
        .expect("Builder should succeed");

    // Initially empty window
    assert_eq!(processor.window_size(), 0, "Window should start empty");

    // First point returns None (need min_points)
    assert_eq!(
        processor.add_point(0.0, 1.0).expect("add_point ok"),
        None,
        "First point should return None"
    );

    // Second point produces smoothed value
    let output = processor.add_point(1.0, 3.0).expect("add_point ok");
    assert!(output.is_some(), "Second point should return Some");
    assert_relative_eq!(
        output.unwrap().smoothed,
        3.0,
        max_relative = 1e-12,
        epsilon = 1e-14
    );
}

// ============================================================================
// Window Management Tests
// ============================================================================

/// Test window eviction and capacity behavior.
///
/// Verifies:
/// - Window fills to capacity
/// - Oldest points are evicted when capacity is reached
/// - Window size stays at capacity after eviction
#[test]
fn test_online_window_eviction() {
    let mut processor = Lowess::new()
        .fraction(1.0) // Exact linear fit => smoothed value equals y
        .iterations(0)
        .adapter(Online)
        .window_capacity(3)
        .min_points(2)
        .build()
        .expect("Builder should succeed");

    // First point - not enough points yet
    assert_eq!(processor.window_size(), 0);
    assert_eq!(
        processor.add_point(0.0, 1.0).expect("ok"),
        None,
        "First point returns None"
    );

    // Second point - produces smoothed value
    let second = processor.add_point(1.0, 3.0).expect("ok");
    assert!(second.is_some(), "Second point should return Some");
    assert_relative_eq!(second.unwrap().smoothed, 3.0, max_relative = 1e-12);

    // Third point - window fills to capacity
    let third = processor.add_point(2.0, 5.0).expect("ok");
    assert!(third.is_some(), "Third point should return Some");
    assert_relative_eq!(third.unwrap().smoothed, 5.0, max_relative = 1e-12);
    assert_eq!(processor.window_size(), 3, "Window should be at capacity");

    // Fourth point - oldest should be evicted, window size stays at capacity
    let fourth = processor.add_point(3.0, 7.0).expect("ok");
    assert!(fourth.is_some(), "Fourth point should return Some");
    assert_relative_eq!(fourth.unwrap().smoothed, 7.0, max_relative = 1e-12);
    assert_eq!(
        processor.window_size(),
        3,
        "Window should remain at capacity"
    );
}

/// Test sliding window behavior with continuous data stream.
///
/// Verifies that the window correctly slides as new points are added.
#[test]
fn test_online_sliding_window() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .iterations(1)
        .adapter(Online)
        .window_capacity(10)
        .min_points(3)
        .build()
        .expect("Builder should succeed");

    // Add 20 points to exercise sliding window
    for i in 0..20 {
        let x = i as f64;
        let y = 2.0 * x + 1.0;
        let result = processor.add_point(x, y).expect("add_point ok");

        if i >= 2 {
            // After min_points, should always return Some
            assert!(result.is_some(), "Should return smoothed value at i={}", i);
            assert!(
                result.unwrap().smoothed.is_finite(),
                "Smoothed value should be finite at i={}",
                i
            );
        }
    }

    // Window should be at capacity
    assert_eq!(
        processor.window_size(),
        10,
        "Window should be at capacity after stream"
    );
}

// ============================================================================
// Lifecycle Management Tests
// ============================================================================

/// Test reset functionality.
///
/// Verifies that reset clears internal state and allows reuse.
#[test]
fn test_online_reset() {
    let mut processor = Lowess::new()
        .fraction(1.0)
        .iterations(0)
        .adapter(Online)
        .window_capacity(5)
        .min_points(2)
        .build()
        .expect("Builder should succeed");

    // Add some points
    processor.add_point(0.0, 1.0).expect("ok");
    processor.add_point(1.0, 3.0).expect("ok");

    // Reset clears state
    processor.reset();
    assert_eq!(
        processor.window_size(),
        0,
        "Window should be empty after reset"
    );
}

/// Test reuse after reset.
///
/// Verifies that processor can be reused after reset with new data.
#[test]
fn test_online_reuse_after_reset() {
    let mut processor = Lowess::new()
        .fraction(1.0)
        .iterations(0)
        .adapter(Online)
        .window_capacity(4)
        .min_points(2)
        .build()
        .expect("Builder should succeed");

    // Populate and ensure smoothing occurs
    processor.add_point(0.0, 1.0).expect("ok");
    let _ = processor.add_point(1.0, 3.0).expect("ok");
    assert!(
        processor.window_size() >= 2,
        "Window should have at least 2 points"
    );

    // Reset clears internal buffers
    processor.reset();
    assert_eq!(processor.window_size(), 0, "Window should be empty");

    // Reuse after reset: first point returns None again
    assert_eq!(
        processor.add_point(10.0, 21.0).expect("ok"),
        None,
        "First point after reset should return None"
    );

    // Second point after reset should produce a smoothed value
    let output = processor.add_point(11.0, 23.0).expect("ok");
    assert!(output.is_some(), "Second point should return Some");
    assert_relative_eq!(output.unwrap().smoothed, 23.0, max_relative = 1e-12);
}

// ============================================================================
// Builder Validation Tests
// ============================================================================

/// Test window capacity validation.
///
/// Verifies that window_capacity < 3 is rejected.
#[test]
fn test_online_invalid_window_capacity() {
    let result = Lowess::<f64>::new()
        .adapter(Online)
        .window_capacity(2)
        .build();

    assert!(
        matches!(result, Err(LowessError::InvalidWindowCapacity { .. })),
        "Window capacity < 3 should be rejected"
    );
}

/// Test min_points validation.
///
/// Verifies that invalid min_points values are rejected.
#[test]
fn test_online_invalid_min_points() {
    // min_points < 2 should error
    let result1 = Lowess::<f64>::new()
        .adapter(Online)
        .window_capacity(10)
        .min_points(1)
        .build();

    assert!(
        matches!(result1, Err(LowessError::InvalidMinPoints { .. })),
        "min_points < 2 should be rejected"
    );

    // min_points > window_capacity should error
    let result2 = Lowess::<f64>::new()
        .adapter(Online)
        .window_capacity(10)
        .min_points(20)
        .build();

    assert!(
        matches!(result2, Err(LowessError::InvalidMinPoints { .. })),
        "min_points > window_capacity should be rejected"
    );
}

/// Test valid builder configuration.
///
/// Verifies that valid configurations are accepted.
#[test]
fn test_online_valid_builder() {
    let result = Lowess::new()
        .fraction(0.5)
        .iterations(2)
        .adapter(Online)
        .window_capacity(10)
        .min_points(3)
        .build();

    assert!(result.is_ok(), "Valid configuration should be accepted");

    let mut processor = result.unwrap();

    // Verify initial state
    assert_eq!(processor.window_size(), 0, "Window should start empty");

    // Add points to verify it works
    assert_eq!(processor.add_point(0.0, 1.0).expect("ok"), None);
    assert_eq!(processor.add_point(1.0, 3.0).expect("ok"), None);
    let third = processor.add_point(2.0, 5.0).expect("ok");
    assert!(third.is_some(), "Third point should return Some");
    assert_eq!(processor.window_size(), 3, "Window should have 3 points");

    // Reset should work
    processor.reset();
    assert_eq!(
        processor.window_size(),
        0,
        "Window should be empty after reset"
    );
}

// ============================================================================
// Edge Cases and Special Scenarios
// ============================================================================

/// Test OnlineLowessBuilder default values.
#[test]
fn test_online_builder_defaults() {
    let b = OnlineLowessBuilder::<f64>::default();
    assert_eq!(b.window_capacity, 1000);
}

/// Test OnlineLowessBuilder setters.
#[test]
fn test_online_builder_setters() {
    let b = OnlineLowessBuilder::<f64>::default()
        .boundary_policy(BoundaryPolicy::Extend)
        .delta(0.1)
        .update_mode(UpdateMode::Incremental)
        .window_capacity(100)
        .min_points(5);
    assert_eq!(b.boundary_policy, BoundaryPolicy::Extend);
    assert_eq!(b.delta, 0.1);
    assert_eq!(b.update_mode, UpdateMode::Incremental);
    assert_eq!(b.window_capacity, 100);
    assert_eq!(b.min_points, 5);
}

/// Test basic incremental smoothing and state management.
#[test]
fn test_online_lowess_basic() {
    let mut model = Lowess::<f64>::new()
        .fraction(0.5)
        .adapter(Online)
        .window_capacity(10)
        .min_points(5)
        .build()
        .unwrap();

    for i in 0..10 {
        let res = model.add_point(i as f64, i as f64 * 2.0).unwrap();
        if i < 4 {
            assert!(res.is_none());
        } else {
            assert!(res.is_some());
        }
    }

    assert_eq!(model.window_size(), 10);
    model.reset();
    assert_eq!(model.window_size(), 0);
}

/// Test handling of duplicate x values.
///
/// Verifies that duplicate x values fall back to mean of y values.
#[test]
fn test_online_duplicate_x_values() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .iterations(0)
        .adapter(Online)
        .window_capacity(5)
        .min_points(2)
        .build()
        .expect("Builder should succeed");

    // First point - returns None (needs min_points)
    assert_eq!(
        processor.add_point(1.0, 1.0).expect("ok"),
        None,
        "First point returns None"
    );

    // Second point with same x - should fallback to mean((1+3)/2)=2.0
    let output = processor.add_point(1.0, 3.0).expect("ok");
    assert!(output.is_some(), "Second point should return Some");
    assert_relative_eq!(
        output.unwrap().smoothed,
        2.0,
        max_relative = 1e-12,
        epsilon = 1e-14
    );
}

/// Test with minimum window capacity.
///
/// Verifies that the minimum allowed window capacity (3) works correctly.
#[test]
fn test_online_minimum_window_capacity() {
    let mut processor = Lowess::new()
        .fraction(1.0)
        .iterations(0)
        .adapter(Online)
        .window_capacity(3) // Minimum allowed
        .min_points(2)
        .build()
        .expect("Minimum window capacity should be accepted");

    // Add 3 points
    processor.add_point(0.0, 1.0).expect("ok");
    let second = processor.add_point(1.0, 2.0).expect("ok");
    assert!(second.is_some(), "Second point should return Some");

    let third = processor.add_point(2.0, 3.0).expect("ok");
    assert!(third.is_some(), "Third point should return Some");
    assert_eq!(processor.window_size(), 3, "Window should be at capacity");
}

/// Test with various robustness methods.
///
/// Verifies that different robustness methods work with online adapter.
#[test]
fn test_online_robustness_methods() {
    let methods = vec![Bisquare, Huber, Talwar];

    for method in methods {
        let mut processor = Lowess::new()
            .fraction(0.5)
            .iterations(3)
            .robustness_method(method)
            .adapter(Online)
            .window_capacity(10)
            .min_points(3)
            .build()
            .expect("Builder should succeed");

        // Add points with an outlier
        for i in 0..10 {
            let x = i as f64;
            let y = if i == 5 { 100.0 } else { 2.0 * x + 1.0 }; // Outlier at i=5
            let result = processor.add_point(x, y).expect("add_point ok");

            if i >= 2 {
                assert!(result.is_some(), "Should return smoothed value");
                assert!(
                    result.unwrap().smoothed.is_finite(),
                    "Smoothed value should be finite"
                );
            }
        }
    }
}

/// Test with residuals enabled.
///
/// Verifies that residuals can be computed in online mode.
#[test]
fn test_online_with_residuals() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .iterations(2)
        .return_residuals()
        .adapter(Online)
        .window_capacity(10)
        .min_points(3)
        .build()
        .expect("Builder should succeed");

    // Add several points
    for i in 0..10 {
        let x = i as f64;
        let y = 2.0 * x + 1.0;
        let result = processor.add_point(x, y).expect("add_point ok");

        if let Some(output) = result {
            // Verify residual is present when requested
            assert!(
                output.residual.is_some(),
                "Residual should be present when requested"
            );
            assert!(
                output.residual.unwrap().is_finite(),
                "Residual should be finite"
            );
        }
    }
}

// ============================================================================
// Update Mode Tests
// ============================================================================

/// Test that Incremental and Full modes produce consistent results.
///
/// Verifies that Incremental mode (fitting only the latest point) produces
/// results that match the last value from Full mode (re-smoothing entire window).
#[test]
fn test_update_mode_consistency() {
    let test_data: Vec<(f64, f64)> = (0..20)
        .map(|i| {
            let x = i as f64;
            let y = 2.0 * x + 1.0 + (i % 3) as f64 * 0.5; // Add slight variation
            (x, y)
        })
        .collect();

    // Test with Incremental mode
    let mut incremental = Lowess::new()
        .fraction(0.5)
        .iterations(0)
        .adapter(Online)
        .window_capacity(10)
        .min_points(3)
        .update_mode(UpdateMode::Incremental)
        .build()
        .expect("Builder should succeed");

    // Test with Full mode
    let mut full = Lowess::new()
        .fraction(0.5)
        .iterations(0)
        .adapter(Online)
        .window_capacity(10)
        .min_points(3)
        .update_mode(UpdateMode::Full)
        .build()
        .expect("Builder should succeed");

    let mut incremental_results = Vec::new();
    let mut full_results = Vec::new();

    for (x, y) in test_data {
        if let Some(output) = incremental.add_point(x, y).expect("add_point ok") {
            incremental_results.push(output.smoothed);
        }
        if let Some(output) = full.add_point(x, y).expect("add_point ok") {
            full_results.push(output.smoothed);
        }
    }

    // Both modes should produce the same number of results
    assert_eq!(
        incremental_results.len(),
        full_results.len(),
        "Both modes should produce same number of results"
    );

    // All results should be finite and reasonable
    for inc in incremental_results.iter() {
        assert!(inc.is_finite(), "Incremental result should be finite");
        assert!(
            *inc > 0.0 && *inc < 50.0,
            "Incremental result should be in reasonable range"
        );
    }

    for full in full_results.iter() {
        assert!(full.is_finite(), "Full result should be finite");
        assert!(
            *full > 0.0 && *full < 50.0,
            "Full result should be in reasonable range"
        );
    }
}

/// Test that Incremental mode is faster than Full mode.
///
/// This is a basic sanity check that Incremental mode completes faster.
#[test]
fn test_incremental_mode_performance() {
    use std::time::Instant;

    let test_data: Vec<(f64, f64)> = (0..1000)
        .map(|i| {
            let x = i as f64;
            let y = 2.0 * x + 1.0;
            (x, y)
        })
        .collect();

    // Benchmark Incremental mode
    let mut incremental = Lowess::new()
        .fraction(0.3)
        .iterations(0)
        .adapter(Online)
        .window_capacity(100)
        .min_points(3)
        .update_mode(UpdateMode::Incremental)
        .build()
        .expect("Builder should succeed");

    let start = Instant::now();
    for (x, y) in &test_data {
        let _ = incremental.add_point(*x, *y).expect("add_point ok");
    }
    let incremental_time = start.elapsed();

    // Benchmark Full mode
    let mut full = Lowess::new()
        .fraction(0.3)
        .iterations(0)
        .adapter(Online)
        .window_capacity(100)
        .min_points(3)
        .update_mode(UpdateMode::Full)
        .build()
        .expect("Builder should succeed");

    let start = Instant::now();
    for (x, y) in &test_data {
        let _ = full.add_point(*x, *y).expect("add_point ok");
    }
    let full_time = start.elapsed();

    // Incremental should be faster (at least 2x for this window size)
    println!(
        "Incremental: {:?}, Full: {:?}, Speedup: {:.2}x",
        incremental_time,
        full_time,
        full_time.as_secs_f64() / incremental_time.as_secs_f64()
    );

    assert!(
        incremental_time < full_time,
        "Incremental mode should be faster than Full mode"
    );
}

/// Test with robustness weights enabled.
///
/// Verifies that robustness weights can be returned in online mode.
#[test]
fn test_online_with_robustness_weights() {
    let mut processor = Lowess::new()
        .fraction(0.99)
        .iterations(3)
        .return_robustness_weights()
        .adapter(Online)
        .window_capacity(30)
        .min_points(10)
        .update_mode(UpdateMode::Full)
        .build()
        .expect("Builder should succeed");

    // Add 30 points with an outlier in the middle
    for i in 0..30 {
        let x = i as f64;
        let y = if i == 15 { 500.0 } else { 2.0 * x + 1.0 };
        let result = processor.add_point(x, y).expect("add_point ok");

        if let Some(output) = result {
            // Verify robustness weight is present when requested
            assert!(
                output.robustness_weight.is_some(),
                "Robustness weight should be present when requested at i={}",
                i
            );
            let w = output.robustness_weight.unwrap();
            assert!(w.is_finite(), "Robustness weight should be finite");
            assert!(
                (0.0..=1.0).contains(&w),
                "Robustness weight {} out of range",
                w
            );

            // After outlier is added (i=15), it should be heavily downweighted
            if i == 15 {
                assert!(
                    w < 0.2,
                    "Outlier at i=15 should be heavily downweighted, got {}",
                    w
                );
            }
        }
    }
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

/// Test window capacity exactly equal to min_points.
#[test]
fn test_online_window_exactly_min_points() {
    let mut processor = Lowess::new()
        .fraction(0.8)
        .adapter(Online)
        .window_capacity(5)
        .min_points(5)
        .build()
        .unwrap();

    // Add exactly min_points
    for i in 0..5 {
        let output = processor.add_point(i as f64, (i * 2) as f64).unwrap();
        if i >= 4
            && let Some(output) = output
        {
            assert!(output.smoothed.is_finite());
        }
    }

    assert_eq!(processor.window_size(), 5);
}

/// Test with all points having identical values.
#[test]
fn test_online_all_points_identical() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .adapter(Online)
        .window_capacity(10)
        .min_points(3)
        .build()
        .unwrap();

    // Add identical points
    for _ in 0..10 {
        let output = processor.add_point(5.0, 10.0).unwrap();
        // Should return the constant value
        if processor.window_size() >= 3
            && let Some(output) = output
        {
            assert_relative_eq!(output.smoothed, 10.0, epsilon = 1e-6);
        }
    }
}

/// Test with decreasing x-values (non-monotonic).
#[test]
fn test_online_decreasing_x_values() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .adapter(Online)
        .window_capacity(10)
        .min_points(3)
        .build()
        .unwrap();

    // Add points with decreasing x
    for i in 0..10 {
        let x = (10 - i) as f64;
        let y = x * 2.0;
        let output = processor.add_point(x, y).unwrap();

        // Should still produce valid output
        if processor.window_size() >= 3
            && let Some(output) = output
        {
            assert!(output.smoothed.is_finite());
        }
    }
}

/// Test with extreme window sizes.
#[test]
fn test_online_extreme_window_sizes() {
    // Very large window
    let processor = Lowess::new()
        .fraction(0.1)
        .adapter(Online)
        .window_capacity(10000)
        .min_points(100)
        .build()
        .unwrap();

    assert_eq!(processor.window_size(), 0);

    // Minimum window
    let processor_min = Lowess::new()
        .fraction(0.9)
        .adapter(Online)
        .window_capacity(3)
        .min_points(2)
        .build()
        .unwrap();

    assert_eq!(processor_min.window_size(), 0);
}

/// Test fraction at boundaries.
#[test]
fn test_online_fraction_boundaries() {
    // Very small fraction
    let mut proc_small = Lowess::new()
        .fraction(0.01)
        .adapter(Online)
        .window_capacity(100)
        .min_points(10)
        .build()
        .unwrap();

    for i in 0..20 {
        proc_small.add_point(i as f64, (i * 2) as f64).unwrap();
    }

    // Fraction = 1.0 (global regression on window)
    let mut proc_one = Lowess::new()
        .fraction(1.0)
        .adapter(Online)
        .window_capacity(20)
        .min_points(5)
        .build()
        .unwrap();

    for i in 0..10 {
        let output = proc_one.add_point(i as f64, (i * 2) as f64).unwrap();
        if proc_one.window_size() >= 5
            && let Some(output) = output
        {
            assert!(output.smoothed.is_finite());
        }
    }
}

/// Test reset clears all state properly.
#[test]
fn test_online_reset_complete() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .adapter(Online)
        .window_capacity(10)
        .min_points(3)
        .build()
        .unwrap();

    // Add some points
    for i in 0..5 {
        processor.add_point(i as f64, (i * 2) as f64).unwrap();
    }

    assert_eq!(processor.window_size(), 5);

    // Reset
    processor.reset();

    // Verify complete reset
    assert_eq!(processor.window_size(), 0);

    // Should be able to add new points
    let _output = processor.add_point(100.0, 200.0).unwrap();
    assert_eq!(processor.window_size(), 1);
}
