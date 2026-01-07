#![cfg(feature = "dev")]
//! Tests for the Streaming adapter.
//!
//! The Streaming adapter provides chunked LOWESS processing for large datasets,
//! designed for:
//! - Large datasets (>100K points) that don't fit in memory
//! - Batch processing pipelines
//! - File-based data processing
//! - ETL (Extract, Transform, Load) workflows
//!
//! ## Test Organization
//!
//! 1. **Utility Functions** - Partition and merge helpers
//! 2. **Builder Validation** - Parameter validation and error handling
//! 3. **Chunk Processing** - process_chunk and finalize behavior
//! 4. **Overlap Management** - Chunk overlap and merging
//! 5. **Lifecycle Management** - Reset and state management
//! 6. **Edge Cases** - Boundary conditions and special scenarios

use approx::assert_relative_eq;
use lowess::prelude::*;

use lowess::internals::adapters::streaming::MergeStrategy;
use lowess::internals::adapters::streaming::StreamingLowessBuilder;
use lowess::internals::math::boundary::BoundaryPolicy;
use lowess::internals::primitives::errors::LowessError;

// ============================================================================
// Builder Validation Tests
// ============================================================================

/// Test chunk size validation.
///
/// Verifies that chunk_size < 10 is rejected.
#[test]
fn test_streaming_invalid_chunk_size() {
    let result = Lowess::<f64>::new()
        .adapter(Streaming)
        .chunk_size(5)
        .build();

    assert!(
        matches!(result, Err(LowessError::InvalidChunkSize { .. })),
        "chunk_size < 10 should be rejected"
    );
}

/// Test overlap validation.
///
/// Verifies that overlap >= chunk_size is rejected.
#[test]
fn test_streaming_invalid_overlap() {
    let result = Lowess::<f64>::new()
        .adapter(Streaming)
        .chunk_size(10)
        .overlap(10)
        .build();

    assert!(
        matches!(result, Err(LowessError::InvalidOverlap { .. })),
        "overlap >= chunk_size should be rejected"
    );
}

/// Test fraction validation.
///
/// Verifies that invalid fraction values are rejected.
#[test]
fn test_streaming_invalid_fraction() {
    let result = Lowess::<f64>::new()
        .fraction(0.0)
        .adapter(Streaming)
        .chunk_size(10)
        .overlap(1)
        .build();

    assert!(
        matches!(result, Err(LowessError::InvalidFraction(_))),
        "fraction <= 0 should be rejected"
    );
}

/// Test mismatched input lengths.
///
/// Verifies that process_chunk rejects mismatched x and y arrays.
#[test]
fn test_streaming_mismatched_inputs() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .iterations(0)
        .adapter(Streaming)
        .chunk_size(10)
        .overlap(1)
        .build()
        .expect("Builder should succeed");

    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![1.0f64, 2.0]; // Mismatched length

    let result = processor.process_chunk(&x, &y);
    assert!(
        matches!(result, Err(LowessError::MismatchedInputs { .. })),
        "Mismatched inputs should be rejected"
    );
}

// ============================================================================
// Chunk Processing Tests
// ============================================================================

/// Test basic process_chunk and finalize roundtrip.
///
/// Verifies:
/// - process_chunk returns non-overlapping portion
/// - finalize returns remaining overlap buffer
#[test]
fn test_streaming_basic_roundtrip() {
    let mut processor = Lowess::new()
        .fraction(1.0) // Global linear fit => exact predictions for linear data
        .iterations(0)
        .adapter(Streaming)
        .chunk_size(10)
        .overlap(2)
        .build()
        .expect("Builder should succeed");

    let x = vec![0.0f64, 1.0, 2.0, 3.0, 4.0];
    let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 1.0).collect();

    // process_chunk returns non-overlapping portion (n - overlap)
    let result = processor.process_chunk(&x, &y).expect("process_chunk ok");
    assert_eq!(
        result.y.len(),
        3,
        "Should return n - overlap = 5 - 2 = 3 values"
    );

    for (i, &v) in result.y.iter().enumerate() {
        assert_relative_eq!(v, y[i], max_relative = 1e-12, epsilon = 1e-14);
    }

    // finalize returns remaining overlap buffer (last 2 points)
    let remaining = processor.finalize().expect("finalize ok");
    assert_eq!(remaining.y.len(), 2, "Should return 2 remaining points");

    for (i, &v) in remaining.y.iter().enumerate() {
        assert_relative_eq!(v, y[3 + i], max_relative = 1e-12, epsilon = 1e-14);
    }
}

/// Test processing multiple chunks with overlap.
///
/// Verifies that overlap between chunks is correctly merged.
#[test]
fn test_streaming_multi_chunk_overlap() {
    let x_all: Vec<f64> = (0..15).map(|i| i as f64).collect();
    let y_all: Vec<f64> = x_all.iter().map(|xi| 2.0 * xi + 1.0).collect();

    let mut processor = Lowess::new()
        .fraction(1.0)
        .iterations(0)
        .adapter(Streaming)
        .chunk_size(10)
        .overlap(2)
        .build()
        .expect("Builder should succeed");

    // First chunk (indices 0..10)
    let x_a = &x_all[0..10];
    let y_a = &y_all[0..10];
    let out_a = processor.process_chunk(x_a, y_a).expect("process_chunk ok");

    // With chunk_size=10 and overlap=2, non-overlap return length is 10 - 2 = 8
    assert_eq!(out_a.y.len(), 8, "First chunk should return 8 values");
    assert_eq!(
        out_a.y,
        y_all[0..8].to_vec(),
        "First chunk values should match"
    );

    // Second chunk (indices 8..15)
    let x_b = &x_all[8..15];
    let y_b = &y_all[8..15];
    let out_b = processor.process_chunk(x_b, y_b).expect("process_chunk ok");

    assert_eq!(out_b.y.len(), 7, "Second chunk should return 7 values");
    assert!(
        out_b.y.iter().all(|v| v.is_finite()),
        "All values should be finite"
    );

    // Verify first values match expected
    assert_relative_eq!(out_b.y[0], y_all[8], max_relative = 1e-12);
    assert_relative_eq!(out_b.y[1], y_all[9], max_relative = 1e-12);
}

/// Test that finalize on fresh processor returns empty.
///
/// Verifies that finalize without any processing returns empty result.
#[test]
fn test_streaming_finalize_fresh() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .iterations(0)
        .adapter(Streaming)
        .chunk_size(10)
        .overlap(2)
        .build()
        .expect("Builder should succeed");

    // No processing done => finalize should return empty
    let remaining = processor.finalize().expect("finalize ok");
    assert!(
        remaining.y.is_empty(),
        "Fresh processor should return empty on finalize"
    );
}

#[test]
fn test_single_chunk_returns_all_points() {
    let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    let mut processor = Lowess::new()
        .fraction(0.3)
        .adapter(Streaming)
        .chunk_size(50)
        .overlap(10)
        .build()
        .unwrap();

    let result = processor.process_chunk(&x, &y).unwrap();
    let final_result = processor.finalize().unwrap();

    let total = result.y.len() + final_result.y.len();

    println!("Input: {} points", x.len());
    println!("process_chunk returned: {} points", result.y.len());
    println!("finalize returned: {} points", final_result.y.len());
    println!("Total output: {} points", total);

    assert_eq!(
        total,
        x.len(),
        "Expected {} points but got {} (process_chunk: {}, finalize: {})",
        x.len(),
        total,
        result.y.len(),
        final_result.y.len()
    );
}

// ============================================================================
// Lifecycle Management Tests
// ============================================================================

/// Test reset functionality.
///
/// Verifies that reset clears buffers and resets state.
#[test]
fn test_streaming_reset() {
    let x_all: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let y_all: Vec<f64> = x_all.iter().map(|xi| 2.0 * xi + 1.0).collect();

    let mut processor = Lowess::new()
        .fraction(1.0)
        .iterations(0)
        .adapter(Streaming)
        .chunk_size(10)
        .overlap(2)
        .build()
        .expect("Builder should succeed");

    // Process first chunk to populate overlap
    let _ = processor
        .process_chunk(&x_all[..], &y_all[..])
        .expect("process_chunk ok");

    // Reset should clear buffers and set chunks_processed=0
    processor.reset();

    // After reset, processing the same chunk should behave like the first call
    let out_after_reset = processor
        .process_chunk(&x_all[..], &y_all[..])
        .expect("process_chunk ok");

    // With chunk_size=10 and n=8, non-overlap return length is max(0, n - overlap) = 6
    assert_eq!(
        out_after_reset.y.len(),
        6,
        "After reset should return 6 values"
    );
    assert_eq!(
        out_after_reset.y,
        y_all[0..6].to_vec(),
        "Values should match first 6"
    );
}

// ============================================================================
// Edge Cases and Special Scenarios
// ============================================================================

/// Test StreamingLowessBuilder default values.
#[test]
fn test_streaming_builder_defaults() {
    let b = StreamingLowessBuilder::<f64>::default();
    assert_eq!(b.chunk_size, 5000);
}

/// Test StreamingLowessBuilder setters.
#[test]
fn test_streaming_builder_setters() {
    let b = StreamingLowessBuilder::<f64>::default()
        .boundary_policy(BoundaryPolicy::Extend)
        .chunk_size(100)
        .overlap(10);
    assert_eq!(b.boundary_policy, BoundaryPolicy::Extend);
    assert_eq!(b.chunk_size, 100);
    assert_eq!(b.overlap, 10);
}

/// Test different merge strategies for overlapping chunks.
#[test]
fn test_streaming_merge_strategies() {
    let strategies = [
        MergeStrategy::Average,
        MergeStrategy::TakeFirst,
        MergeStrategy::TakeLast,
        MergeStrategy::WeightedAverage,
    ];

    let x: Vec<f64> = (0..40).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..40).map(|i| i as f64 * 2.0).collect();

    for strategy in strategies {
        let mut model = Lowess::<f64>::new()
            .fraction(0.2)
            .merge_strategy(strategy)
            .adapter(Streaming)
            .chunk_size(20)
            .overlap(10)
            .build()
            .unwrap();

        let r1 = model.process_chunk(&x[0..20], &y[0..20]).unwrap();
        let r2 = model.process_chunk(&x[20..40], &y[20..40]).unwrap();
        let r3 = model.finalize().unwrap();

        let total_len = r1.y.len() + r2.y.len() + r3.y.len();
        assert_eq!(total_len, 40);
    }
}

/// Test merging of robustness weights in streaming mode.
#[test]
fn test_streaming_robustness_merge() {
    let x: Vec<f64> = (0..40).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..40).map(|i| i as f64 * 2.0).collect();

    let mut model = Lowess::<f64>::new()
        .fraction(0.2)
        .iterations(1)
        .return_robustness_weights()
        .merge_strategy(MergeStrategy::Average)
        .adapter(Streaming)
        .chunk_size(20)
        .overlap(10)
        .build()
        .unwrap();

    let r1 = model.process_chunk(&x[0..20], &y[0..20]).unwrap();
    let r2 = model.process_chunk(&x[20..40], &y[20..40]).unwrap();
    let r3 = model.finalize().unwrap();

    assert!(r1.robustness_weights.is_some());
    assert!(r2.robustness_weights.is_some());
    assert!(r3.robustness_weights.is_some());

    let total_rw_len = r1.robustness_weights.unwrap().len()
        + r2.robustness_weights.unwrap().len()
        + r3.robustness_weights.unwrap().len();
    assert_eq!(total_rw_len, 40);
}

/// Test with minimum chunk size.
///
/// Verifies that minimum allowed chunk_size (10) works correctly.
#[test]
fn test_streaming_minimum_chunk_size() {
    let result = Lowess::new()
        .fraction(0.5)
        .iterations(0)
        .adapter(Streaming)
        .chunk_size(10) // Minimum allowed
        .overlap(2)
        .build();

    assert!(result.is_ok(), "Minimum chunk size (10) should be accepted");
}

/// Test with various overlap sizes.
///
/// Verifies that different overlap sizes work correctly.
#[test]
fn test_streaming_various_overlaps() {
    let overlaps = vec![0, 1, 5, 9]; // 9 is max for chunk_size=10

    for overlap in overlaps {
        let result = Lowess::new()
            .fraction(0.5)
            .iterations(0)
            .adapter(Streaming)
            .chunk_size(10)
            .overlap(overlap)
            .build();

        assert!(
            result.is_ok(),
            "Overlap {} should be valid for chunk_size=10",
            overlap
        );
    }
}

/// Test with robustness iterations.
///
/// Verifies that robustness iterations work with streaming adapter.
#[test]
fn test_streaming_with_robustness() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .iterations(3)
        .robustness_method(Bisquare)
        .adapter(Streaming)
        .chunk_size(20)
        .overlap(5)
        .build()
        .expect("Builder should succeed");

    // Create data with outliers
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            if i == 10 || i == 20 {
                100.0 // Outliers
            } else {
                2.0 * xi + 1.0
            }
        })
        .collect();

    // Process in chunks
    let result1 = processor.process_chunk(&x[0..20], &y[0..20]).expect("ok");
    assert!(result1.y.iter().all(|v| v.is_finite()));

    let result2 = processor.process_chunk(&x[15..30], &y[15..30]).expect("ok");
    assert!(result2.y.iter().all(|v| v.is_finite()));

    let final_result = processor.finalize().expect("ok");
    assert!(final_result.y.iter().all(|v| v.is_finite()));
}

/// Test with residuals enabled.
///
/// Verifies that residuals can be computed in streaming mode.
#[test]
fn test_streaming_with_residuals() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .iterations(2)
        .return_residuals()
        .adapter(Streaming)
        .chunk_size(15)
        .overlap(3)
        .build()
        .expect("Builder should succeed");

    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    let result = processor.process_chunk(&x, &y).expect("ok");

    // Verify residuals are present
    assert!(
        result.residuals.is_some(),
        "Residuals should be present when requested"
    );
    assert_eq!(
        result.residuals.as_ref().unwrap().len(),
        result.y.len(),
        "Residual count should match output count"
    );
}

/// Test processing very large simulated dataset.
///
/// Verifies that streaming can handle large datasets efficiently.
#[test]
fn test_streaming_large_dataset() {
    let mut processor = Lowess::new()
        .fraction(0.3)
        .iterations(1)
        .adapter(Streaming)
        .chunk_size(100)
        .overlap(10)
        .build()
        .expect("Builder should succeed");

    // Simulate processing 1000 points in chunks
    let total_points = 1000;
    let chunk_size = 100;
    let overlap = 10;
    let step = chunk_size - overlap;

    let mut total_output = 0;

    for chunk_start in (0..total_points).step_by(step) {
        let chunk_end = (chunk_start + chunk_size).min(total_points);
        let x: Vec<f64> = (chunk_start..chunk_end).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let result = processor.process_chunk(&x, &y).expect("process_chunk ok");
        total_output += result.y.len();

        assert!(
            result.y.iter().all(|v| v.is_finite()),
            "All values should be finite"
        );
    }

    // Finalize to get remaining points
    let final_result = processor.finalize().expect("finalize ok");
    total_output += final_result.y.len();

    // Should have processed approximately all points
    assert!(
        total_output >= total_points - overlap,
        "Should process most points (got {}, expected ~{})",
        total_output,
        total_points
    );
}

/// Test with robustness weights enabled.
///
/// Verifies that robustness weights can be returned in streaming mode.
#[test]
fn test_streaming_with_robustness_weights() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .iterations(2)
        .return_robustness_weights()
        .adapter(Streaming)
        .chunk_size(15)
        .overlap(3)
        .build()
        .expect("Builder should succeed");

    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            if (xi - 10.0).abs() < 0.1 {
                100.0
            } else {
                2.0 * xi + 1.0
            }
        })
        .collect();

    // Process first chunk (0..15)
    let result = processor.process_chunk(&x[0..15], &y[0..15]).expect("ok");

    // Verify robustness weights are present
    assert!(
        result.robustness_weights.is_some(),
        "Robustness weights should be present when requested"
    );
    let rw = result.robustness_weights.as_ref().unwrap();
    assert_eq!(
        rw.len(),
        result.y.len(),
        "Robustness weight count should match output count"
    );

    // Point at 10.0 is an outlier and should be downweighted
    // result.y contains first 15 - 3 = 12 points (0..12)
    assert!(
        rw[10] < 0.5,
        "Outlier at x=10 should be downweighted, got {}",
        rw[10]
    );

    // Finalize to get remaining points
    let final_result = processor.finalize().expect("ok");
    assert!(final_result.robustness_weights.is_some());
    assert_eq!(
        final_result.robustness_weights.as_ref().unwrap().len(),
        final_result.y.len()
    );
}

/// Test with diagnostics enabled.
///
/// Verifies that cumulative diagnostics are computed and returned in streaming mode.
#[test]
fn test_streaming_with_diagnostics() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .return_diagnostics()
        .adapter(Streaming)
        .chunk_size(15)
        .overlap(3)
        .build()
        .expect("Builder should succeed");

    // Create a simple constant relationship: y = 10.0
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|_| 10.0).collect();
    // Constant data should have perfect fit even with Extend boundary policy

    // Process first chunk (0..15) -> emits 0..12
    let result = processor.process_chunk(&x[0..15], &y[0..15]).expect("ok");
    assert!(result.diagnostics.is_some());
    let diag = result.diagnostics.as_ref().unwrap();
    assert!(diag.rmse < 1e-10);
    // For constant data, R2 should be 1.0 in our implementation
    assert!((diag.r_squared - 1.0).abs() < 1e-10);

    // Process second chunk (15..30)
    let result2 = processor.process_chunk(&x[15..30], &y[15..30]).expect("ok");
    assert!(result2.diagnostics.is_some());
    let diag2 = result2.diagnostics.as_ref().unwrap();
    // Should still be perfect
    assert!(diag2.rmse < 1e-10);
    assert!((diag2.r_squared - 1.0).abs() < 1e-10);

    // Finalize -> emits 24..30
    let final_result = processor.finalize().expect("ok");
    assert!(final_result.diagnostics.is_some());
    let final_diag = final_result.diagnostics.as_ref().unwrap();
    assert!(final_diag.rmse < 1e-10);
    assert!((final_diag.r_squared - 1.0).abs() < 1e-10);
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

/// Test chunk_size exactly overlap + 1 (minimum valid).
#[test]
fn test_streaming_chunk_exactly_overlap_plus_one() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .adapter(Streaming)
        .chunk_size(11)
        .overlap(10)
        .build()
        .unwrap();

    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * 2.0).collect();

    let result1 = processor.process_chunk(&x[0..11], &y[0..11]).unwrap();
    assert_eq!(result1.x.len(), 1); // Only 1 non-overlapping point

    let result2 = processor.process_chunk(&x[1..12], &y[1..12]).unwrap();
    assert!(!result2.x.is_empty());
}

/// Test with zero overlap.
#[test]
fn test_streaming_zero_overlap() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .adapter(Streaming)
        .chunk_size(10)
        .overlap(0)
        .build()
        .unwrap();

    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * 2.0).collect();

    // Process chunks with no overlap
    let result1 = processor.process_chunk(&x[0..10], &y[0..10]).unwrap();
    assert_eq!(result1.x.len(), 10);

    let result2 = processor.process_chunk(&x[10..20], &y[10..20]).unwrap();
    assert_eq!(result2.x.len(), 10);

    let result3 = processor.finalize().unwrap();
    assert!(result3.x.is_empty()); // No overlap buffer
}

/// Test with very small chunks (1-2 points each).
#[test]
fn test_streaming_single_point_chunks() {
    // This should fail validation since chunk_size must be >= 10
    let result = Lowess::<f64>::new()
        .adapter(Streaming)
        .chunk_size(2)
        .overlap(0)
        .build();

    assert!(result.is_err());
}

/// Test with unsorted chunks.
#[test]
fn test_streaming_unsorted_chunks() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .adapter(Streaming)
        .chunk_size(10)
        .overlap(2)
        .build()
        .unwrap();

    // Create unsorted chunk
    let x = vec![5.0, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0, 4.0, 6.0, 0.0];
    let y = vec![10.0, 6.0, 16.0, 2.0, 18.0, 4.0, 14.0, 8.0, 12.0, 0.0];

    let result = processor.process_chunk(&x, &y).unwrap();

    // Should handle unsorted data (internally sorts)
    assert_eq!(result.x.len(), 8); // 10 - 2 overlap
}

/// Test all merge strategies with identical overlap data.
#[test]
fn test_streaming_all_merge_strategies_identical_data() {
    let strategies = vec![
        MergeStrategy::Average,
        MergeStrategy::WeightedAverage,
        MergeStrategy::TakeFirst,
        MergeStrategy::TakeLast,
    ];

    for strategy in strategies {
        let mut processor = Lowess::new()
            .fraction(0.5)
            .merge_strategy(strategy)
            .adapter(Streaming)
            .chunk_size(10)
            .overlap(3)
            .build()
            .unwrap();

        let x: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 2.0).collect();

        let result1 = processor.process_chunk(&x[0..10], &y[0..10]).unwrap();
        let result2 = processor.process_chunk(&x[7..15], &y[7..15]).unwrap();

        // All strategies should produce valid results
        assert!(!result1.x.is_empty());
        assert!(!result2.x.is_empty());
    }
}

/// Test calling finalize() multiple times.
#[test]
fn test_streaming_finalize_multiple_times() {
    let mut processor = Lowess::new()
        .fraction(0.5)
        .adapter(Streaming)
        .chunk_size(10)
        .overlap(2)
        .build()
        .unwrap();

    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * 2.0).collect();

    processor.process_chunk(&x, &y).unwrap();

    // First finalize
    let result1 = processor.finalize().unwrap();
    assert_eq!(result1.x.len(), 2); // overlap buffer

    // Second finalize should return empty
    let result2 = processor.finalize().unwrap();
    assert!(result2.x.is_empty());

    // Third finalize should also return empty
    let result3 = processor.finalize().unwrap();
    assert!(result3.x.is_empty());
}
