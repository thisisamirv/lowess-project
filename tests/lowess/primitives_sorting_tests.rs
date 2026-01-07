#![cfg(feature = "dev")]
//! Tests for sorting utilities.
//!
//! These tests verify the sorting functionality used in LOWESS for:
//! - Sorting input data by x-coordinates
//! - Maintaining index mappings for order restoration
//! - Handling non-finite values (NaN, Infinity)
//! - Unsorting results back to original order
//!
//! ## Test Organization
//!
//! 1. **Basic Sorting** - sort_by_x with normal data
//! 2. **Non-Finite Handling** - NaN and Infinity placement
//! 3. **Unsort Operations** - Single and multiple array restoration
//! 4. **Edge Cases** - Empty, single, duplicate values
//! 5. **Stability** - Stable sort verification
//! 6. **Roundtrip** - Sort-unsort integrity

use approx::assert_relative_eq;
use num_traits::Float;

use lowess::internals::primitives::sorting::{SortedData, sort_by_x, unsort};

/// Extension trait for testing SortedData properties.
trait SortedDataExt {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}

impl<T> SortedDataExt for SortedData<T> {
    fn len(&self) -> usize {
        self.x.len()
    }

    fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

// ============================================================================
// Basic Sorting Tests
// ============================================================================

/// Test basic sorting with simple data.
///
/// Verifies that data is sorted in ascending order.
#[test]
fn test_sort_basic() {
    let x = vec![3.0, 1.0, 4.0, 2.0];
    let y = vec![30.0, 10.0, 40.0, 20.0];

    let sorted = sort_by_x(&x, &y);

    assert_eq!(sorted.x, vec![1.0, 2.0, 3.0, 4.0], "X should be sorted");
    assert_eq!(sorted.y, vec![10.0, 20.0, 30.0, 40.0], "Y should follow X");
    assert_eq!(
        sorted.indices,
        vec![1, 3, 0, 2],
        "Indices should map to original"
    );
}

/// Test sorting with already sorted data.
///
/// Verifies that sorted data remains unchanged.
#[test]
fn test_sort_already_sorted() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![10.0, 20.0, 30.0, 40.0];

    let sorted = sort_by_x(&x, &y);

    assert_eq!(sorted.x, x, "X should remain unchanged");
    assert_eq!(sorted.y, y, "Y should remain unchanged");
    assert_eq!(
        sorted.indices,
        vec![0, 1, 2, 3],
        "Indices should be identity"
    );
}

/// Test sorting with reverse sorted data.
///
/// Verifies that reverse order is correctly sorted.
#[test]
fn test_sort_reverse_order() {
    let x = vec![4.0, 3.0, 2.0, 1.0];
    let y = vec![40.0, 30.0, 20.0, 10.0];

    let sorted = sort_by_x(&x, &y);

    assert_eq!(sorted.x, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(sorted.y, vec![10.0, 20.0, 30.0, 40.0]);
    assert_eq!(sorted.indices, vec![3, 2, 1, 0]);
}

/// Test sorting with duplicate x values.
///
/// Verifies stable sort preserves original order for duplicates.
#[test]
fn test_sort_duplicates() {
    let x = vec![2.0, 1.0, 2.0, 1.0];
    let y = vec![20.0, 10.0, 22.0, 12.0];

    let sorted = sort_by_x(&x, &y);

    // Should be sorted, with duplicates in original order
    assert_eq!(sorted.x, vec![1.0, 1.0, 2.0, 2.0]);
    // First 1.0 (index 1) should come before second 1.0 (index 3)
    assert_eq!(sorted.y, vec![10.0, 12.0, 20.0, 22.0]);
    assert_eq!(sorted.indices, vec![1, 3, 0, 2]);
}

/// Test SortedData len and is_empty methods.
///
/// Verifies utility methods work correctly.
#[test]
fn test_sorted_data_methods() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![10.0, 20.0, 30.0];

    let sorted = sort_by_x(&x, &y);

    assert_eq!(sorted.len(), 3, "Length should be 3");
    assert!(!sorted.is_empty(), "Should not be empty");
}

// ============================================================================
// Unsort Operations Tests
// ============================================================================

/// Test basic unsort operation.
///
/// Verifies that unsort restores original order.
#[test]
fn test_unsort_basic() {
    let x = vec![3.0, 1.0, 4.0, 2.0];
    let y = vec![30.0, 10.0, 40.0, 20.0];

    let sorted = sort_by_x(&x, &y);

    // Simulate processing: add 1 to each y value
    let processed: Vec<f64> = sorted.y.iter().map(|&v| v + 1.0).collect();

    // Unsort back to original order
    let result = unsort(&processed, &sorted.indices);

    // Should be in original order: [31, 11, 41, 21]
    assert_eq!(result, vec![31.0, 11.0, 41.0, 21.0]);
}

/// Test unsort with identity mapping.
///
/// Verifies that identity indices preserve order.
#[test]
fn test_unsort_identity() {
    let values = vec![10.0, 20.0, 30.0];
    let indices = vec![0, 1, 2]; // Identity mapping

    let result = unsort(&values, &indices);

    assert_eq!(result, values, "Identity mapping should preserve order");
}

/// Test unsort with reverse mapping.
///
/// Verifies that reverse indices reverse order.
#[test]
fn test_unsort_reverse() {
    let values = vec![10.0, 20.0, 30.0];
    let indices = vec![2, 1, 0]; // Reverse mapping

    let result = unsort(&values, &indices);

    assert_eq!(result, vec![30.0, 20.0, 10.0], "Should reverse order");
}

/// Test sorting with single element.
///
/// Verifies correct handling of minimal input.
#[test]
fn test_sort_single() {
    let x = vec![5.0];
    let y = vec![50.0];

    let sorted = sort_by_x(&x, &y);

    assert_eq!(sorted.x, vec![5.0]);
    assert_eq!(sorted.y, vec![50.0]);
    assert_eq!(sorted.indices, vec![0]);
}

/// Test sorting with two elements.
///
/// Verifies correct handling of pair.
#[test]
fn test_sort_two_elements() {
    let x = vec![2.0, 1.0];
    let y = vec![20.0, 10.0];

    let sorted = sort_by_x(&x, &y);

    assert_eq!(sorted.x, vec![1.0, 2.0]);
    assert_eq!(sorted.y, vec![10.0, 20.0]);
    assert_eq!(sorted.indices, vec![1, 0]);
}

/// Test unsort with single element.
///
/// Verifies correct handling of single value.
#[test]
fn test_unsort_single() {
    let values = vec![42.0];
    let indices = vec![0];

    let result = unsort(&values, &indices);

    assert_eq!(result, vec![42.0]);
}

// ============================================================================
// Stability Tests
// ============================================================================

/// Test stable sort with many duplicates.
///
/// Verifies that stable sort preserves original order for equal values.
#[test]
fn test_stable_sort_many_duplicates() {
    let x = vec![1.0, 2.0, 1.0, 2.0, 1.0];
    let y = vec![10.0, 20.0, 11.0, 21.0, 12.0];

    let sorted = sort_by_x(&x, &y);

    // All 1.0 values should come before all 2.0 values
    assert_eq!(sorted.x, vec![1.0, 1.0, 1.0, 2.0, 2.0]);

    // Within each group, original order should be preserved
    // 1.0 values: indices 0, 2, 4 => y values 10, 11, 12
    assert_eq!(sorted.y[0], 10.0);
    assert_eq!(sorted.y[1], 11.0);
    assert_eq!(sorted.y[2], 12.0);

    // 2.0 values: indices 1, 3 => y values 20, 21
    assert_eq!(sorted.y[3], 20.0);
    assert_eq!(sorted.y[4], 21.0);
}

/// Test stable sort with all equal values.
///
/// Verifies that all equal values preserve original order.
#[test]
fn test_stable_sort_all_equal() {
    let x = vec![5.0, 5.0, 5.0, 5.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];

    let sorted = sort_by_x(&x, &y);

    // X should all be 5.0
    assert!(sorted.x.iter().all(|&v| v == 5.0));

    // Y should preserve original order
    assert_eq!(sorted.y, y);
    assert_eq!(sorted.indices, vec![0, 1, 2, 3]);
}

// ============================================================================
// Roundtrip Tests
// ============================================================================

/// Test sort-unsort roundtrip preserves data.
///
/// Verifies that sorting and unsorting returns to original order.
#[test]
fn test_sort_unsort_roundtrip() {
    let x = vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0];
    let y = vec![50.0, 20.0, 80.0, 10.0, 90.0, 30.0];

    // Sort
    let sorted = sort_by_x(&x, &y);

    // Unsort
    let x_restored = unsort(&sorted.x, &sorted.indices);
    let y_restored = unsort(&sorted.y, &sorted.indices);

    // Should match original
    assert_eq!(x_restored, x, "X should be restored");
    assert_eq!(y_restored, y, "Y should be restored");
}

/// Test roundtrip with large dataset.
///
/// Verifies correctness at scale.
#[test]
fn test_roundtrip_large_dataset() {
    let n = 1000;
    let x: Vec<f64> = (0..n).map(|i| ((i * 7) % n) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64) * 2.0).collect();

    let sorted = sort_by_x(&x, &y);
    let x_restored = unsort(&sorted.x, &sorted.indices);
    let y_restored = unsort(&sorted.y, &sorted.indices);

    // Should match original exactly
    for i in 0..n {
        assert_relative_eq!(x_restored[i], x[i], epsilon = 1e-12);
        assert_relative_eq!(y_restored[i], y[i], epsilon = 1e-12);
    }
}

/// Test that sorted x is actually sorted.
///
/// Verifies ascending order of finite values.
#[test]
fn test_sorted_x_is_sorted() {
    let x = vec![9.0, 3.0, 7.0, 1.0, 5.0, 2.0, 8.0, 4.0, 6.0];
    let y = vec![90.0, 30.0, 70.0, 10.0, 50.0, 20.0, 80.0, 40.0, 60.0];

    let sorted = sort_by_x(&x, &y);

    // Check that finite values are in ascending order
    let finite_values: Vec<f64> = sorted.x.iter().filter(|v| v.is_finite()).copied().collect();

    for i in 1..finite_values.len() {
        assert!(
            finite_values[i - 1] <= finite_values[i],
            "Values should be in ascending order"
        );
    }
}
// ============================================================================
// Additional Edge Case Tests
// ============================================================================

/// Test sorting with very large values.
#[test]
fn test_sort_large_values() {
    let x = vec![f64::MAX, f64::MIN, 0.0, 1.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];

    let sorted = sort_by_x(&x, &y);

    // Should handle extreme values correctly
    assert_eq!(sorted.x[0], f64::MIN);
    assert_eq!(sorted.x[1], 0.0);
    assert_eq!(sorted.x[2], 1.0);
    assert_eq!(sorted.x[3], f64::MAX);
}

/// Test sorting with subnormal values.
#[test]
fn test_sort_subnormal_values() {
    let x = vec![f64::MIN_POSITIVE, 0.0, -f64::MIN_POSITIVE, 1.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];

    let sorted = sort_by_x(&x, &y);

    // Should handle very small values correctly
    assert!(sorted.x[0] < 0.0);
    assert_eq!(sorted.x[1], 0.0);
    assert!(sorted.x[2] > 0.0 && sorted.x[2] < 1.0);
    assert_eq!(sorted.x[3], 1.0);
}

/// Test unsort with permutation that swaps pairs.
#[test]
fn test_unsort_swap_pairs() {
    let values = vec![10.0, 20.0, 30.0, 40.0];
    let indices = vec![1, 0, 3, 2]; // Swap pairs

    let result = unsort(&values, &indices);

    assert_eq!(result, vec![20.0, 10.0, 40.0, 30.0]);
}

/// Test sorting preserves exact floating point values.
#[test]
fn test_sort_exact_float_preservation() {
    // Use values that might have precision issues
    let x = vec![0.1 + 0.2, 0.3, 0.1, 0.2];
    let y = vec![1.0, 2.0, 3.0, 4.0];

    let sorted = sort_by_x(&x, &y);
    let restored = unsort(&sorted.x, &sorted.indices);

    // Should preserve exact bit patterns (use relative comparison for floats)
    for i in 0..x.len() {
        assert_relative_eq!(restored[i], x[i], epsilon = 0.0);
    }
}

/// Test sorting with negative zero.
#[test]
fn test_sort_negative_zero() {
    let x = vec![0.0, -0.0, 1.0, -1.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];

    let sorted = sort_by_x(&x, &y);

    // Both zeros should be sorted together
    assert!(sorted.x[1].abs() < 1e-10);
    assert!(sorted.x[2].abs() < 1e-10);
}
