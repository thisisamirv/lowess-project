#![cfg(feature = "dev")]
//! Tests for Cross-Validation (CV) utilities.
//!
//! These tests verify the cross-validation functionality used in LOWESS for:
//! - Automatic parameter selection (fraction, iterations)
//! - Model performance evaluation
//! - Prediction interpolation for held-out points
//!
//! ## Test Organization
//!
//! 1. **Interpolation** - Prediction interpolation for CV
//! 2. **Subset Building** - Training/test set construction
//! 3. **Edge Cases** - Empty, single, and duplicate value handling

use approx::assert_relative_eq;

use lowess::internals::evaluation::cv::CVKind;

// ============================================================================
// Interpolation Tests
// ============================================================================

/// Test basic interpolation for CV predictions.
///
/// Verifies linear interpolation between fitted points.
#[test]
fn test_interpolate_prediction_basic() {
    let tx = vec![1.0, 3.0, 4.0];
    let ty = vec![2.0, 6.0, 8.0];

    // Interpolation inside range
    // Between 3.0=>6.0 and 4.0=>8.0, at 3.5 should be 7.0
    let pred = CVKind::interpolate_prediction(&tx, &ty, 3.5);
    assert_relative_eq!(pred, 7.0, epsilon = 1e-12);
}

/// Test extrapolation beyond data range.
///
/// Verifies that extrapolation uses nearest value.
#[test]
fn test_interpolate_prediction_extrapolation() {
    let tx = vec![1.0, 3.0, 4.0];
    let ty = vec![2.0, 6.0, 8.0];

    // Extrapolation before range => use first value
    assert_relative_eq!(
        CVKind::interpolate_prediction(&tx, &ty, 0.0),
        2.0,
        epsilon = 1e-12
    );

    // Extrapolation after range => use last value
    assert_relative_eq!(
        CVKind::interpolate_prediction(&tx, &ty, 10.0),
        8.0,
        epsilon = 1e-12
    );
}

/// Test interpolation with duplicate x values.
///
/// Verifies handling when x values are not unique.
#[test]
fn test_interpolate_prediction_duplicates() {
    let tx = vec![1.0f64, 1.0, 2.0];
    let ty = vec![3.0f64, 5.0, 7.0];

    // Searching for x_new at duplicate entries should handle denom <= 0
    let pred = CVKind::interpolate_prediction(&tx, &ty, 1.0);

    // Should return finite value (averaging or using one of the values)
    assert!(pred.is_finite(), "Prediction should be finite");
    assert!(
        (3.0..=5.0).contains(&pred),
        "Prediction should be within y range at duplicates"
    );
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

/// Test interpolation with empty arrays.
///
/// Verifies that empty input returns 0.
#[test]
fn test_interpolate_prediction_empty() {
    let empty_x: Vec<f64> = vec![];
    let empty_y: Vec<f64> = vec![];

    let pred = CVKind::interpolate_prediction(&empty_x, &empty_y, 1.0);
    assert_relative_eq!(pred, 0.0, epsilon = 1e-12);
}

/// Test interpolation with single element.
///
/// Verifies that single value is returned for any query point.
#[test]
fn test_interpolate_prediction_single() {
    let single_x = vec![5.0f64];
    let single_y = vec![10.0f64];

    // Any query point should return the single y value
    assert_relative_eq!(
        CVKind::interpolate_prediction(&single_x, &single_y, 3.0),
        10.0,
        epsilon = 1e-12
    );

    assert_relative_eq!(
        CVKind::interpolate_prediction(&single_x, &single_y, 7.0),
        10.0,
        epsilon = 1e-12
    );
}

/// Test interpolation with all duplicate x values.
///
/// Verifies handling when all x values are identical.
#[test]
fn test_interpolate_prediction_all_duplicates() {
    let dup_x = vec![2.0f64, 2.0, 2.0];
    let dup_y = vec![3.0f64, 5.0, 7.0];

    let pred = CVKind::interpolate_prediction(&dup_x, &dup_y, 2.0);

    assert!(pred.is_finite(), "Prediction should be finite");
    assert!(
        (3.0..=7.0).contains(&pred),
        "Prediction should be within y range"
    );
}

// ============================================================================
// Subset Building Tests
// ============================================================================

/// Test building subset from indices.
///
/// Verifies correct extraction of training data.
#[test]
fn test_build_subset_basic() {
    let x = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0f64, 4.0, 6.0, 8.0, 10.0];
    let indices = vec![0, 2, 4]; // Use points 0, 2, 4

    let (x_train, y_train) = CVKind::build_subset_from_indices(&x, &y, 1, &indices);

    assert_eq!(
        x_train,
        vec![1.0, 3.0, 5.0],
        "X subset should match indices"
    );
    assert_eq!(
        y_train,
        vec![2.0, 6.0, 10.0],
        "Y subset should match indices"
    );
}

/// Test subset building with empty indices.
///
/// Verifies that empty indices produce empty subset.
#[test]
fn test_build_subset_empty_indices() {
    let x = vec![1.0f64, 2.0, 3.0];
    let y = vec![2.0f64, 4.0, 6.0];
    let indices: Vec<usize> = vec![];

    let (x_train, y_train) = CVKind::build_subset_from_indices(&x, &y, 1, &indices);

    assert!(x_train.is_empty(), "X subset should be empty");
    assert!(y_train.is_empty(), "Y subset should be empty");
}

/// Test subset building with all indices.
///
/// Verifies that using all indices returns full dataset.
#[test]
fn test_build_subset_all_indices() {
    let x = vec![1.0f64, 2.0, 3.0];
    let y = vec![2.0f64, 4.0, 6.0];
    let indices = vec![0, 1, 2];

    let (x_train, y_train) = CVKind::build_subset_from_indices(&x, &y, 1, &indices);

    assert_eq!(x_train, x, "X subset should match full dataset");
    assert_eq!(y_train, y, "Y subset should match full dataset");
}

/// Test subset building with single index.
///
/// Verifies correct extraction of single point.
#[test]
fn test_build_subset_single_index() {
    let x = vec![1.0f64, 2.0, 3.0];
    let y = vec![2.0f64, 4.0, 6.0];
    let indices = vec![1];

    let (x_train, y_train) = CVKind::build_subset_from_indices(&x, &y, 1, &indices);

    assert_eq!(x_train, vec![2.0], "X subset should have single value");
    assert_eq!(y_train, vec![4.0], "Y subset should have single value");
}

/// Test interpolation at exact data points.
///
/// Verifies that interpolation at exact x values returns corresponding y.
#[test]
fn test_interpolate_at_exact_points() {
    let tx = vec![1.0, 2.0, 3.0];
    let ty = vec![10.0, 20.0, 30.0];

    // At exact points, should return exact y values
    assert_relative_eq!(
        CVKind::interpolate_prediction(&tx, &ty, 1.0),
        10.0,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        CVKind::interpolate_prediction(&tx, &ty, 2.0),
        20.0,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        CVKind::interpolate_prediction(&tx, &ty, 3.0),
        30.0,
        epsilon = 1e-12
    );
}

/// Test interpolation with unsorted data.
///
/// Verifies that interpolation works with unsorted x values.
#[test]
fn test_interpolate_unsorted() {
    // Note: Implementation may require sorted data
    // This test verifies behavior with unsorted input
    let tx = vec![3.0, 1.0, 2.0];
    let ty = vec![30.0, 10.0, 20.0];

    let pred: f64 = CVKind::interpolate_prediction(&tx, &ty, 1.5);

    // Should return finite value
    assert!(pred.is_finite(), "Prediction should be finite");
}

// ============================================================================
// CV Method Edge Cases
// ============================================================================

/// Test CVKind::run with various edge cases for k and fractions.
#[test]
fn test_cv_method_run_edge_cases() {
    let smoother = |_: &[f64], _: &[f64], _: f64| vec![0.0; 5];

    // 1. n < 2 in LOOCV
    let x_mini = vec![1.0];
    let y_mini = vec![2.0];
    use lowess::internals::primitives::buffer::CVBuffer;
    let mut cv_buffer = CVBuffer::new();
    let (best_mini, _) = CVKind::LOOCV.run(
        &x_mini,
        &y_mini,
        1,
        &[0.5, 0.8],
        None,
        smoother,
        None::<fn(&[f64], &[f64], &[f64], f64) -> Vec<f64>>,
        &mut cv_buffer,
    );
    assert_eq!(best_mini, 0.5);
}

/// Test k-fold cross-validation when k > n.
#[test]
fn test_kfold_insufficient_data() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];
    let fractions = vec![0.5, 0.8];
    let smoother = |_: &[f64], _: &[f64], _: f64| vec![0.0; 3];

    // k=5 > n=3. This should work as the last fold will just cover the remaining points (which might be empty or small).
    // The implementation n/k = 0 for k > n.
    // fold_size = 0.
    // fold 0..4: test_start = f*0 = 0. test_end = 0.
    // Last fold (k-1): test_start = 0, test_end = 3.
    use lowess::internals::primitives::buffer::CVBuffer;
    let mut cv_buffer = CVBuffer::new();
    let (best, scores) = CVKind::KFold(5).run(
        &x,
        &y,
        1,
        &fractions,
        None,
        smoother,
        None::<fn(&[f64], &[f64], &[f64], f64) -> Vec<f64>>,
        &mut cv_buffer,
    );
    assert!(best > 0.0);
    assert_eq!(scores.len(), 2);
}

/// Test leave-one-out cross-validation with minimal data.
#[test]
fn test_loocv_minimal_data() {
    let x = vec![1.0, 2.0];
    let y = vec![10.0, 20.0];
    let fractions = vec![0.5];

    // Smoother returns original y
    let smoother = |_: &[f64], y: &[f64], _: f64| y.to_vec();

    use lowess::internals::primitives::buffer::CVBuffer;
    let mut cv_buffer = CVBuffer::new();
    let (best, scores) = CVKind::LOOCV.run(
        &x,
        &y,
        1,
        &fractions,
        None,
        smoother,
        None::<fn(&[f64], &[f64], &[f64], f64) -> Vec<f64>>,
        &mut cv_buffer,
    );
    assert_eq!(best, 0.5);
    assert_eq!(scores.len(), 1);

    // LOOCV on n=2:
    // Fold 0: train on [2.0], test on [1.0]. Predict at 1.0 using single point train => 20.0. Residual = 10-20 = -10.
    // Fold 1: train on [1.0], test on [2.0]. Predict at 2.0 using single point train => 10.0. Residual = 20-10 = 10.
    // RMSE = sqrt((100 + 100) / 2) = sqrt(100) = 10.
    assert_relative_eq!(scores[0], 10.0, epsilon = 1e-12);
}
