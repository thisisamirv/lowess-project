#![cfg(feature = "dev")]
//! Tests for diagnostic metrics computation.
//!
//! These tests verify the diagnostic metrics used in LOWESS for:
//! - Model quality assessment (RMSE, MAE, R²)
//! - Residual analysis
//! - Model selection (AIC, AICc)
//! - Effective degrees of freedom
//! - Interval coverage
//!
//! ## Test Organization
//!
//! 1. **Basic Metrics** - RMSE, MAE, R² computation
//! 2. **Residual Statistics** - Residual SD and effective DF
//! 3. **Model Selection** - AIC and AICc
//! 4. **Coverage** - Interval coverage computation
//! 5. **Edge Cases** - Empty, single, and special inputs

use approx::{assert_abs_diff_eq, assert_relative_eq};

use lowess::internals::evaluation::diagnostics::{Diagnostics, DiagnosticsState};
use lowess::internals::math::scaling::ScalingMethod;

const MIN_TUNED_SCALE: f64 = 1e-12;
const MAD_TO_STD_FACTOR: f64 = 1.4826;

// ============================================================================
// Basic Metrics Tests
// ============================================================================

/// Test RMSE and MAE with single point.
///
/// Verifies correct calculation for minimal case.
#[test]
fn test_rmse_mae_single() {
    let y = vec![5.0f64];
    let ys = vec![4.0f64];

    // Residual = 1 => RMSE = 1, MAE = 1
    assert_relative_eq!(
        Diagnostics::calculate_rmse(&y, &ys),
        1.0f64,
        epsilon = 1e-12
    );
    assert_relative_eq!(Diagnostics::calculate_mae(&y, &ys), 1.0f64, epsilon = 1e-12);
}

/// Test RMSE and MAE with typical values.
///
/// Verifies correct computation for normal case.
#[test]
fn test_rmse_mae_typical() {
    let y = vec![0.0f64, 2.0, 4.0];
    let ys = vec![0.0f64, 1.0, 3.0];

    // Residuals: [0, 1, 1]
    // MAE = (0 + 1 + 1) / 3 = 2/3
    // RSS = 0 + 1 + 1 = 2
    // RMSE = sqrt(2/3)
    assert_relative_eq!(
        Diagnostics::calculate_mae(&y, &ys),
        2.0f64 / 3.0f64,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        Diagnostics::calculate_rmse(&y, &ys),
        (2.0f64 / 3.0f64).sqrt(),
        epsilon = 1e-12
    );
}

/// Test R² with single point.
///
/// Verifies that single point returns 1.
#[test]
fn test_r_squared_single() {
    let y = vec![2.0f64];
    let ys = vec![1.0f64];

    assert_relative_eq!(
        Diagnostics::calculate_r_squared(&y, &ys),
        1.0f64,
        epsilon = 1e-12
    );
}

/// Test R² with perfect fit.
///
/// Verifies that perfect fit returns 1.
#[test]
fn test_r_squared_perfect_fit() {
    let y = vec![3.0f64, 3.0, 3.0];
    let ys = vec![3.0f64, 3.0, 3.0];

    assert_relative_eq!(
        Diagnostics::calculate_r_squared(&y, &ys),
        1.0f64,
        epsilon = 1e-12
    );
}

/// Test R² with zero total variance.
///
/// Verifies handling when all y values are identical.
#[test]
fn test_r_squared_zero_variance() {
    let y_eq = vec![3.0f64, 3.0, 3.0];

    // Imperfect fit with zero variance => R² = 0
    let ys_bad = vec![2.0f64, 3.0, 4.0];
    assert_relative_eq!(
        Diagnostics::calculate_r_squared(&y_eq, &ys_bad),
        0.0f64,
        epsilon = 1e-12
    );
}

/// Test R² with typical case.
///
/// Verifies correct R² calculation.
#[test]
fn test_r_squared_typical() {
    // y = [1, 2, 3], ys = [1, 2, 2]
    // Mean(y) = 2
    // SS_res = (1-1)² + (2-2)² + (3-2)² = 1
    // SS_tot = (1-2)² + (2-2)² + (3-2)² = 2
    // R² = 1 - 1/2 = 0.5
    let y = vec![1.0f64, 2.0, 3.0];
    let ys = vec![1.0f64, 2.0, 2.0];

    assert_relative_eq!(
        Diagnostics::calculate_r_squared(&y, &ys),
        0.5f64,
        epsilon = 1e-12
    );
}

// ============================================================================
// Residual Statistics Tests
// ============================================================================

/// Test residual SD with zero residuals.
///
/// Verifies that zero residuals return zero SD.
#[test]
fn test_residual_sd_zeros() {
    let zeros = vec![0.0f64; 4];

    assert_abs_diff_eq!(
        Diagnostics::calculate_residual_sd(&zeros),
        0.0f64,
        epsilon = 1e-11
    );
}

/// Test residual SD with typical residuals.
///
/// Verifies correct SD calculation using MAD.
#[test]
fn test_residual_sd_typical() {
    let res = vec![0.0f64, 1.0, -1.0];
    let sd = Diagnostics::calculate_residual_sd(&res);

    assert!(sd >= 0.0, "SD should be non-negative");
    assert!(sd.is_finite(), "SD should be finite");
}

/// Test residual SD with single element.
///
/// Verifies correct calculation for single residual.
#[test]
fn test_residual_sd_single() {
    let single = vec![-2.0f64];

    assert_relative_eq!(
        Diagnostics::calculate_residual_sd(&single),
        single[0].abs() * MAD_TO_STD_FACTOR,
        epsilon = 1e-12
    );
}

/// Test residual SD with normal case.
///
/// Verifies that SD equals MAD * factor.
#[test]
fn test_residual_sd_normal() {
    let res = vec![0.0f64, 1.0, -1.0, 2.0, -2.0];
    let mad = ScalingMethod::MAD.compute(&mut res.clone());

    assert!(mad > 0.0, "MAD should be positive");
    assert_relative_eq!(
        Diagnostics::calculate_residual_sd(&res),
        mad * MAD_TO_STD_FACTOR,
        epsilon = 1e-12
    );
}

/// Test residual SD when MAD is zero.
///
/// Verifies fallback to minimum scale.
#[test]
fn test_residual_sd_mad_zero() {
    // All-equal residuals => MAD == 0
    let mut vals = vec![2.0f64, 2.0, 2.0, 2.0];
    assert_eq!(ScalingMethod::MAD.compute(&mut vals), 0.0);

    let expected = MIN_TUNED_SCALE * MAD_TO_STD_FACTOR;
    assert_relative_eq!(
        Diagnostics::calculate_residual_sd(&vals),
        expected,
        epsilon = 1e-12
    );
}

/// Test residual SD with large arrays.
///
/// Verifies numerical stability for large datasets.
#[test]
fn test_residual_sd_large_arrays() {
    let n = 10000;
    let residuals: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();

    let mad = ScalingMethod::MAD.compute(&mut residuals.clone());
    assert!(mad > 0.0, "MAD should be positive");

    let mad_scale = Diagnostics::calculate_residual_sd(&residuals);
    assert!(mad_scale > 0.0, "SD should be positive");
    assert_relative_eq!(mad_scale / mad, MAD_TO_STD_FACTOR, epsilon = 1e-12);
}

/// Test effective DF calculation.
///
/// Verifies sum of leverages.
#[test]
fn test_effective_df() {
    let leverage = vec![0.2f64, 0.3, 0.5];
    let eff = Diagnostics::calculate_effective_df_from_leverages(&leverage);

    // Sum = 0.2 + 0.3 + 0.5 = 1.0
    assert_relative_eq!(eff, 1.0f64, epsilon = 1e-12);
}

// ============================================================================
// Model Selection Tests
// ============================================================================

/// Test AIC calculation.
///
/// Verifies AIC formula.
#[test]
fn test_aic_calculation() {
    // RSS/n == 1 => ln(1) == 0
    // With effective_df = 1 => AIC = 2 * eff_df = 2
    let residuals = vec![1.0f64, 1.0];
    let aic = Diagnostics::calculate_aic(&residuals, 1.0f64);

    assert_relative_eq!(aic, 2.0f64, epsilon = 1e-12);
}

/// Test AIC with zero RSS.
///
/// Verifies that zero RSS produces infinite AIC.
#[test]
fn test_aic_zero_rss() {
    let zeros = vec![0.0f64; 3];
    let aic = Diagnostics::calculate_aic(&zeros, 1.0f64);

    assert!(aic.is_infinite(), "AIC should be infinite for zero RSS");
}

/// Test AICc calculation.
///
/// Verifies AICc includes small-sample correction.
#[test]
fn test_aicc_calculation() {
    let residuals = vec![1.0f64, 1.0];
    let aicc = Diagnostics::calculate_aicc(&residuals, 1.0f64);

    // With n=2, eff_df=1: denominator = n - eff_df - 1 = 0 => infinite
    assert!(
        aicc.is_infinite(),
        "AICc should be infinite when denominator is zero"
    );
}

/// Test AICc with sufficient sample size.
///
/// Verifies AICc is finite for adequate n.
#[test]
fn test_aicc_sufficient_sample() {
    let residuals = vec![1.0f64; 10];
    let aicc = Diagnostics::calculate_aicc(&residuals, 2.0f64);

    assert!(
        aicc.is_finite(),
        "AICc should be finite for sufficient sample size"
    );
}

// ============================================================================
// Diagnostics State and Finalize Edge Cases
// ============================================================================

/// Test DiagnosticsState::finalize with empty input.
#[test]
fn test_diagnostics_state_empty() {
    let state = DiagnosticsState::<f64>::new();
    let diag = state.finalize();

    assert_eq!(diag.rmse, 0.0);
    assert_eq!(diag.mae, 0.0);
    assert_eq!(diag.r_squared, 0.0);
    assert!(diag.aic.is_none());
}

/// Test AICc behavior when degrees of freedom approach sample size.
#[test]
fn test_aicc_correction_limit() {
    let residuals = vec![1.0f64; 5];
    // n=5, eff_df=3 => denom = 5 - 3 - 1 = 1. Finite.
    let aicc_ok: f64 = Diagnostics::calculate_aicc(&residuals, 3.0);
    assert!(aicc_ok.is_finite());

    // n=5, eff_df=4 => denom = 5 - 4 - 1 = 0. Infinite.
    let aicc_inf: f64 = Diagnostics::calculate_aicc(&residuals, 4.0);
    assert!(aicc_inf.is_infinite());
}

/// Test diagnostics finalize for n=1.
#[test]
fn test_diagnostics_finalize_single() {
    let mut state = DiagnosticsState::<f64>::new();
    state.update(&[10.0], &[9.0]); // Residual = 1

    let diag = state.finalize();
    assert_eq!(diag.rmse, 1.0);
    assert_eq!(diag.mae, 1.0);

    // For n=1 in finalize: ss_tot = 100 - (100)/1 = 0.
    // If ss_tot is zero and rss != 0 -> R2 = 0.0 in finalize (line 198)
    assert_eq!(diag.r_squared, 0.0);
}
