//! Diagnostic metrics for LOWESS fit quality assessment.
//!
//! ## Purpose
//!
//! This module provides comprehensive diagnostic tools for evaluating the
//! quality of LOWESS smoothing results. It computes goodness-of-fit metrics,
//! model selection criteria, and coverage statistics.
//!
//! ## Design notes
//!
//! * **Residual-based**: Metrics are computed from residuals (y - ŷ) and fitted values.
//! * **Robustness**: Uses MAD (Median Absolute Deviation) for robust scale estimation.
//! * **Model selection**: AIC and AICc provide criteria for bandwidth tuning.
//! * **Generics**: All computations are generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Residual Metrics**: RSS, RMSE, and MAE measure prediction error.
//! * **Goodness-of-Fit**: R^2 measures variance explained by the smoother.
//! * **Effective DF**: Trace of the smoother matrix, approximated from standard errors.
//! * **Model Selection**: AIC/AICc trade off fit quality and model complexity.
//!
//! ## Invariants
//!
//! * Error metrics (RMSE, MAE, RSS) and df_eff are non-negative.
//! * R^2 <= 1 (R^2 = 1 is a perfect fit).
//! * Residual SD is robustly estimated using MAD * 1.4826 (Direct mode) or standard variance (Streaming).
//!
//! ## Non-goals
//!
//! * This module does not perform the smoothing or optimization.
//! * This module does not provide p-values or formal hypothesis tests.
//! * This module does not compute weighted diagnostic metrics.

// External dependencies
use core::fmt::{Display, Formatter, Result};
use num_traits::Float;

// Internal dependencies
use crate::math::scaling::ScalingMethod;

// ============================================================================
// Diagnostics Structure
// ============================================================================

/// Diagnostic metrics for assessing LOWESS fit quality.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics<T> {
    /// Root Mean Squared Error (RMSE).
    pub rmse: T,

    /// Mean Absolute Error (MAE).
    pub mae: T,

    /// Coefficient of determination (R^2).
    pub r_squared: T,

    /// Akaike Information Criterion (AIC).
    pub aic: Option<T>,

    /// Corrected AIC with finite-sample correction (AICc).
    pub aicc: Option<T>,

    /// Estimated effective degrees of freedom (df_eff).
    pub effective_df: Option<T>,

    /// Robust residual standard deviation estimated from MAD.
    pub residual_sd: T,
}

/// Cumulative state for computing diagnostics in streaming mode.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticsState<T> {
    /// Total number of observations.
    pub n: usize,
    /// Sum of y values.
    pub sum_y: T,
    /// Sum of squared y values.
    pub sum_y_sq: T,
    /// Sum of residuals (y - ŷ).
    pub sum_r: T,
    /// Sum of squared residuals.
    pub sum_r_sq: T,
    /// Sum of absolute residuals.
    pub sum_abs_r: T,
}

impl<T: Float> Default for DiagnosticsState<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> DiagnosticsState<T> {
    /// Create a new, empty diagnostics state.
    pub fn new() -> Self {
        Self {
            n: 0,
            sum_y: T::zero(),
            sum_y_sq: T::zero(),
            sum_r: T::zero(),
            sum_r_sq: T::zero(),
            sum_abs_r: T::zero(),
        }
    }

    /// Update the state with a new chunk of data.
    pub fn update(&mut self, y: &[T], y_smooth: &[T]) {
        for (&yi, &ys) in y.iter().zip(y_smooth.iter()) {
            let r = yi - ys;
            self.n += 1;
            self.sum_y = self.sum_y + yi;
            self.sum_y_sq = self.sum_y_sq + yi * yi;
            self.sum_r = self.sum_r + r;
            self.sum_r_sq = self.sum_r_sq + r * r;
            self.sum_abs_r = self.sum_abs_r + r.abs();
        }
    }

    /// Compute final diagnostics from the accumulated state.
    pub fn finalize(&self) -> Diagnostics<T> {
        let n_t = T::from(self.n).unwrap_or(T::one());
        if self.n == 0 {
            return Diagnostics {
                rmse: T::zero(),
                mae: T::zero(),
                r_squared: T::zero(),
                aic: None,
                aicc: None,
                effective_df: None,
                residual_sd: T::zero(),
            };
        }

        let rmse = (self.sum_r_sq / n_t).sqrt();
        let mae = self.sum_abs_r / n_t;

        // R-squared: 1 - SS_res / SS_tot
        let ss_tot = self.sum_y_sq - (self.sum_y * self.sum_y) / n_t;
        let r_squared = if ss_tot > T::from(1e-12).unwrap() * self.sum_y_sq.abs() {
            T::one() - self.sum_r_sq / ss_tot
        } else if self.sum_r_sq < T::from(1e-12).unwrap() * self.sum_y_sq.abs()
            || self.sum_r_sq == T::zero()
        {
            T::one()
        } else {
            T::zero()
        };

        // Residual SD: estimated from global variance of residuals
        // Var(r) = (sum_r_sq - (sum_r)^2 / n) / (n - 1)
        let residual_sd = if self.n > 1 {
            let var_r = (self.sum_r_sq - (self.sum_r * self.sum_r) / n_t) / (n_t - T::one());
            var_r.max(T::zero()).sqrt()
        } else {
            rmse
        };

        Diagnostics {
            rmse,
            mae,
            r_squared,
            aic: None,
            aicc: None,
            effective_df: None,
            residual_sd,
        }
    }
}

impl<T: Float> Diagnostics<T> {
    // ========================================================================
    // Constants
    // ========================================================================

    /// Number of parameters in local linear regression (intercept + slope).
    const LINEAR_PARAMS: f64 = 2.0;

    /// Minimum tuned-scale absolute epsilon to avoid division-by-zero.
    const MIN_TUNED_SCALE: f64 = 1e-12;

    /// Constant to convert MAD to an unbiased estimate of sigma for normal data.
    ///
    /// For normally distributed data, MAD × 1.4826 ≈ standard deviation.
    const MAD_TO_STD_FACTOR: f64 = 1.4826;

    // ========================================================================
    // Main Computation
    // ========================================================================

    /// Compute diagnostic statistics from fit results.
    pub fn compute(y: &[T], y_smooth: &[T], residuals: &[T], std_errors: Option<&[T]>) -> Self {
        let rmse = Self::calculate_rmse(y, y_smooth);
        let mae = Self::calculate_mae(y, y_smooth);
        let r_squared = Self::calculate_r_squared(y, y_smooth);
        let residual_sd = Self::calculate_residual_sd(residuals);

        let effective_df = std_errors.and_then(|se| Self::calculate_effective_df(se, residual_sd));

        let (aic, aicc) = if let Some(df) = effective_df {
            (
                Some(Self::calculate_aic(residuals, df)),
                Some(Self::calculate_aicc(residuals, df)),
            )
        } else {
            (None, None)
        };

        Diagnostics {
            rmse,
            mae,
            r_squared,
            aic,
            aicc,
            effective_df,
            residual_sd,
        }
    }

    // ========================================================================
    // Error Metrics
    // ========================================================================

    /// Compute the residual sum of squares (RSS).
    /// RSS = sum r_i^2.
    fn calculate_rss(residuals: &[T]) -> T {
        residuals.iter().fold(T::zero(), |acc, &r| acc + r * r)
    }

    /// Compute the root mean squared error (RMSE).
    /// RMSE = sqrt((1/n) * sum (y_i - y_hat_i)^2).
    pub fn calculate_rmse(y: &[T], y_smooth: &[T]) -> T {
        let n_t = T::from(y.len()).unwrap_or(T::one());
        let rss = y
            .iter()
            .zip(y_smooth.iter())
            .fold(T::zero(), |acc, (&yi, &ys)| {
                let r = yi - ys;
                acc + r * r
            });

        (rss / n_t).sqrt()
    }

    /// Compute the mean absolute error (MAE).
    /// MAE = (1/n) * sum |y_i - y_hat_i|.
    pub fn calculate_mae(y: &[T], y_smooth: &[T]) -> T {
        let n_t = T::from(y.len()).unwrap_or(T::one());
        let sum = y
            .iter()
            .zip(y_smooth.iter())
            .fold(T::zero(), |acc, (&yi, &ys)| acc + (yi - ys).abs());

        sum / n_t
    }

    // ========================================================================
    // Goodness-of-Fit Metrics
    // ========================================================================

    /// Compute the coefficient of determination (R^2).
    /// R^2 = 1 - SS_res / SS_tot, where SS_res is the residual
    /// sum of squares and SS_tot is the total sum of squares.
    pub fn calculate_r_squared(y: &[T], y_smooth: &[T]) -> T {
        let n = y.len();
        if n == 1 {
            return T::one();
        }

        let n_t = T::from(n).unwrap_or(T::one());

        // Compute mean
        let sum = y.iter().copied().fold(T::zero(), |acc, v| acc + v);
        let mean = sum / n_t;

        // Compute SS_tot and SS_res in one pass
        let (ss_tot, ss_res) =
            y.iter()
                .zip(y_smooth.iter())
                .fold((T::zero(), T::zero()), |(tot, res), (&yi, &ys)| {
                    let deviation = yi - mean;
                    let residual = yi - ys;
                    (tot + deviation * deviation, res + residual * residual)
                });

        if ss_tot == T::zero() {
            // All y values are identical
            if ss_res == T::zero() {
                T::one() // Perfect fit
            } else {
                T::zero() // No variance to explain
            }
        } else {
            T::one() - ss_res / ss_tot
        }
    }

    // ========================================================================
    // Robust Scale Estimation
    // ========================================================================

    /// Estimate the residual standard deviation using a robust method.
    /// sigma_hat = 1.4826 * MAD(residuals).
    pub fn calculate_residual_sd(residuals: &[T]) -> T {
        let n = residuals.len();
        let scale_const = T::from(Self::MAD_TO_STD_FACTOR).unwrap();

        if n == 1 {
            return residuals[0].abs() * scale_const;
        }

        let mut vals = residuals.to_vec();
        let mad = ScalingMethod::MAD.compute(&mut vals);
        if mad > T::zero() {
            mad * scale_const
        } else {
            // Apply minimum scale to avoid division by zero
            let min_eps = T::from(Self::MIN_TUNED_SCALE).unwrap();
            min_eps * scale_const
        }
    }

    // ========================================================================
    // Model Complexity
    // ========================================================================

    /// Estimate the effective degrees of freedom from standard errors.
    /// df_eff approx sum (SE_i / sigma_hat)^2.
    fn calculate_effective_df(std_errors: &[T], residual_sd: T) -> Option<T> {
        if residual_sd <= T::zero() {
            return None;
        }

        let leverage_sum = std_errors.iter().fold(T::zero(), |acc, &se| {
            let ratio = se / residual_sd;
            let mut leverage = ratio * ratio;

            // Clamp leverage to valid range [0, 1]
            if leverage < T::zero() {
                leverage = T::zero();
            } else if leverage > T::one() {
                leverage = T::one();
            }

            acc + leverage
        });

        if leverage_sum > T::zero() {
            Some(leverage_sum)
        } else {
            None
        }
    }

    /// Compute the effective degrees of freedom from leverage values.
    /// df_eff = sum l_ii = tr(L).
    pub fn calculate_effective_df_from_leverages(leverage_values: &[T]) -> T {
        leverage_values
            .iter()
            .copied()
            .fold(T::zero(), |acc, v| acc + v)
    }

    // ========================================================================
    // Model Selection Criteria
    // ========================================================================

    /// Compute the Akaike Information Criterion (AIC).
    /// AIC = n * ln(RSS / n) + 2 * df_eff.
    pub fn calculate_aic(residuals: &[T], effective_df: T) -> T {
        let n = T::from(residuals.len()).unwrap_or(T::one());
        let rss = Self::calculate_rss(residuals);

        if rss <= T::zero() || n <= T::zero() {
            return T::infinity();
        }

        n * (rss / n).ln() + T::from(Self::LINEAR_PARAMS).unwrap() * effective_df
    }

    /// Compute the corrected Akaike Information Criterion (AICc).
    /// AICc = AIC + (2 * df_eff * (df_eff + 1)) / (n - df_eff - 1).
    pub fn calculate_aicc(residuals: &[T], effective_df: T) -> T {
        let n = T::from(residuals.len()).unwrap_or(T::one());
        let aic = Self::calculate_aic(residuals, effective_df);
        let denom = n - effective_df - T::one();

        if denom <= T::zero() {
            return T::infinity();
        }

        let correction =
            (T::from(Self::LINEAR_PARAMS).unwrap() * effective_df * (effective_df + T::one()))
                / denom;

        aic + correction
    }
}

// ============================================================================
// Display Implementation
// ============================================================================

impl<T: Float + Display> Display for Diagnostics<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "LOWESS Diagnostics:")?;
        writeln!(f, "  RMSE:         {:.6}", self.rmse)?;
        writeln!(f, "  MAE:          {:.6}", self.mae)?;
        writeln!(f, "  R²:           {:.6}", self.r_squared)?;
        writeln!(f, "  Residual SD:  {:.6}", self.residual_sd)?;

        if let Some(df) = self.effective_df {
            writeln!(f, "  Effective DF: {:.2}", df)?;
        }
        if let Some(aic) = self.aic {
            writeln!(f, "  AIC:          {:.2}", aic)?;
        }
        if let Some(aicc) = self.aicc {
            writeln!(f, "  AICc:         {:.2}", aicc)?;
        }

        Ok(())
    }
}
