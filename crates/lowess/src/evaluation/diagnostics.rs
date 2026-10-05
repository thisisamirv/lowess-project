//! Diagnostic metrics for LOWESS fit quality assessment.
//!
//! This module provides comprehensive diagnostic tools for evaluating the
//! quality of LOWESS smoothing results. It computes goodness-of-fit metrics,
//! model selection criteria, and coverage statistics.
// ## srrstats Compliance
//
// @srrstats {RE3.0} Goodness-of-fit metrics: RMSE, MAE, R2 (coefficient of determination).
// @srrstats {RE3.2} Model selection criteria: AIC, AICc with finite-sample correction.
// @srrstats {G2.6} Standard error estimation from residuals via MAD-based sigma.

// External dependencies
use core::fmt::{Display, Formatter, Result};
use num_traits::Float;

// Internal dependencies
use crate::math::scaling::ScalingMethod;

// Diagnostic metrics for assessing LOWESS fit quality.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics<T> {
    // Root Mean Squared Error (RMSE).
    pub rmse: T,

    // Mean Absolute Error (MAE).
    pub mae: T,

    // Coefficient of determination (R^2).
    pub r_squared: T,

    // Akaike Information Criterion (AIC).
    pub aic: Option<T>,

    // Corrected AIC with finite-sample correction (AICc).
    pub aicc: Option<T>,

    // Estimated effective degrees of freedom (df_eff).
    pub effective_df: Option<T>,

    // Residual scale estimate: Batch uses scaled MAD; Streaming uses sample SD.
    pub residual_sd: T,
}

// Cumulative state for computing diagnostics in streaming mode.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticsState<T> {
    // Total number of observations.
    pub n: usize,
    // Sum of y values.
    pub sum_y: T,
    // Sum of squared y values.
    pub sum_y_sq: T,
    // First response value, used as the origin for centered variance updates.
    pub origin_y: T,
    // Running mean of raw y values, retained alongside the centered accumulator.
    pub mean_y: T,
    // Running mean of response offsets from `origin_y`.
    pub mean_y_offset: T,
    // Centered sum of squares of offsets from `origin_y`, updated with Welford's method.
    pub sum_y_centered_sq: T,
    // Sum of residuals (y - ŷ).
    pub sum_r: T,
    // Sum of squared residuals.
    pub sum_r_sq: T,
    // Sum of absolute residuals.
    pub sum_abs_r: T,
}

impl<T: Float> Default for DiagnosticsState<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> DiagnosticsState<T> {
    // Create a new, empty diagnostics state.
    pub fn new() -> Self {
        Self {
            n: 0,
            sum_y: T::zero(),
            sum_y_sq: T::zero(),
            origin_y: T::zero(),
            mean_y: T::zero(),
            mean_y_offset: T::zero(),
            sum_y_centered_sq: T::zero(),
            sum_r: T::zero(),
            sum_r_sq: T::zero(),
            sum_abs_r: T::zero(),
        }
    }

    // Update the state with a new chunk of data.
    pub fn update(&mut self, y: &[T], y_smooth: &[T]) {
        for (&yi, &ys) in y.iter().zip(y_smooth.iter()) {
            let r = yi - ys;
            if self.n == 0 {
                self.origin_y = yi;
            }
            let y_offset = yi - self.origin_y;
            self.n += 1;
            let count = T::from(self.n).unwrap_or(T::one());
            let offset_delta = y_offset - self.mean_y_offset;
            self.mean_y_offset = self.mean_y_offset + offset_delta / count;
            let offset_delta_after_update = y_offset - self.mean_y_offset;
            self.sum_y_centered_sq =
                self.sum_y_centered_sq + offset_delta * offset_delta_after_update;
            let delta = yi - self.mean_y;
            self.mean_y = self.mean_y + delta / count;
            self.sum_y = self.sum_y + yi;
            self.sum_y_sq = self.sum_y_sq + yi * yi;
            self.sum_r = self.sum_r + r;
            self.sum_r_sq = self.sum_r_sq + r * r;
            self.sum_abs_r = self.sum_abs_r + r.abs();
        }
    }

    // Compute final diagnostics from the accumulated state.
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

        // Welford's centered sum avoids cancellation for data with a large offset.
        let ss_tot = self.sum_y_centered_sq.max(T::zero());
        let r_squared = if ss_tot > T::zero() {
            T::one() - self.sum_r_sq / ss_tot
        } else {
            let response_scale = self.origin_y.abs().max(T::one());
            let roundoff = T::from(16).unwrap_or(T::one()) * T::epsilon() * response_scale;
            let residual_rms = (self.sum_r_sq / n_t).sqrt();
            if residual_rms <= roundoff {
                T::one()
            } else {
                T::zero()
            }
        };

        // Streaming residual SD is the sample SD of emitted residuals.
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
    // Number of parameters in local linear regression (intercept + slope).
    const LINEAR_PARAMS: f64 = 2.0;

    // Minimum tuned-scale absolute epsilon to avoid division-by-zero.
    const MIN_TUNED_SCALE: f64 = 1e-12;

    // Constant to convert MAD to an unbiased estimate of sigma for normal data.
    // For normally distributed data, MAD × 1.4826 ≈ standard deviation.
    const MAD_TO_STD_FACTOR: f64 = 1.4826;

    // Compute diagnostic statistics from fit results.
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

    // Compute the residual sum of squares (RSS).
    // RSS = sum r_i^2.
    fn calculate_rss(residuals: &[T]) -> T {
        residuals.iter().fold(T::zero(), |acc, &r| acc + r * r)
    }

    // Compute the root mean squared error (RMSE).
    // RMSE = sqrt((1/n) * sum (y_i - y_hat_i)^2).
    pub fn calculate_rmse(y: &[T], y_smooth: &[T]) -> T {
        let n_t = T::from(y.len()).unwrap_or(T::one());
        let residual_scale = y
            .iter()
            .zip(y_smooth.iter())
            .fold(T::zero(), |scale, (&yi, &ys)| scale.max((yi - ys).abs()));
        if residual_scale == T::zero() {
            return T::zero();
        }

        let scaled_sum_squares =
            y.iter()
                .zip(y_smooth.iter())
                .fold(T::zero(), |sum, (&yi, &ys)| {
                    let scaled_residual = (yi - ys) / residual_scale;
                    sum + scaled_residual * scaled_residual
                });

        residual_scale * (scaled_sum_squares / n_t).sqrt()
    }

    // Compute the mean absolute error (MAE).
    // MAE = (1/n) * sum |y_i - y_hat_i|.
    pub fn calculate_mae(y: &[T], y_smooth: &[T]) -> T {
        let n_t = T::from(y.len()).unwrap_or(T::one());
        let residual_scale = y
            .iter()
            .zip(y_smooth.iter())
            .fold(T::zero(), |scale, (&yi, &ys)| scale.max((yi - ys).abs()));
        if residual_scale == T::zero() {
            return T::zero();
        }

        let scaled_sum_abs = y
            .iter()
            .zip(y_smooth.iter())
            .fold(T::zero(), |sum, (&yi, &ys)| {
                sum + (yi - ys).abs() / residual_scale
            });

        residual_scale * (scaled_sum_abs / n_t)
    }

    // Compute the coefficient of determination (R^2).
    // R^2 = 1 - SS_res / SS_tot, where SS_res is the residual
    // sum of squares and SS_tot is the total sum of squares.
    pub fn calculate_r_squared(y: &[T], y_smooth: &[T]) -> T {
        let n = y.len();
        if n == 0 {
            return T::zero();
        }
        if n == 1 {
            return T::one();
        }

        let origin = y[0];
        let mut mean_offset = T::zero();
        for (index, &yi) in y.iter().enumerate() {
            let offset = yi - origin;
            let count = T::from(index + 1).unwrap_or(T::one());
            mean_offset = mean_offset + (offset - mean_offset) / count;
        }

        // Scale both sums before squaring so large, finite values don't overflow.
        let (response_scale, residual_scale) = y.iter().zip(y_smooth.iter()).fold(
            (T::zero(), T::zero()),
            |(y_scale, r_scale), (&yi, &ys)| {
                let deviation = (yi - origin) - mean_offset;
                let residual = yi - ys;
                (y_scale.max(deviation.abs()), r_scale.max(residual.abs()))
            },
        );

        if response_scale == T::zero() {
            // All y values are identical
            if residual_scale == T::zero() {
                T::one() // Perfect fit
            } else {
                T::zero() // No variance to explain
            }
        } else {
            let (scaled_ss_tot, scaled_ss_res) = y.iter().zip(y_smooth.iter()).fold(
                (T::zero(), T::zero()),
                |(tot, res), (&yi, &ys)| {
                    let scaled_deviation = ((yi - origin) - mean_offset) / response_scale;
                    let scaled_residual = if residual_scale == T::zero() {
                        T::zero()
                    } else {
                        (yi - ys) / residual_scale
                    };
                    (
                        tot + scaled_deviation * scaled_deviation,
                        res + scaled_residual * scaled_residual,
                    )
                },
            );
            let scale_ratio = residual_scale / response_scale;
            let rss_ratio = scale_ratio * scale_ratio * (scaled_ss_res / scaled_ss_tot);
            T::one() - rss_ratio
        }
    }

    // Estimate the residual standard deviation using a robust method.
    // sigma_hat = 1.4826 * MAD(residuals).
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

    // Estimate the effective degrees of freedom from standard errors.
    // df_eff approx sum (SE_i / sigma_hat)^2.
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

    // Compute the effective degrees of freedom from leverage values.
    // df_eff = sum l_ii = tr(L).
    pub fn calculate_effective_df_from_leverages(leverage_values: &[T]) -> T {
        leverage_values
            .iter()
            .copied()
            .fold(T::zero(), |acc, v| acc + v)
    }

    // Compute the Akaike Information Criterion (AIC).
    // AIC = n * ln(RSS / n) + 2 * df_eff.
    pub fn calculate_aic(residuals: &[T], effective_df: T) -> T {
        let n = T::from(residuals.len()).unwrap_or(T::one());
        let rss = Self::calculate_rss(residuals);

        if rss <= T::zero() || n <= T::zero() {
            return T::infinity();
        }

        n * (rss / n).ln() + T::from(Self::LINEAR_PARAMS).unwrap() * effective_df
    }

    // Compute the corrected Akaike Information Criterion (AICc).
    // AICc = AIC + (2 * df_eff * (df_eff + 1)) / (n - df_eff - 1).
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

impl<T: Float + Display> Display for Diagnostics<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "LOWESS Diagnostics:")?;
        writeln!(f, "  RMSE:         {:.6}", self.rmse)?;
        writeln!(f, "  MAE:          {:.6}", self.mae)?;
        writeln!(f, "  R2:           {:.6}", self.r_squared)?;
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
