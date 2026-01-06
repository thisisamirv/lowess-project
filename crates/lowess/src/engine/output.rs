//! Output types and result structures for LOWESS operations.
//!
//! ## Purpose
//!
//! This module defines the `LowessResult` struct which encapsulates all
//! outputs from a LOWESS smoothing operation, including smoothed values,
//! diagnostics, and confidence/prediction intervals.
//!
//! ## Design notes
//!
//! * **Memory Efficiency**: All optional outputs use `Option<Vec<T>>`.
//! * **Generics**: Results are generic over `Float` types.
//! * **Ergonomics**: Implements `Display` for human-readable output.
//! * **Consistency**: Sorted x-values are stored to maintain correspondence.
//!
//! ## Key concepts
//!
//! * **Optional Outputs**: Results are only populated when specific features are enabled.
//! * **Intervals**: Confidence (mean curve) and Prediction (new observations).
//! * **Metadata**: Tracks iterations, fraction used, and CV scores.
//!
//! ## Invariants
//!
//! * All populated vectors have the same length as the input data.
//! * x-values are sorted in monotonically increasing order.
//! * Lower bounds are always less than or equal to upper bounds for all intervals.
//! * Robustness weights are always in the range [0, 1].
//!
//! ## Non-goals
//!
//! * This module does not perform calculations; it only stores results.
//! * This module does not validate result consistency (responsibility of the engine).
//! * This module does not provide serialization/deserialization logic.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::cmp::Ordering;
use core::fmt::{Debug, Display, Formatter, Result};
use num_traits::Float;

// Internal dependencies
use crate::evaluation::diagnostics::Diagnostics;

// ============================================================================
// Result Structure
// ============================================================================

/// Comprehensive LOWESS output containing smoothed values and diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct LowessResult<T> {
    /// Sorted x-values (independent variable).
    pub x: Vec<T>,

    /// Smoothed y-values (dependent variable).
    pub y: Vec<T>,

    /// Standard errors of the fit at each point.
    pub standard_errors: Option<Vec<T>>,

    /// Lower bounds of the confidence intervals for the mean response.
    pub confidence_lower: Option<Vec<T>>,

    /// Upper bounds of the confidence intervals for the mean response.
    pub confidence_upper: Option<Vec<T>>,

    /// Lower bounds of the prediction intervals for new observations.
    pub prediction_lower: Option<Vec<T>>,

    /// Upper bounds of the prediction intervals for new observations.
    pub prediction_upper: Option<Vec<T>>,

    /// Residuals from the fit (y_i - y_hat_i).
    pub residuals: Option<Vec<T>>,

    /// Final robustness weights from the iterative refinement process.
    pub robustness_weights: Option<Vec<T>>,

    /// Comprehensive diagnostic metrics (RMSE, R^2, AIC, etc.).
    pub diagnostics: Option<Diagnostics<T>>,

    /// Number of robustness iterations actually performed.
    pub iterations_used: Option<usize>,

    /// Smoothing fraction used for the fit (optimal if selected by CV).
    pub fraction_used: T,

    /// RMSE scores for each tested fraction during cross-validation.
    pub cv_scores: Option<Vec<T>>,
}

impl<T: Float> LowessResult<T> {
    // ========================================================================
    // Query Methods
    // ========================================================================

    /// Check if confidence intervals were computed.
    pub fn has_confidence_intervals(&self) -> bool {
        self.confidence_lower.is_some() && self.confidence_upper.is_some()
    }

    /// Check if prediction intervals were computed.
    pub fn has_prediction_intervals(&self) -> bool {
        self.prediction_lower.is_some() && self.prediction_upper.is_some()
    }

    /// Check if cross-validation was performed.
    pub fn has_cv_scores(&self) -> bool {
        self.cv_scores.is_some()
    }

    /// Get the best (minimum) CV score.
    pub fn best_cv_score(&self) -> Option<T> {
        self.cv_scores.as_ref().and_then(|scores| {
            scores
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        })
    }
}

// ============================================================================
// Display Implementation
// ============================================================================

impl<T: Float + Display + Debug> Display for LowessResult<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "Summary:")?;
        writeln!(f, "  Data points: {}", self.x.len())?;
        writeln!(f, "  Fraction:    {}", self.fraction_used)?;

        if let Some(iters) = self.iterations_used {
            writeln!(f, "  Iterations: {}", iters)?;
        }

        // Show robustness status
        if self.robustness_weights.is_some() {
            writeln!(f, "  Robustness: Applied")?;
        }

        if self.has_cv_scores() {
            if let Some(best_score) = self.best_cv_score() {
                writeln!(f, "  Best CV score: {}", best_score)?;
            }
        }
        writeln!(f)?;

        if let Some(diag) = &self.diagnostics {
            writeln!(f, "{}", diag)?;
        }

        writeln!(f, "Smoothed Data:")?;

        // Determine which columns to show
        let has_std_err = self.standard_errors.is_some();
        let has_conf = self.has_confidence_intervals();
        let has_pred = self.has_prediction_intervals();
        let has_resid = self.residuals.is_some();
        let has_weights = self.robustness_weights.is_some();

        // Build header
        write!(f, "{:>8} {:>12}", "X", "Y_smooth")?;
        if has_std_err {
            write!(f, " {:>12}", "Std_Err")?;
        }
        if has_conf {
            write!(f, " {:>12} {:>12}", "Conf_Lower", "Conf_Upper")?;
        }
        if has_pred {
            write!(f, " {:>12} {:>12}", "Pred_Lower", "Pred_Upper")?;
        }
        if has_resid {
            write!(f, " {:>12}", "Residual")?;
        }
        if has_weights {
            write!(f, " {:>10}", "Rob_Weight")?;
        }
        writeln!(f)?;

        // Separator line
        let line_width = 21
            + if has_std_err { 13 } else { 0 }
            + if has_conf { 26 } else { 0 }
            + if has_pred { 26 } else { 0 }
            + if has_resid { 13 } else { 0 }
            + if has_weights { 11 } else { 0 };
        writeln!(f, "{:-<width$}", "", width = line_width)?;

        // Data rows (show first 10 and last 10 if more than 20 points)
        let n = self.x.len();
        let show_all = n <= 20;
        let rows_to_show: Vec<usize> = if show_all {
            (0..n).collect()
        } else {
            (0..10).chain(n - 10..n).collect()
        };

        let mut prev_idx = 0;
        for (i, &idx) in rows_to_show.iter().enumerate() {
            // Add ellipsis if we skipped rows
            if i > 0 && idx != prev_idx + 1 {
                writeln!(f, "{:>8}", "...")?;
            }
            prev_idx = idx;

            write!(f, "{:>8.2} {:>12.6}", self.x[idx], self.y[idx])?;

            // Standard error
            if has_std_err {
                if let Some(se) = &self.standard_errors {
                    write!(f, " {:>12.6}", se[idx])?;
                }
            }

            // Confidence intervals
            if has_conf {
                if let (Some(lower), Some(upper)) = (&self.confidence_lower, &self.confidence_upper)
                {
                    write!(f, " {:>12.6} {:>12.6}", lower[idx], upper[idx])?;
                }
            }

            // Prediction intervals
            if has_pred {
                if let (Some(lower), Some(upper)) = (&self.prediction_lower, &self.prediction_upper)
                {
                    write!(f, " {:>12.6} {:>12.6}", lower[idx], upper[idx])?;
                }
            }

            // Residuals
            if has_resid {
                if let Some(resid) = &self.residuals {
                    write!(f, " {:>12.6}", resid[idx])?;
                }
            }

            // Robustness weights
            if has_weights {
                if let Some(weights) = &self.robustness_weights {
                    write!(f, " {:>10.4}", weights[idx])?;
                }
            }

            writeln!(f)?;
        }

        Ok(())
    }
}
