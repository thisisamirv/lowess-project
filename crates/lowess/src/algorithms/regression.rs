//! Regression Logic
//!
//! This module provides the core data types and logic for local regression fitting (LOWESS),
//! including:
//! - Context for managing regression state.
//! - Weighted least squares (WLS) fitting matching R's `lowess()` arithmetic order.
//! - Data structures for calibration and fitting results.
// ## srrstats Compliance
//
// Input validation: x/y slices must have matching lengths; empty inputs return defaults.
// Implements weighted least squares (WLS) via `fit_wls` for local regression.
// @srrstats {RE2.0} Core LOWESS algorithm: local linear fits with distance-based kernel weighting.
// @srrstats {RE2.1} Predictions at fitted x-values via `LinearFit::predict`.
// @srrstats {G2.0} Slice lengths validated; mismatched lengths would panic at runtime.
// @srrstats {G2.1} Edge cases documented: zero-length input returns zero-initialized result.
// @srrstats {G2.4} Numeric tolerance (1e-12) used for near-zero variance detection.

// External dependencies
#[cfg(not(feature = "std"))]
use alloc::string::ToString;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::fmt::Debug;
use core::str::FromStr;
use num_traits::Float;

// Internal dependencies
use crate::math::kernel::WeightFunction;
use crate::primitives::errors::LowessError;
use crate::primitives::window::Window;

// Policy for handling cases where all weights are zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZeroWeightFallback {
    // Use local mean (default).
    #[default]
    UseLocalMean,

    // Return the original y-value.
    ReturnOriginal,

    // Return None (propagate failure).
    ReturnNone,
}

impl FromStr for ZeroWeightFallback {
    type Err = LowessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "use_local_mean" | "local_mean" | "mean" => Ok(Self::UseLocalMean),
            "return_original" | "original" => Ok(Self::ReturnOriginal),
            "return_none" | "none" | "nan" => Ok(Self::ReturnNone),
            _ => Err(LowessError::InvalidOption {
                option: "zero_weight_fallback",
                value: s.to_string(),
                valid: "use_local_mean, return_original, return_none",
            }),
        }
    }
}

impl ZeroWeightFallback {
    // Create from u8 flag for backward compatibility.
    #[inline]
    pub fn from_u8(flag: u8) -> Self {
        match flag {
            0 => ZeroWeightFallback::UseLocalMean,
            1 => ZeroWeightFallback::ReturnOriginal,
            2 => ZeroWeightFallback::ReturnNone,
            _ => ZeroWeightFallback::UseLocalMean,
        }
    }

    // Convert to u8 flag for backward compatibility.
    #[inline]
    pub fn to_u8(self) -> u8 {
        match self {
            ZeroWeightFallback::UseLocalMean => 0,
            ZeroWeightFallback::ReturnOriginal => 1,
            ZeroWeightFallback::ReturnNone => 2,
        }
    }
}

// Parameters for weight computation.
pub struct WeightParams<T: Float> {
    // Current x-value being fitted
    pub x_current: T,

    // Window radius - defines the scale of the local fit
    pub window_radius: T,

    // Near-threshold: points closer than this get weight 1.0.
    pub h1: T,

    // Far-threshold: points farther than this get weight 0.0.
    pub h9: T,
}

impl<T: Float> WeightParams<T> {
    // Construct WeightParams with validated window radius.
    pub fn new(x_current: T, window_radius: T, _use_robustness: bool) -> Self {
        debug_assert!(
            window_radius > T::zero(),
            "WeightParams::new: window_radius must be positive"
        );

        let radius = if window_radius > T::zero() {
            window_radius
        } else {
            T::from(1e-12).unwrap_or_else(T::epsilon)
        };

        let h1 = T::from(0.001).unwrap_or_else(T::epsilon) * radius;
        let h9 = T::from(0.999).unwrap_or_else(|| T::one() - T::epsilon()) * radius;

        Self {
            x_current,
            window_radius: radius,
            h1,
            h9,
        }
    }
}

pub trait WLSSolver: Float {}

impl WLSSolver for f64 {}

impl WLSSolver for f32 {}

// Linear regression fit result (slope and intercept).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearFit<T: Float> {
    // Slope (beta_1)
    pub slope: T,

    // Intercept (beta_0)
    pub intercept: T,

    // Weighted mean of x-values
    pub x_mean: T,

    // Weighted mean of y-values
    pub y_mean: T,

    // Fitted value at the pivot used for the local WLS solve.
    pub fitted: T,
}

impl<T: Float> LinearFit<T> {
    // Create a zero-initialized fit.
    pub fn zero() -> Self {
        Self {
            slope: T::zero(),
            intercept: T::zero(),
            x_mean: T::zero(),
            y_mean: T::zero(),
            fitted: T::zero(),
        }
    }

    // Predict y-value for a given x using the model.
    #[inline]
    pub fn predict(&self, x: T) -> T {
        self.intercept + self.slope * x
    }

    // Fit Ordinary Least Squares (OLS) regression.
    pub fn fit_ols(x: &[T], y: &[T]) -> Self {
        let n = x.len();
        if n == 0 {
            return Self::zero();
        }

        let n_t = T::from(n).unwrap_or(T::one());

        let mut sum_x = T::zero();
        let mut sum_y = T::zero();
        let mut sum_x_sq = T::zero();

        for i in 0..n {
            sum_x = sum_x + x[i];
            sum_y = sum_y + y[i];
            sum_x_sq = sum_x_sq + x[i] * x[i];
        }

        let x_mean = sum_x / n_t;
        let y_mean = sum_y / n_t;

        let mut variance = T::zero();
        let mut covariance = T::zero();

        for i in 0..n {
            let dx = x[i] - x_mean;
            let dy = y[i] - y_mean;
            variance = variance + dx * dx;
            covariance = covariance + dx * dy;
        }

        // Relative degeneracy tolerance (see `fit_wls`): `variance` is bounded by
        // `sum(x_i^2)`, so scaling by it tolerates any x magnitude instead of
        // silently treating small-magnitude x-values as a degenerate design.
        let tol = T::epsilon() * sum_x_sq;
        if variance <= tol {
            return Self {
                slope: T::zero(),
                intercept: y_mean,
                x_mean,
                y_mean,
                fitted: y_mean,
            };
        }

        let slope = covariance / variance;
        let intercept = y_mean - slope * x_mean;

        Self {
            slope,
            intercept,
            x_mean,
            y_mean,
            fitted: y_mean,
        }
    }

    // Compute the OLS standard error of the fitted value at each observed `x`.
    //
    // For simple linear regression the standard error of the mean response at
    // `x0` is `sigma_hat * sqrt(1/n + (x0 - x_mean)^2 / Sxx)`, where
    // `sigma_hat^2 = SSE / (n - 2)` and `Sxx = sum((x_i - x_mean)^2)`. This
    // matches `stats::lm`'s `se.fit` for the same model.
    pub fn ols_std_errors(&self, x: &[T], y: &[T]) -> Vec<T> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }

        let n_t = T::from(n).unwrap_or(T::one());

        // Residual sum of squares and corrected sum of squares of x.
        let mut sse = T::zero();
        let mut sxx = T::zero();
        let mut sum_x_sq = T::zero();
        for i in 0..n {
            let r = y[i] - self.predict(x[i]);
            sse = sse + r * r;
            let dx = x[i] - self.x_mean;
            sxx = sxx + dx * dx;
            sum_x_sq = sum_x_sq + x[i] * x[i];
        }

        // Relative degeneracy tolerance (see `fit_wls`).
        let tol = T::epsilon() * sum_x_sq;
        let inv_n = T::one() / n_t;

        if sxx <= tol {
            // Degenerate design (all x equal): the fit is a constant with a
            // single parameter, so the standard error is sigma_hat * sqrt(1/n).
            let df = n_t - T::one();
            if df <= T::zero() {
                return vec![T::zero(); n];
            }
            let se = (sse / df).sqrt() * inv_n.sqrt();
            return vec![se; n];
        }

        // Two-parameter linear model: sigma_hat^2 = SSE / (n - 2).
        let df = n_t - T::from(2).unwrap();
        if df <= T::zero() {
            return vec![T::zero(); n];
        }
        let sigma_hat = (sse / df).sqrt();

        x.iter()
            .map(|&xi| {
                let dx = xi - self.x_mean;
                let leverage = inv_n + (dx * dx) / sxx;
                sigma_hat * leverage.sqrt()
            })
            .collect()
    }
}

impl<T: Float + WLSSolver> LinearFit<T> {
    // Fit weighted least squares using Cleveland/R `lowest()` arithmetic order.
    pub fn fit_wls(x: &[T], y: &[T], weights: &mut [T], x_current: T, global_x_range: T) -> Self {
        let n = x.len();
        if n == 0 {
            return Self::zero();
        }

        let sum_w = weights.iter().copied().fold(T::zero(), |sum, w| sum + w);
        if sum_w <= T::zero() {
            return Self::zero();
        }

        for weight in weights.iter_mut() {
            *weight = *weight / sum_w;
        }

        let mut x_mean = T::zero();
        for i in 0..n {
            x_mean = x_mean + weights[i] * x[i];
        }

        let mut spread = T::zero();
        for i in 0..n {
            let dx = x[i] - x_mean;
            spread = spread + weights[i] * (dx * dx);
        }

        let min_spread = T::from(0.001).unwrap_or_else(T::epsilon) * global_x_range;
        let use_linear = spread.sqrt() > min_spread;

        let target_adjustment = if use_linear {
            (x_current - x_mean) / spread
        } else {
            T::zero()
        };
        let mut y_mean = T::zero();
        let mut covariance = T::zero();
        for i in 0..n {
            let dx = x[i] - x_mean;
            y_mean = y_mean + weights[i] * y[i];
            covariance = covariance + weights[i] * dx * y[i];
        }
        for i in 0..n {
            let dx = x[i] - x_mean;
            weights[i] = weights[i] * (T::one() + target_adjustment * dx);
        }
        let mut fitted = T::zero();
        for i in 0..n {
            fitted = fitted + weights[i] * y[i];
        }
        let slope = if use_linear {
            covariance / spread
        } else {
            T::zero()
        };

        Self {
            slope,
            intercept: fitted - slope * x_current,
            x_mean,
            y_mean,
            fitted,
        }
    }
}

// Context containing all data needed to fit a single point.
pub struct RegressionContext<'a, T: Float> {
    // Slice of x-values (independent variable)
    pub x: &'a [T],

    // Slice of y-values (dependent variable)
    pub y: &'a [T],

    // Index of the point to fit
    pub idx: usize,

    // Window for the local fit (defines neighborhood)
    pub window: Window,

    // Whether to use robustness weights
    pub use_robustness: bool,

    // Slice of robustness weights (all 1.0 if not using robustness)
    pub robustness_weights: &'a [T],

    // Mutable slice of weights to be used in fitting
    pub weights: &'a mut [T],

    // Weight function (kernel)
    pub weight_function: WeightFunction,

    // Zero-weight fallback policy
    pub zero_weight_fallback: ZeroWeightFallback,

    // Optional per-observation custom weights applied as `w_ij = custom_weights[j] * K(d_ij / h)`.
    // When provided, multiplies each kernel weight by the corresponding observation weight.
    pub custom_weights: Option<&'a [T]>,
}

impl<'a, T: Float + WLSSolver> RegressionContext<'a, T> {
    #[allow(dead_code)]
    // Perform the local linear fit using the context configuration.
    pub fn fit(&mut self) -> Option<T> {
        self.fit_with_derivative().map(|(y, _slope)| y)
    }

    // Same as `fit()`, but also returns the local fit's slope (derivative) at the fitted
    // point. `eval_at` already computes this internally, so this is effectively free -
    // `fit()` is just a thin wrapper that discards it.
    pub fn fit_with_derivative(&mut self) -> Option<(T, T)> {
        let n = self.x.len();

        if self.idx >= n || self.window.left >= n || self.window.right >= n {
            return None;
        }

        let x_current = self.x[self.idx];
        let orig_y = self.y[self.idx];
        self.eval_at(x_current, Some(orig_y))
    }

    // Evaluate the local WLS fit at an arbitrary out-of-sample query point, reusing the
    // same weighting/model logic as `fit()`. Unlike `fit()`, there is no training
    // observation at `x_query`, so `ZeroWeightFallback::ReturnOriginal` falls back to
    // `UseLocalMean` instead (see `eval_at`). Returns `(y, slope)`, where `slope` is the
    // local fit's derivative at `x_query` (0 for degenerate/fallback branches).
    pub fn predict_at(&mut self, x_query: T) -> Option<(T, T)> {
        let n = self.x.len();

        if self.window.left >= n || self.window.right >= n {
            return None;
        }

        self.eval_at(x_query, None)
    }

    // Shared local-WLS evaluation logic for `fit()` (in-sample, `orig_y = Some(y[idx])`)
    // and `predict_at()` (out-of-sample, `orig_y = None`).
    // Shared local-WLS evaluation logic for `fit()` (in-sample, `orig_y = Some(y[idx])`)
    // and `predict_at()` (out-of-sample, `orig_y = None`).
    fn eval_at(&mut self, x_pivot: T, orig_y: Option<T>) -> Option<(T, T)> {
        let n = self.x.len();
        let window_radius = self.window.max_distance(self.x, x_pivot);
        let weight_left = if self.weight_function.support().is_none() {
            0
        } else {
            self.window.left
        };

        if window_radius <= T::zero() {
            // R's `lowest` scans from the window's left edge to the end of the data and
            // stops at the first x past the pivot, so a zero-radius window takes in every
            // point tied with the pivot rather than only the nearest `q`.
            let mut tied_end = self.window.right;
            while tied_end + 1 < n && self.x[tied_end + 1] == x_pivot {
                tied_end += 1;
            }
            let tied_count = T::from(tied_end - weight_left + 1).unwrap_or(T::one());

            let mut sum_w = T::zero();
            let mut j = weight_left;
            while j <= tied_end {
                let w_base = if self.use_robustness {
                    self.robustness_weights[j]
                } else {
                    T::one()
                };
                let w = if let Some(cw) = self.custom_weights {
                    w_base * cw[j]
                } else {
                    w_base
                };
                self.weights[j] = w;
                sum_w = sum_w + w;
                j += 1;
            }

            if sum_w > T::zero() {
                // `lowest` divides each window weight by their sum before accumulating the
                // fitted value. Dividing the weighted sum at the end instead is equivalent
                // in exact arithmetic but rounds differently, and the robustness iteration
                // amplifies that difference geometrically.
                let mut fitted = T::zero();
                let mut j = weight_left;
                while j <= tied_end {
                    fitted = fitted + (self.weights[j] / sum_w) * self.y[j];
                    j += 1;
                }
                return Some((fitted, T::zero()));
            } else {
                return match self.zero_weight_fallback {
                    ZeroWeightFallback::UseLocalMean => {
                        let mean = self.y[weight_left..=tied_end]
                            .iter()
                            .copied()
                            .fold(T::zero(), |acc, v| acc + v)
                            / tied_count;
                        Some((mean, T::zero()))
                    }
                    // No original observation exists for an out-of-sample query point.
                    ZeroWeightFallback::ReturnOriginal => match orig_y {
                        Some(v) => Some((v, T::zero())),
                        None => {
                            let mean = self.y[weight_left..=tied_end]
                                .iter()
                                .copied()
                                .fold(T::zero(), |acc, v| acc + v)
                                / tied_count;
                            Some((mean, T::zero()))
                        }
                    },
                    ZeroWeightFallback::ReturnNone => None,
                };
            }
        }

        let weight_params = WeightParams::new(x_pivot, window_radius, self.use_robustness);

        let (mut weight_sum, rightmost_idx) = self.weight_function.compute_window_weights(
            self.x,
            weight_left,
            n - 1,
            weight_params.x_current,
            weight_params.window_radius,
            weight_params.h1,
            weight_params.h9,
            self.weights,
        );

        if self.use_robustness || self.custom_weights.is_some() {
            weight_sum = T::zero();
            let mut j = weight_left;
            while j <= rightmost_idx {
                let w_k = self.weights[j];
                if w_k > T::zero() {
                    let w_robust = if self.use_robustness {
                        self.robustness_weights[j]
                    } else {
                        T::one()
                    };
                    let w_custom = if let Some(cw) = self.custom_weights {
                        cw[j]
                    } else {
                        T::one()
                    };
                    let w_final = w_k * w_robust * w_custom;
                    self.weights[j] = w_final;
                    weight_sum = weight_sum + w_final;
                }
                j += 1;
            }
        }

        if weight_sum <= T::zero() {
            match self.zero_weight_fallback {
                ZeroWeightFallback::UseLocalMean => {
                    let window_size = self.window.len();
                    let cnt = T::from(window_size).unwrap_or(T::one());
                    let mean = self.y[self.window.left..=self.window.right]
                        .iter()
                        .copied()
                        .fold(T::zero(), |acc, v| acc + v)
                        / cnt;
                    return Some((mean, T::zero()));
                }
                // No original observation exists for an out-of-sample query point.
                ZeroWeightFallback::ReturnOriginal => {
                    return match orig_y {
                        Some(v) => Some((v, T::zero())),
                        None => {
                            let window_size = self.window.len();
                            let cnt = T::from(window_size).unwrap_or(T::one());
                            let mean = self.y[self.window.left..=self.window.right]
                                .iter()
                                .copied()
                                .fold(T::zero(), |acc, v| acc + v)
                                / cnt;
                            Some((mean, T::zero()))
                        }
                    };
                }
                ZeroWeightFallback::ReturnNone => return None,
            }
        }

        let window_x = &self.x[weight_left..=rightmost_idx];
        let window_y = &self.y[weight_left..=rightmost_idx];
        let window_weights = &mut self.weights[weight_left..=rightmost_idx];

        let global_x_range = self.x[self.x.len() - 1] - self.x[0];
        let model = LinearFit::fit_wls(window_x, window_y, window_weights, x_pivot, global_x_range);
        Some((model.fitted, model.slope))
    }
}
