//! Regression Logic
//!
//! ## Purpose
//!
//! This module provides the core data types and logic for local regression fitting (LOWESS),
//! including:
//! - Context for managing regression state.
//! - Generic and SIMD-optimized solvers for weighted least squares (WLS).
//! - Data structures for calibration and fitting results.

// External dependencies
use core::fmt::Debug;
use num_traits::Float;
use wide::{f32x8, f64x2};

// Internal dependencies
use crate::math::kernel::WeightFunction;
use crate::primitives::window::Window;

// ============================================================================
// Zero-Weight Fallback Policy
// ============================================================================

/// Policy for handling cases where all weights are zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZeroWeightFallback {
    /// Use local mean (default).
    #[default]
    UseLocalMean,

    /// Return the original y-value.
    ReturnOriginal,

    /// Return None (propagate failure).
    ReturnNone,
}

impl ZeroWeightFallback {
    /// Create from u8 flag for backward compatibility.
    #[inline]
    pub fn from_u8(flag: u8) -> Self {
        match flag {
            0 => ZeroWeightFallback::UseLocalMean,
            1 => ZeroWeightFallback::ReturnOriginal,
            2 => ZeroWeightFallback::ReturnNone,
            _ => ZeroWeightFallback::UseLocalMean,
        }
    }

    /// Convert to u8 flag for backward compatibility.
    #[inline]
    pub fn to_u8(self) -> u8 {
        match self {
            ZeroWeightFallback::UseLocalMean => 0,
            ZeroWeightFallback::ReturnOriginal => 1,
            ZeroWeightFallback::ReturnNone => 2,
        }
    }
}

// ============================================================================
// Weight Parameters
// ============================================================================

/// Parameters for weight computation.
pub struct WeightParams<T: Float> {
    /// Current x-value being fitted
    pub x_current: T,

    /// Window radius - defines the scale of the local fit
    pub window_radius: T,

    /// Near-threshold: points closer than this get weight 1.0.
    pub h1: T,

    /// Far-threshold: points farther than this get weight 0.0.
    pub h9: T,
}

impl<T: Float> WeightParams<T> {
    /// Construct WeightParams with validated window radius.
    pub fn new(x_current: T, window_radius: T, _use_robustness: bool) -> Self {
        debug_assert!(
            window_radius > T::zero(),
            "WeightParams::new: window_radius must be positive"
        );

        let radius = if window_radius > T::zero() {
            window_radius
        } else {
            T::from(1e-12).unwrap()
        };

        let h1 = T::from(0.001).unwrap() * radius;
        let h9 = T::from(0.999).unwrap() * radius;

        Self {
            x_current,
            window_radius: radius,
            h1,
            h9,
        }
    }
}

// ============================================================================
// Generic Accumulation and Solving
// ============================================================================

/// Scalar accumulation for 1D weighted least squares (generic Float).
#[inline]
pub fn accumulate_wls_scalar<T: Float>(x: &[T], y: &[T], weights: &[T]) -> (T, T, T, T, T) {
    let n = x.len();
    if n == 0 {
        return (T::zero(), T::zero(), T::zero(), T::zero(), T::zero());
    }

    let mut sum_w = T::zero();
    let mut sum_wx = T::zero();
    let mut sum_wy = T::zero();
    let mut sum_wxx = T::zero();
    let mut sum_wxy = T::zero();

    for i in 0..n {
        let w = weights[i];
        let x_val = x[i];
        let y_val = y[i];

        let wx = w * x_val;

        sum_w = sum_w + w;
        sum_wx = sum_wx + wx;
        sum_wy = sum_wy + w * y_val;
        sum_wxx = sum_wxx + wx * x_val;
        sum_wxy = sum_wxy + wx * y_val;
    }

    (sum_w, sum_wx, sum_wy, sum_wxx, sum_wxy)
}

/// Generic linear system solver for 1D weighted least squares.
#[inline]
pub fn solve_wls_scalar<T: Float>(
    sum_w: T,
    sum_wx: T,
    sum_wy: T,
    sum_wxx: T,
    sum_wxy: T,
    tol: T,
) -> Option<(T, T, T, T)> {
    if sum_w <= T::zero() {
        return None;
    }

    let x_mean = sum_wx / sum_w;
    let y_mean = sum_wy / sum_w;
    let variance = sum_wxx - (sum_wx * sum_wx) / sum_w;

    if variance <= tol {
        return Some((T::zero(), y_mean, x_mean, y_mean));
    }

    let covariance = sum_wxy - (sum_wx * sum_wy) / sum_w;
    let slope = covariance / variance;
    let intercept = y_mean - slope * x_mean;

    Some((slope, intercept, x_mean, y_mean))
}

// ============================================================================
// Specialized Accumulation (SIMD)
// ============================================================================

/// SIMD-optimized accumulation for 1D weighted least squares (f64).
#[inline]
pub fn accumulate_wls_simd_f64(x: &[f64], y: &[f64], weights: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = x.len();
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let mut i = 0;
    let mut s_w = f64x2::splat(0.0);
    let mut s_wx = f64x2::splat(0.0);
    let mut s_wy = f64x2::splat(0.0);
    let mut s_wxx = f64x2::splat(0.0);
    let mut s_wxy = f64x2::splat(0.0);

    unsafe {
        while i + 2 <= n {
            let w = f64x2::new([*weights.get_unchecked(i), *weights.get_unchecked(i + 1)]);
            let x_val = f64x2::new([*x.get_unchecked(i), *x.get_unchecked(i + 1)]);
            let y_val = f64x2::new([*y.get_unchecked(i), *y.get_unchecked(i + 1)]);

            let wx = w * x_val;
            let wy = w * y_val;

            s_w += w;
            s_wx += wx;
            s_wy += wy;
            s_wxx += wx * x_val;
            s_wxy += wx * y_val;

            i += 2;
        }
    }

    let mut a_w = s_w.reduce_add();
    let mut a_wx = s_wx.reduce_add();
    let mut a_wy = s_wy.reduce_add();
    let mut a_wxx = s_wxx.reduce_add();
    let mut a_wxy = s_wxy.reduce_add();

    unsafe {
        while i < n {
            let w = *weights.get_unchecked(i);
            let x_val = *x.get_unchecked(i);
            let y_val = *y.get_unchecked(i);

            let wx = w * x_val;

            a_w += w;
            a_wx += wx;
            a_wy += w * y_val;
            a_wxx += wx * x_val;
            a_wxy += wx * y_val;

            i += 1;
        }
    }

    (a_w, a_wx, a_wy, a_wxx, a_wxy)
}

/// SIMD-optimized accumulation for 1D weighted least squares (f32).
#[inline]
pub fn accumulate_wls_simd_f32(x: &[f32], y: &[f32], weights: &[f32]) -> (f32, f32, f32, f32, f32) {
    let n = x.len();
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let mut i = 0;
    let mut s_w = f32x8::splat(0.0);
    let mut s_wx = f32x8::splat(0.0);
    let mut s_wy = f32x8::splat(0.0);
    let mut s_wxx = f32x8::splat(0.0);
    let mut s_wxy = f32x8::splat(0.0);

    unsafe {
        while i + 8 <= n {
            let w = f32x8::new([
                *weights.get_unchecked(i),
                *weights.get_unchecked(i + 1),
                *weights.get_unchecked(i + 2),
                *weights.get_unchecked(i + 3),
                *weights.get_unchecked(i + 4),
                *weights.get_unchecked(i + 5),
                *weights.get_unchecked(i + 6),
                *weights.get_unchecked(i + 7),
            ]);
            let x_val = f32x8::new([
                *x.get_unchecked(i),
                *x.get_unchecked(i + 1),
                *x.get_unchecked(i + 2),
                *x.get_unchecked(i + 3),
                *x.get_unchecked(i + 4),
                *x.get_unchecked(i + 5),
                *x.get_unchecked(i + 6),
                *x.get_unchecked(i + 7),
            ]);
            let y_val = f32x8::new([
                *y.get_unchecked(i),
                *y.get_unchecked(i + 1),
                *y.get_unchecked(i + 2),
                *y.get_unchecked(i + 3),
                *y.get_unchecked(i + 4),
                *y.get_unchecked(i + 5),
                *y.get_unchecked(i + 6),
                *y.get_unchecked(i + 7),
            ]);

            let wx = w * x_val;
            let wy = w * y_val;

            s_w += w;
            s_wx += wx;
            s_wy += wy;
            s_wxx += wx * x_val;
            s_wxy += wx * y_val;

            i += 8;
        }
    }

    let mut a_w = s_w.reduce_add();
    let mut a_wx = s_wx.reduce_add();
    let mut a_wy = s_wy.reduce_add();
    let mut a_wxx = s_wxx.reduce_add();
    let mut a_wxy = s_wxy.reduce_add();

    unsafe {
        while i < n {
            let w = *weights.get_unchecked(i);
            let x_val = *x.get_unchecked(i);
            let y_val = *y.get_unchecked(i);

            let wx = w * x_val;

            a_w += w;
            a_wx += wx;
            a_wy += w * y_val;
            a_wxx += wx * x_val;
            a_wxy += wx * y_val;

            i += 1;
        }
    }

    (a_w, a_wx, a_wy, a_wxx, a_wxy)
}

// ============================================================================
// Solver Trait
// ============================================================================

/// Trait for type-specific weighted least squares accumulation and solving.
pub trait WLSSolver: Float {
    /// Accumulate weighted statistics.
    #[inline]
    fn accumulate_wls(x: &[Self], y: &[Self], weights: &[Self]) -> (Self, Self, Self, Self, Self) {
        accumulate_wls_scalar(x, y, weights)
    }

    /// Solve for coefficients.
    #[inline]
    fn solve_wls(
        sum_w: Self,
        sum_wx: Self,
        sum_wy: Self,
        sum_wxx: Self,
        sum_wxy: Self,
        tol: Self,
    ) -> Option<(Self, Self, Self, Self)> {
        solve_wls_scalar(sum_w, sum_wx, sum_wy, sum_wxx, sum_wxy, tol)
    }
}

impl WLSSolver for f64 {
    #[inline]
    fn accumulate_wls(x: &[f64], y: &[f64], weights: &[f64]) -> (f64, f64, f64, f64, f64) {
        accumulate_wls_simd_f64(x, y, weights)
    }
}

impl WLSSolver for f32 {
    #[inline]
    fn accumulate_wls(x: &[f32], y: &[f32], weights: &[f32]) -> (f32, f32, f32, f32, f32) {
        accumulate_wls_simd_f32(x, y, weights)
    }
}

// ============================================================================
// LinearFit
// ============================================================================

/// Linear regression fit result (slope and intercept).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearFit<T: Float> {
    /// Slope (beta_1)
    pub slope: T,

    /// Intercept (beta_0)
    pub intercept: T,

    /// Weighted mean of x-values
    pub x_mean: T,

    /// Weighted mean of y-values
    pub y_mean: T,
}

impl<T: Float> LinearFit<T> {
    /// Create a zero-initialized fit.
    pub fn zero() -> Self {
        Self {
            slope: T::zero(),
            intercept: T::zero(),
            x_mean: T::zero(),
            y_mean: T::zero(),
        }
    }

    /// Predict y-value for a given x using the model.
    #[inline]
    pub fn predict(&self, x: T) -> T {
        self.intercept + self.slope * x
    }

    /// Fit Ordinary Least Squares (OLS) regression.
    pub fn fit_ols(x: &[T], y: &[T]) -> Self {
        let n = x.len();
        if n == 0 {
            return Self::zero();
        }

        let n_t = T::from(n).unwrap_or(T::one());

        let mut sum_x = T::zero();
        let mut sum_y = T::zero();

        for i in 0..n {
            sum_x = sum_x + x[i];
            sum_y = sum_y + y[i];
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

        let tol = T::from(1e-12).unwrap();
        if variance <= tol {
            return Self {
                slope: T::zero(),
                intercept: y_mean,
                x_mean,
                y_mean,
            };
        }

        let slope = covariance / variance;
        let intercept = y_mean - slope * x_mean;

        Self {
            slope,
            intercept,
            x_mean,
            y_mean,
        }
    }
}

impl<T: Float + WLSSolver> LinearFit<T> {
    /// Fit Weighted Least Squares (WLS) regression using SIMD-optimized accumulation.
    pub fn fit_wls(x: &[T], y: &[T], weights: &[T], window_radius: T) -> Self {
        let n = x.len();
        if n == 0 {
            return Self::zero();
        }

        // SIMD-optimized single-pass accumulation
        let (sum_w, sum_wx, sum_wy, sum_wxx, sum_wxy) = T::accumulate_wls(x, y, weights);

        // Numerical stability tolerance
        let abs_tol = T::from(1e-7).unwrap();
        let rel_tol = T::epsilon() * window_radius * window_radius;
        let tol = abs_tol.max(rel_tol);

        // Solve for slope and intercept
        match T::solve_wls(sum_w, sum_wx, sum_wy, sum_wxx, sum_wxy, tol) {
            Some((slope, intercept, x_mean, y_mean)) => Self {
                slope,
                intercept,
                x_mean,
                y_mean,
            },
            None => Self::zero(),
        }
    }
}

// ============================================================================
// Regression Context
// ============================================================================

/// Context containing all data needed to fit a single point.
pub struct RegressionContext<'a, T: Float> {
    /// Slice of x-values (independent variable)
    pub x: &'a [T],

    /// Slice of y-values (dependent variable)
    pub y: &'a [T],

    /// Index of the point to fit
    pub idx: usize,

    /// Window for the local fit (defines neighborhood)
    pub window: Window,

    /// Whether to use robustness weights
    pub use_robustness: bool,

    /// Slice of robustness weights (all 1.0 if not using robustness)
    pub robustness_weights: &'a [T],

    /// Mutable slice of weights to be used in fitting
    pub weights: &'a mut [T],

    /// Weight function (kernel)
    pub weight_function: WeightFunction,

    /// Zero-weight fallback policy
    pub zero_weight_fallback: ZeroWeightFallback,
}

impl<'a, T: Float + WLSSolver> RegressionContext<'a, T> {
    /// Perform the local linear fit using the context configuration.
    ///
    /// This orchestrates the weight calculation, robustness application,
    /// and final weighted least squares solver.
    pub fn fit(&mut self) -> Option<T> {
        let n = self.x.len();

        if self.idx >= n || self.window.left >= n || self.window.right >= n {
            return None;
        }

        let x_current = self.x[self.idx];
        let window_radius = self.window.max_distance(self.x, x_current);

        if window_radius <= T::zero() {
            let mut sum_w = T::zero();
            let mut sum_wy = T::zero();
            let mut j = self.window.left;
            while j <= self.window.right {
                let w = if self.use_robustness {
                    self.robustness_weights[j]
                } else {
                    T::one()
                };
                sum_w = sum_w + w;
                sum_wy = sum_wy + w * self.y[j];
                j += 1;
            }

            if sum_w > T::zero() {
                return Some(sum_wy / sum_w);
            } else {
                return match self.zero_weight_fallback {
                    ZeroWeightFallback::UseLocalMean => {
                        let window_size = self.window.len();
                        let mean = self.y[self.window.left..=self.window.right]
                            .iter()
                            .copied()
                            .fold(T::zero(), |acc, v| acc + v)
                            / T::from(window_size).unwrap_or(T::one());
                        Some(mean)
                    }
                    ZeroWeightFallback::ReturnOriginal => Some(self.y[self.idx]),
                    ZeroWeightFallback::ReturnNone => None,
                };
            }
        }

        let weight_params = WeightParams::new(x_current, window_radius, self.use_robustness);

        let (mut weight_sum, rightmost_idx) = self.weight_function.compute_window_weights(
            self.x,
            self.window.left,
            self.window.right,
            weight_params.x_current,
            weight_params.window_radius,
            weight_params.h1,
            weight_params.h9,
            self.weights,
        );

        if self.use_robustness {
            weight_sum = T::zero();
            let mut j = self.window.left;
            while j <= rightmost_idx {
                let w_k = self.weights[j];
                if w_k > T::zero() {
                    let w_robust = self.robustness_weights[j];
                    let w_final = w_k * w_robust;
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
                    return Some(mean);
                }
                ZeroWeightFallback::ReturnOriginal => return Some(self.y[self.idx]),
                ZeroWeightFallback::ReturnNone => return None,
            }
        }

        let window_x = &self.x[self.window.left..=rightmost_idx];
        let window_y = &self.y[self.window.left..=rightmost_idx];
        let window_weights = &self.weights[self.window.left..=rightmost_idx];

        let model = LinearFit::fit_wls(window_x, window_y, window_weights, window_radius);
        Some(model.predict(x_current))
    }
}
