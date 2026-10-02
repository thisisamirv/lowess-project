//! Parallel interval estimation for LOWESS smoothing.
//!
//! This module provides parallel logic for computing standard errors and
//! confidence/prediction intervals. It distributes the uncertainty calculations
//! across CPU cores for maximum performance on large datasets.
// ## srrstats Compliance
//
// @srrstats {RE5.0} Parallel SE computation for confidence/prediction intervals.
// @srrstats {G3.0} Rayon par_iter for pointwise interval estimation and bootstrap refits.

// External dependencies
#[cfg(feature = "cpu")]
use num_traits::Float;
#[cfg(feature = "cpu")]
use rayon::prelude::*;
#[cfg(feature = "cpu")]
use std::fmt::Debug;

// Export dependencies from lowess crate
#[cfg(feature = "cpu")]
use lowess::internals::algorithms::regression::WLSSolver;
#[cfg(feature = "cpu")]
use lowess::internals::engine::executor::{LowessConfig, LowessExecutor};
#[cfg(feature = "cpu")]
use lowess::internals::evaluation::intervals::IntervalMethod;
#[cfg(feature = "cpu")]
use lowess::internals::math::kernel::WeightFunction;
#[cfg(feature = "cpu")]
use lowess::internals::primitives::errors::LowessError;
#[cfg(feature = "cpu")]
use lowess::internals::primitives::window::Window;

// Perform interval estimation in parallel.
#[cfg(feature = "cpu")]
#[allow(clippy::too_many_arguments)]
pub fn interval_pass_parallel<T>(
    x: &[T],
    y: &[T],
    y_smooth: &[T],
    window_size: usize,
    robustness_weights: &[T],
    custom_weights: Option<&[T]>,
    weight_function: WeightFunction,
    method: &IntervalMethod<T>,
) -> Vec<T>
where
    T: Float + Send + Sync + 'static,
{
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }

    // Early exit if no intervals or SE requested
    if !method.se && !method.confidence && !method.prediction {
        return vec![T::zero(); n];
    }

    // Parallelize over indices
    (0..n)
        .into_par_iter()
        .map(|i| {
            // Initialize and center window
            let mut window = Window::initialize(i, window_size, n);
            window.recenter(x, i, n);

            let idx = i;
            let left = window.left;
            let right = window.right;

            // Compute bandwidth
            let x_current = x[idx];
            let bandwidth_left = x_current - x[left];
            let bandwidth_right = x[right] - x_current;
            let bandwidth = T::max(bandwidth_left, bandwidth_right);

            if bandwidth <= T::zero() {
                return T::zero();
            }

            // Compute weight for current point (distance = 0)
            let u_idx = T::zero();
            let kernel_val = weight_function.compute_weight(u_idx);
            let w_idx = kernel_val
                * custom_weights.map_or(T::one(), |weights| weights[idx])
                * robustness_weights[idx];

            // Accumulate weighted residual variance and local design moments.
            let mut sum_w_r2 = T::zero();
            let mut sum_w = T::zero();
            let mut s1 = T::zero();
            let mut s2 = T::zero();
            let mut t0 = T::zero();
            let mut t1 = T::zero();
            let mut t2 = T::zero();

            for j in left..=right {
                let dist = (x[j] - x_current).abs();
                let u = dist / bandwidth;
                let w = if j == idx {
                    w_idx
                } else {
                    weight_function.compute_weight(u)
                        * custom_weights.map_or(T::one(), |weights| weights[j])
                        * robustness_weights[j]
                };

                let r = y[j] - y_smooth[j];
                let dx = x[j] - x_current;
                sum_w_r2 = sum_w_r2 + w * r * r;
                sum_w = sum_w + w;
                s1 = s1 + w * dx;
                s2 = s2 + w * dx * dx;
                t0 = t0 + w * w;
                t1 = t1 + w * w * dx;
                t2 = t2 + w * w * dx * dx;
            }

            IntervalMethod::compute_se(sum_w, sum_w_r2, s1, s2, t0, t1, t2)
        })
        .collect()
}

// Refit residual-bootstrap replicates in parallel (one replicate per task).
#[cfg(feature = "cpu")]
pub fn bootstrap_pass_parallel<T>(
    x: &[T],
    replicates: &[Vec<T>],
    config: &LowessConfig<T>,
) -> Result<Vec<Vec<T>>, LowessError>
where
    T: Float + WLSSolver + Debug + Send + Sync + 'static,
{
    // Parallelism is across replicates, so each refit runs its own passes sequentially.
    let mut refit = config.clone();
    refit.parallel = false;
    refit.custom_smooth_pass = None;
    refit.custom_cv_pass = None;
    refit.custom_interval_pass = None;
    refit.custom_derivative_pass = None;
    refit.custom_predict_pass = None;

    replicates
        .par_iter()
        .map(|y_star| LowessExecutor::run_with_config(x, y_star, refit.clone()).map(|r| r.smoothed))
        .collect()
}
