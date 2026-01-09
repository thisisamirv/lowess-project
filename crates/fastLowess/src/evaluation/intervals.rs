//! Parallel interval estimation for LOWESS smoothing.
//!
//! This module provides parallel logic for computing standard errors and
//! confidence/prediction intervals. It distributes the uncertainty calculations
//! across CPU cores for maximum performance on large datasets.

// External dependencies
#[cfg(feature = "cpu")]
use num_traits::Float;
#[cfg(feature = "cpu")]
use rayon::prelude::*;

// Export dependencies from lowess crate
#[cfg(feature = "cpu")]
use lowess::internals::evaluation::intervals::IntervalMethod;
#[cfg(feature = "cpu")]
use lowess::internals::math::kernel::WeightFunction;
#[cfg(feature = "cpu")]
use lowess::internals::primitives::window::Window;

// Perform interval estimation in parallel.
#[cfg(feature = "cpu")]
pub fn interval_pass_parallel<T>(
    x: &[T],
    y: &[T],
    y_smooth: &[T],
    window_size: usize,
    robustness_weights: &[T],
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
            let w_idx = weight_function.compute_weight(u_idx) * robustness_weights[idx];

            // Accumulate weighted residual variance
            let mut sum_w_r2 = T::zero();
            let mut sum_w = T::zero();

            for j in left..=right {
                let dist = (x[j] - x_current).abs();
                let u = dist / bandwidth;
                let w = if j == idx {
                    w_idx
                } else {
                    weight_function.compute_weight(u) * robustness_weights[j]
                };

                let r = y[j] - y_smooth[j];
                sum_w_r2 = sum_w_r2 + w * r * r;
                sum_w = sum_w + w;
            }

            IntervalMethod::compute_se(sum_w, sum_w_r2, w_idx)
        })
        .collect()
}
