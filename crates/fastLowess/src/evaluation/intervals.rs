//! Parallel interval estimation for LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides parallel logic for computing standard errors and
//! confidence/prediction intervals. It distributes the uncertainty calculations
//! across CPU cores for maximum performance on large datasets.
//!
//! ## Design notes
//!
//! * **Parallelism**: Uses `rayon` to parallelize SE calculations across data points.
//! * **Integration**: Plugs into the execution engine via the `IntervalPassFn` hook.
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Parallel SE Calculation**: Each point's standard error is computed on a separate thread.
//! * **Window Serialization**: Efficiently handles neighborhood searches in parallel.
//!
//! ## Invariants
//!
//! * Standard errors are non-negative.
//! * Thread-local computations do not interfere with shared data.
//!
//! ## Non-goals
//!
//! * This module does not perform the actual interval slicing (delegated to `lowess`).
//! * This module does not handle the mapping to specific probability coverages.

// Feature-gated imports
#[cfg(feature = "cpu")]
use rayon::prelude::*;

// External dependencies
use num_traits::Float;

// Export dependencies from lowess crate
use lowess::internals::evaluation::intervals::IntervalMethod;
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::primitives::window::Window;

/// Perform interval estimation in parallel.
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
