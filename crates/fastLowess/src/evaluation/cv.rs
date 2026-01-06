//! Parallel cross-validation for LOWESS bandwidth selection.
//!
//! ## Purpose
//!
//! This module provides the parallel cross-validation logic for selecting the
//! optimal smoothing fraction. It utilizes all available CPU cores to evaluate
//! multiple candidate fractions concurrently.
//!
//! ## Design notes
//!
//! * **Parallelism**: Uses `rayon` to evaluate candidate fractions in parallel.
//! * **Integration**: Plugs into the iteration loop via the `CVPassFn` hook.
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Parallel Evaluation**: Evaluates each fraction candidate on a separate thread.
//! * **RMSE Optimization**: Identifies the fraction that minimizes prediction error.
//!
//! ## Invariants
//!
//! * Best fraction minimizes RMSE across all evaluated candidates.
//! * Parallel execution maintains data isolation between threads.
//!
//! ## Non-goals
//!
//! * This module does not implement the CV partitioning logic (delegated to `lowess`).
//! * This module does not perform the actual smoothing fits directly.

// Feature-gated imports
#[cfg(feature = "cpu")]
use rayon::prelude::*;

// External dependencies
use num_traits::Float;
use std::cmp::Ordering::Equal;
use std::fmt::Debug;

// Export dependencies from lowess crate
use lowess::internals::algorithms::regression::WLSSolver;
use lowess::internals::engine::executor::{LowessBuffer, LowessConfig, LowessExecutor};
use lowess::internals::evaluation::cv::CVKind;
use lowess::internals::primitives::buffer::CVBuffer;

/// Perform cross-validation to select the best fraction in parallel.
#[cfg(feature = "cpu")]
pub fn cv_pass_parallel<T>(
    x: &[T],
    y: &[T],
    fractions: &[T],
    method: CVKind,
    config: &LowessConfig<T>,
) -> (T, Vec<T>)
where
    T: Float + Send + Sync + Debug + WLSSolver + 'static,
{
    if fractions.is_empty() {
        return (T::zero(), Vec::new());
    }

    // Parallelize over candidate fractions
    let scores: Vec<T> = fractions
        .par_iter()
        .map_init(
            || (CVBuffer::new(), LowessBuffer::default()),
            |init, &frac| {
                let (cv_buffer, lowess_buffer) = init;
                // Use the base CV logic for a single fraction
                // This ensures exact consistency with the sequential implementation in 'lowess'
                let (_, s) = method.run(
                    x,
                    y,
                    1,
                    &[frac],
                    config.cv_seed,
                    |tx, ty, f| {
                        let mut fold_config = config.clone();
                        fold_config.fraction = Some(f);
                        fold_config.cv_fractions = None;

                        let executor = LowessExecutor::from_config(&fold_config);
                        executor.run(tx, ty, Some(lowess_buffer)).smoothed
                    },
                    Option::<&mut fn(&[T], &[T], &[T], T) -> Vec<T>>::None,
                    cv_buffer,
                );
                s[0]
            },
        )
        .collect();

    let best_idx = scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    (fractions[best_idx], scores)
}
