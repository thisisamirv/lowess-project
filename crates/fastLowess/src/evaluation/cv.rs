//! Parallel cross-validation for LOWESS bandwidth selection.
//!
//! This module provides the parallel cross-validation logic for selecting the
//! optimal smoothing fraction. It utilizes all available CPU cores to evaluate
//! multiple candidate fractions concurrently.

// External dependencies
#[cfg(feature = "cpu")]
use num_traits::Float;
#[cfg(feature = "cpu")]
use rayon::prelude::*;
#[cfg(feature = "cpu")]
use std::cmp::Ordering::Equal;
#[cfg(feature = "cpu")]
use std::fmt::Debug;

// Export dependencies from lowess crate
#[cfg(feature = "cpu")]
use lowess::internals::algorithms::regression::WLSSolver;
#[cfg(feature = "cpu")]
use lowess::internals::engine::executor::{LowessBuffer, LowessConfig, LowessExecutor};
#[cfg(feature = "cpu")]
use lowess::internals::evaluation::cv::CVKind;
#[cfg(feature = "cpu")]
use lowess::internals::primitives::buffer::CVBuffer;

// Perform cross-validation to select the best fraction in parallel.
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
