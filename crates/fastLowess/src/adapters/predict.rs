//! Parallel out-of-sample prediction for fitted Batch models.

#[cfg(feature = "cpu")]
use num_traits::Float;
#[cfg(feature = "cpu")]
use rayon::prelude::*;

#[cfg(feature = "cpu")]
use lowess::internals::adapters::predict::predict_one_full;
#[cfg(feature = "cpu")]
use lowess::internals::algorithms::regression::WLSSolver;
#[cfg(feature = "cpu")]
use lowess::internals::engine::executor::{PredictQuery, PredictState, RawPredictValues};
#[cfg(feature = "cpu")]
use lowess::internals::primitives::errors::LowessError;

// Evaluate a fitted Batch model at a batch of out-of-sample query points in parallel.
// Injected as `PredictState::custom_predict_pass` by the Batch adapter's `fit()` when
// `.parallel(true)` is set, mirroring `smooth_pass_parallel`'s per-thread scratch buffer.
// Computes only the per-point values; the shared confidence/prediction interval math is
// applied afterward by `lowess`'s `predict_batch`.
#[cfg(feature = "cpu")]
pub fn predict_pass_parallel<T>(
    state: &PredictState<T>,
    new_x: &[T],
    options: &PredictQuery<T>,
    need_se: bool,
) -> RawPredictValues<T>
where
    T: Float + Send + Sync + WLSSolver,
{
    let n = state.x.len();
    if n == 0 {
        let len = new_x.len();
        return Ok((
            vec![T::zero(); len],
            options.return_derivative().then(|| vec![T::zero(); len]),
            need_se.then(|| vec![T::zero(); len]),
        ));
    }

    let results: Vec<(T, T, Option<T>)> = new_x
        .par_iter()
        .map_init(
            || vec![T::zero(); n],
            |weights, &x_query| predict_one_full(state, x_query, weights, options, need_se),
        )
        .collect::<Result<Vec<_>, LowessError>>()?;

    let mut y = Vec::with_capacity(results.len());
    let mut derivative = options
        .return_derivative()
        .then(|| Vec::with_capacity(results.len()));
    let mut se = need_se.then(|| Vec::with_capacity(results.len()));

    for (yi, slope, sei) in results {
        y.push(yi);
        if let Some(d) = derivative.as_mut() {
            d.push(slope);
        }
        if let Some(s) = se.as_mut() {
            s.push(sei.unwrap_or(T::zero()));
        }
    }

    Ok((y, derivative, se))
}
