//! Parallel execution engine for LOWESS smoothing operations.
//!
//! ## Purpose
//!
//! This module provides the parallel smoothing function that is injected into
//! the `lowess` crate's execution engine. It enables multi-threaded execution
//! of the local regression fits, significantly speeding up LOWESS smoothing
//! for large datasets by utilizing all available CPU cores.
//!
//! ## Design notes
//!
//! * **Implementation**: Provides a drop-in replacement for the sequential smoothing pass.
//! * **Parallelism**: Uses `rayon` for data-parallel execution across CPU cores.
//! * **Optimization**: Reuses weight buffers per thread to minimize allocations.
//! * **Interpolation**: Supports delta optimization for sparse fitting.
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Parallel Fitting**: Distributes points across CPU cores independently.
//! * **Delta Optimization**: Fits only "anchor" points and interpolates between them.
//! * **Buffer Reuse**: Thread-local scratch buffers to avoid allocation overhead.
//! * **Integration**: Plugs into the `lowess` executor via the `SmoothPassFn` hook.
//!
//! ## Invariants
//!
//! * Input x-values are assumed to be monotonically increasing (sorted).
//! * All buffers have the same length as the input data.
//! * Robustness weights are expected to be in [0, 1].
//! * Window size is at least 1 and at most n.
//!
//! ## Non-goals
//!
//! * This module does not handle the iteration loop (handled by `lowess::executor`).
//! * This module does not validate input data (handled by `validator`).
//! * This module does not sort input data (caller's responsibility).

// Feature-gated imports
#[cfg(feature = "cpu")]
use rayon::prelude::*;

// External dependencies
use num_traits::Float;

// Export dependencies from lowess crate
use lowess::internals::algorithms::regression::{RegressionContext, WLSSolver, ZeroWeightFallback};
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::primitives::window::Window;

// ============================================================================
// Parallel Smoothing Function
// ============================================================================

/// Perform a single smoothing pass over all points in parallel.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cpu")]
pub fn smooth_pass_parallel<T>(
    x: &[T],
    y: &[T],
    window_size: usize,
    delta: T,
    use_robustness: bool,
    robustness_weights: &[T],
    y_smooth: &mut [T],
    weight_function: WeightFunction,
    zero_weight_flag: u8,
) where
    T: Float + Send + Sync + WLSSolver,
{
    let n = x.len();
    if n == 0 {
        return;
    }

    let zero_weight_fallback = ZeroWeightFallback::from_u8(zero_weight_flag);

    // If delta > 0, use delta optimization with anchor points
    if delta > T::zero() && n > 2 {
        // Step 1: Pre-compute anchor points (points to fit explicitly)
        let anchors = compute_anchor_points(x, delta);

        if anchors.is_empty() {
            // Fallback: fit all points if no anchors computed
            fit_all_points_parallel(
                x,
                y,
                window_size,
                use_robustness,
                robustness_weights,
                y_smooth,
                weight_function,
                zero_weight_fallback,
            );
            return;
        }

        // Step 2: Parallel fit anchor points
        let anchor_values: Vec<(usize, T)> = anchors
            .par_iter()
            .map_init(
                || vec![T::zero(); n],
                |weights, &i| {
                    weights.fill(T::zero());

                    let mut window = Window::initialize(i, window_size, n);
                    window.recenter(x, i, n);

                    let mut ctx = RegressionContext {
                        x,
                        y,
                        idx: i,
                        window,
                        use_robustness,
                        robustness_weights: if use_robustness {
                            robustness_weights
                        } else {
                            &[]
                        },
                        weights,
                        weight_function,
                        zero_weight_fallback,
                    };

                    (i, ctx.fit().unwrap_or(y[i]))
                },
            )
            .collect();

        // Step 3: Write anchor values, handle ties, and interpolate
        for &(idx, value) in &anchor_values {
            y_smooth[idx] = value;

            // Handle potential ties following this anchor
            let x_val = x[idx];
            for i in (idx + 1)..n {
                if x[i] == x_val {
                    y_smooth[i] = value;
                } else {
                    break;
                }
            }
        }

        // Interpolate between consecutive anchors
        for window in anchors.windows(2) {
            let start = window[0];
            let end = window[1];

            // Only interpolate if there's a gap that wasn't already filled by ties
            let mut gap_start = start;
            let x_start = x[start];
            while gap_start < end && x[gap_start] == x_start {
                gap_start += 1;
            }

            if gap_start < end {
                interpolate_gap(x, y_smooth, gap_start - 1, end);
            }
        }

        // Handle any remaining points after the last anchor
        if let Some(&last_anchor) = anchors.last() {
            if last_anchor < n - 1 {
                // Check if last point is a tie with the last anchor
                if x[n - 1] == x[last_anchor] {
                    y_smooth[n - 1] = y_smooth[last_anchor];
                } else {
                    // Fit the last point sequentially using a reused weights buffer to avoid allocation
                    let mut weights = vec![T::zero(); n];
                    let mut window = Window::initialize(n - 1, window_size, n);
                    window.recenter(x, n - 1, n);

                    let mut ctx = RegressionContext {
                        x,
                        y,
                        idx: n - 1,
                        window,
                        use_robustness,
                        robustness_weights: if use_robustness {
                            robustness_weights
                        } else {
                            &[]
                        },
                        weights: &mut weights,
                        weight_function,
                        zero_weight_fallback,
                    };

                    y_smooth[n - 1] = ctx.fit().unwrap_or(y[n - 1]);

                    // Same tie/gap logic for the final stretch
                    let mut gap_start = last_anchor;
                    let x_start = x[last_anchor];
                    while gap_start < n - 1 && x[gap_start] == x_start {
                        gap_start += 1;
                    }
                    if gap_start < n - 1 {
                        interpolate_gap(x, y_smooth, gap_start - 1, n - 1);
                    }
                }
            }
        }
    } else {
        // No delta optimization: fit all points in parallel
        fit_all_points_parallel(
            x,
            y,
            window_size,
            use_robustness,
            robustness_weights,
            y_smooth,
            weight_function,
            zero_weight_fallback,
        );
    }
}

/// Compute anchor points for delta optimization using O(log n) binary search.
/// For large datasets (n > 100K), uses parallel chunking with sequential merge.
#[cfg(feature = "cpu")]
fn compute_anchor_points<T: Float + Send + Sync>(x: &[T], delta: T) -> Vec<usize> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    // For small datasets, use sequential algorithm
    const PARALLEL_THRESHOLD: usize = 100_000;
    if n < PARALLEL_THRESHOLD {
        return compute_anchor_points_sequential(x, delta);
    }

    // Parallel strategy: divide into chunks, compute local anchors, then merge
    let chunk_size = n / rayon::current_num_threads().max(1);
    let chunk_size = chunk_size.max(PARALLEL_THRESHOLD / 4); // Ensure chunks aren't too small

    // Phase 1: Parallel compute chunk boundaries that are valid anchor candidates
    let chunk_boundaries: Vec<usize> = (0..n).into_par_iter().step_by(chunk_size).collect();

    // Phase 2: For each chunk, find local anchors in parallel
    let local_anchor_sets: Vec<Vec<usize>> = chunk_boundaries
        .par_iter()
        .enumerate()
        .map(|(chunk_idx, &start)| {
            let end = if chunk_idx + 1 < chunk_boundaries.len() {
                chunk_boundaries[chunk_idx + 1]
            } else {
                n
            };
            compute_anchor_points_in_range(x, delta, start, end)
        })
        .collect();

    // Phase 3: Sequential merge respecting delta constraints
    let mut anchors = Vec::with_capacity(n / 100);
    let mut last_anchor_x = x[0] - delta - delta; // Ensure first point is always included

    for chunk_anchors in local_anchor_sets {
        for &anchor in &chunk_anchors {
            let x_anchor = x[anchor];
            if x_anchor - last_anchor_x >= delta || anchors.is_empty() {
                anchors.push(anchor);
                last_anchor_x = x_anchor;
            }
        }
    }

    // Ensure last point is included
    if anchors.is_empty() || *anchors.last().unwrap() != n - 1 {
        anchors.push(n - 1);
    }

    anchors
}

/// Sequential anchor computation for small datasets or chunk ranges.
#[cfg(feature = "cpu")]
fn compute_anchor_points_sequential<T: Float>(x: &[T], delta: T) -> Vec<usize> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    let mut anchors = vec![0];
    let mut last_fitted = 0usize;

    while last_fitted < n - 1 {
        let cutpoint = x[last_fitted] + delta;
        let next_idx = x[last_fitted + 1..].partition_point(|&xi| xi <= cutpoint) + last_fitted + 1;

        let x_last = x[last_fitted];
        let mut tie_end = last_fitted;
        for (i, &xi) in x
            .iter()
            .enumerate()
            .take(next_idx.min(n))
            .skip(last_fitted + 1)
        {
            if xi == x_last {
                tie_end = i;
            } else {
                break;
            }
        }
        if tie_end > last_fitted {
            last_fitted = tie_end;
        }

        let current = usize::max(next_idx.saturating_sub(1), last_fitted + 1).min(n - 1);
        if current <= last_fitted {
            break;
        }

        anchors.push(current);
        last_fitted = current;
    }

    if *anchors.last().unwrap_or(&0) != n - 1 {
        anchors.push(n - 1);
    }

    anchors
}

/// Compute anchors within a specific range [start, end).
#[cfg(feature = "cpu")]
fn compute_anchor_points_in_range<T: Float>(
    x: &[T],
    delta: T,
    start: usize,
    end: usize,
) -> Vec<usize> {
    if start >= end {
        return vec![];
    }

    let mut anchors = vec![start];
    let mut last_fitted = start;

    while last_fitted < end - 1 {
        let cutpoint = x[last_fitted] + delta;
        let search_start = last_fitted + 1;
        let search_end = end;

        let next_idx =
            x[search_start..search_end].partition_point(|&xi| xi <= cutpoint) + search_start;

        let x_last = x[last_fitted];
        let mut tie_end = last_fitted;
        for (i, &xi) in x
            .iter()
            .enumerate()
            .take(next_idx.min(end))
            .skip(last_fitted + 1)
        {
            if xi == x_last {
                tie_end = i;
            } else {
                break;
            }
        }
        if tie_end > last_fitted {
            last_fitted = tie_end;
        }

        let current = usize::max(next_idx.saturating_sub(1), last_fitted + 1).min(end - 1);
        if current <= last_fitted {
            break;
        }

        anchors.push(current);
        last_fitted = current;
    }

    anchors
}

/// Linearly interpolate between two fitted anchor points.
#[cfg(feature = "cpu")]
fn interpolate_gap<T: Float>(x: &[T], y_smooth: &mut [T], start: usize, end: usize) {
    if end <= start + 1 {
        return;
    }

    let x0 = x[start];
    let x1 = x[end];
    let y0 = y_smooth[start];
    let y1 = y_smooth[end];

    let denom = x1 - x0;
    if denom <= T::zero() {
        let avg = (y0 + y1) / T::from(2.0).unwrap();
        y_smooth[(start + 1)..end].fill(avg);
        return;
    }

    let slope = (y1 - y0) / denom;
    for k in (start + 1)..end {
        y_smooth[k] = y0 + (x[k] - x0) * slope;
    }
}

/// Fit all points in parallel (no delta optimization).
/// Uses cache-aware tile processing for large datasets.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cpu")]
fn fit_all_points_parallel<T>(
    x: &[T],
    y: &[T],
    window_size: usize,
    use_robustness: bool,
    robustness_weights: &[T],
    y_smooth: &mut [T],
    weight_function: WeightFunction,
    zero_weight_fallback: ZeroWeightFallback,
) where
    T: Float + Send + Sync + WLSSolver,
{
    let n = x.len();
    if n == 0 {
        return;
    }

    // For very large window sizes, use tile-based processing for cache locality
    // Tile size chosen to fit x,y,weights for window in L2 cache (~256KB)
    // Each f64 is 8 bytes, so ~10K elements per tile is reasonable
    const TILE_SIZE: usize = 8192;
    const USE_TILING_THRESHOLD: usize = 50_000;

    if n >= USE_TILING_THRESHOLD && window_size > TILE_SIZE / 4 {
        // Tile-based processing: group points by their window overlap
        fit_all_points_tiled(
            x,
            y,
            window_size,
            use_robustness,
            robustness_weights,
            y_smooth,
            weight_function,
            zero_weight_fallback,
        );
    } else {
        // Standard parallel processing for smaller datasets
        fit_all_points_standard(
            x,
            y,
            window_size,
            use_robustness,
            robustness_weights,
            y_smooth,
            weight_function,
            zero_weight_fallback,
        );
    }
}

/// Standard parallel fit (original implementation).
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cpu")]
fn fit_all_points_standard<T>(
    x: &[T],
    y: &[T],
    window_size: usize,
    use_robustness: bool,
    robustness_weights: &[T],
    y_smooth: &mut [T],
    weight_function: WeightFunction,
    zero_weight_fallback: ZeroWeightFallback,
) where
    T: Float + Send + Sync + WLSSolver,
{
    let n = x.len();

    (0..n)
        .into_par_iter()
        .zip(y_smooth.par_iter_mut())
        .for_each_init(
            || vec![T::zero(); n],
            |weights, (i, smoothed_val)| {
                weights.fill(T::zero());

                let mut window = Window::initialize(i, window_size, n);
                window.recenter(x, i, n);

                let mut ctx = RegressionContext {
                    x,
                    y,
                    idx: i,
                    window,
                    use_robustness,
                    robustness_weights: if use_robustness {
                        robustness_weights
                    } else {
                        &[]
                    },
                    weights,
                    weight_function,
                    zero_weight_fallback,
                };

                *smoothed_val = ctx.fit().unwrap_or(y[i]);
            },
        );
}

/// Tile-based parallel fit for better cache locality on large datasets.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cpu")]
fn fit_all_points_tiled<T>(
    x: &[T],
    y: &[T],
    window_size: usize,
    use_robustness: bool,
    robustness_weights: &[T],
    y_smooth: &mut [T],
    weight_function: WeightFunction,
    zero_weight_fallback: ZeroWeightFallback,
) where
    T: Float + Send + Sync + WLSSolver,
{
    let n = x.len();
    const TILE_SIZE: usize = 8192;

    // Split y_smooth into chunks for parallel mutable access
    let y_chunks: Vec<&mut [T]> = y_smooth.chunks_mut(TILE_SIZE).collect();

    y_chunks
        .into_par_iter()
        .enumerate()
        .for_each(|(tile_idx, y_chunk)| {
            let tile_start = tile_idx * TILE_SIZE;
            let tile_end = (tile_start + TILE_SIZE).min(n);
            let chunk_len = tile_end - tile_start;

            // Thread-local weight buffer
            let mut weights = vec![T::zero(); n];

            for (local_i, smoothed_val) in y_chunk.iter_mut().enumerate().take(chunk_len) {
                let i = tile_start + local_i;
                weights.fill(T::zero());

                let mut window = Window::initialize(i, window_size, n);
                window.recenter(x, i, n);

                let mut ctx = RegressionContext {
                    x,
                    y,
                    idx: i,
                    window,
                    use_robustness,
                    robustness_weights: if use_robustness {
                        robustness_weights
                    } else {
                        &[]
                    },
                    weights: &mut weights,
                    weight_function,
                    zero_weight_fallback,
                };

                *smoothed_val = ctx.fit().unwrap_or(y[i]);
            }
        });
}
