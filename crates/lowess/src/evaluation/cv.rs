//! Cross-validation for LOWESS bandwidth selection.
//!
//! ## Purpose
//!
//! This module provides cross-validation tools for selecting the optimal
//! smoothing fraction (bandwidth) in LOWESS regression. It implements
//! generic k-fold and leave-one-out cross-validation strategies.
//!
//! ## Design notes
//!
//! * **Generic Strategy**: Supports both k-fold and leave-one-out (LOOCV).
//! * **Interpolation**: Uses linear interpolation for minimizing prediction error.
//! * **Optimization**: Selects the fraction that minimizes RMSE.
//!
//! ## Key concepts
//!
//! * **K-Fold**: Partitions data into k subsamples (train on k-1, test on 1).
//! * **LOOCV**: Extreme case where k equals sample size (n iterations).
//! * **Interpolation**: Linear interpolation handles test points outside training set.
//!
//! ## Invariants
//!
//! * Training and test sets are disjoint in each fold.
//! * The best fraction minimizes RMSE across all folds.
//!
//! ## Non-goals
//!
//! * This module does not perform the actual smoothing (done via callback).
//! * This module does not provide confidence intervals for CV scores.

// Feature-gated dependencies
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::cmp::Ordering::Equal;
use core::fmt::Debug;
use num_traits::Float;

// Internal dependencies
use crate::primitives::buffer::CVBuffer;

// ============================================================================
// Internal PRNG
// ============================================================================

/// Minimal PRNG for no-std shuffling.
///
/// Uses an LCG (Linear Congruential Generator) with constants from PCG/MQL.
#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        // LCG constants for 64-bit state
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }
}

// ============================================================================
// Internal CV Kind (for storage)
// ============================================================================

/// Internal representation of CV method for storage (no lifetime needed).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CVKind {
    /// K-fold cross-validation with k folds.
    KFold(usize),
    /// Leave-one-out cross-validation.
    #[allow(clippy::upper_case_acronyms)]
    LOOCV,
}

// ============================================================================
// Cross-Validation Configuration
// ============================================================================

/// Cross-validation configuration combining strategy, fractions, and seed.
#[derive(Debug, Clone)]
pub struct CVConfig<'a, T> {
    /// The CV strategy kind.
    pub(crate) kind: CVKind,
    /// Candidate smoothing fractions to evaluate.
    pub(crate) fractions: &'a [T],
    /// Random seed for reproducible fold shuffling (K-Fold only).
    pub(crate) seed: Option<u64>,
}

impl<'a, T> CVConfig<'a, T> {
    /// Set the random seed for reproducible K-Fold cross-validation.
    ///
    /// The seed controls shuffling of data indices before fold assignment.
    /// Using the same seed produces identical fold assignments across runs.
    ///
    /// # Note
    ///
    /// This only affects K-Fold CV. LOOCV is deterministic and ignores the seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get the fractions slice.
    pub fn fractions(&self) -> &[T] {
        self.fractions
    }

    /// Get the CV kind for internal use.
    pub(crate) fn kind(&self) -> CVKind {
        self.kind
    }

    /// Get the seed for internal use.
    pub(crate) fn get_seed(&self) -> Option<u64> {
        self.seed
    }
}

/// Create a K-fold cross-validation configuration.
#[allow(non_snake_case)]
pub fn KFold<T>(k: usize, fractions: &[T]) -> CVConfig<'_, T> {
    CVConfig {
        kind: CVKind::KFold(k),
        fractions,
        seed: None,
    }
}

/// Create a leave-one-out cross-validation configuration.
#[allow(non_snake_case)]
pub fn LOOCV<T>(fractions: &[T]) -> CVConfig<'_, T> {
    CVConfig {
        kind: CVKind::LOOCV,
        fractions,
        seed: None,
    }
}

// ============================================================================
// Cross-Validation Execution
// ============================================================================

impl CVKind {
    /// Run cross-validation to select the best fraction.
    #[allow(clippy::too_many_arguments)]
    pub fn run<T, F, P>(
        self,
        x: &[T],
        y: &[T],
        dimensions: usize,
        fractions: &[T],
        seed: Option<u64>,
        mut smoother: F,
        mut predictor: Option<P>,
        cv_buffer: &mut CVBuffer<T>,
    ) -> (T, Vec<T>)
    where
        T: Float + Debug + Send + Sync + 'static,
        F: FnMut(&[T], &[T], T) -> Vec<T>,
        P: FnMut(&[T], &[T], &[T], T) -> Vec<T>,
    {
        match self {
            CVKind::KFold(k) => Self::kfold_cross_validation(
                x,
                y,
                dimensions,
                fractions,
                k,
                seed,
                &mut smoother,
                predictor.as_mut(),
                cv_buffer,
            ),
            CVKind::LOOCV => Self::leave_one_out_cross_validation(
                x,
                y,
                dimensions,
                fractions,
                &mut smoother,
                predictor.as_mut(),
                cv_buffer,
            ),
        }
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Build a data subset from a list of indices into provided scratch buffers.
    pub fn build_subset_inplace<T: Float>(
        x: &[T],
        y: &[T],
        dims: usize,
        indices: &[usize],
        tx: &mut Vec<T>,
        ty: &mut Vec<T>,
    ) {
        tx.clear();
        ty.clear();
        for &i in indices {
            let offset = i * dims;
            tx.extend_from_slice(&x[offset..offset + dims]);
            ty.push(y[i]);
        }
    }

    /// Build a data subset from a list of indices.
    pub fn build_subset_from_indices<T: Float>(
        x: &[T],
        y: &[T],
        dims: usize,
        indices: &[usize],
    ) -> (Vec<T>, Vec<T>) {
        let mut tx = Vec::with_capacity(indices.len() * dims);
        let mut ty = Vec::with_capacity(indices.len());
        Self::build_subset_inplace(x, y, dims, indices, &mut tx, &mut ty);
        (tx, ty)
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Interpolate prediction at a new x value given fitted training values.
    ///
    /// # Implementation notes
    ///
    /// * Uses binary search for O(log n) bracketing
    /// * Handles bracketing points with identical x-values by averaging their y-values
    /// * Constant extrapolation prevents unbounded predictions
    pub fn interpolate_prediction<T: Float>(x_train: &[T], y_train: &[T], x_new: T) -> T {
        let n = x_train.len();

        // Edge case: empty training set
        if n == 0 {
            return T::zero();
        }

        // Edge case: single training point
        if n == 1 {
            return y_train[0];
        }

        // Boundary handling: constant extrapolation
        if x_new <= x_train[0] {
            return y_train[0];
        }
        if x_new >= x_train[n - 1] {
            return y_train[n - 1];
        }

        // Binary search for bracketing points
        let mut left = 0;
        let mut right = n - 1;

        while right - left > 1 {
            let mid = (left + right) / 2;
            if x_train[mid] <= x_new {
                left = mid;
            } else {
                right = mid;
            }
        }

        // Linear interpolation between left and right
        let x0 = x_train[left];
        let x1 = x_train[right];
        let y0 = y_train[left];
        let y1 = y_train[right];

        let denom = x1 - x0;
        if denom <= T::zero() {
            // X-values are identical: return average of y-bracketing points
            return (y0 + y1) / T::from(2.0).unwrap();
        }

        let alpha = (x_new - x0) / denom;
        y0 + alpha * (y1 - y0)
    }

    /// Predict values at multiple new x points using linear interpolation.
    ///
    /// # Implementation notes
    ///
    /// * Leverages sorted order of `x_new` for O(n_train + n_new) linear scan.
    pub fn interpolate_prediction_batch<T: Float>(
        x_train: &[T],
        y_train: &[T],
        x_new: &[T],
        y_pred: &mut [T],
    ) {
        let n_train = x_train.len();
        let n_new = x_new.len();

        if n_new == 0 {
            return;
        }

        if n_train == 0 {
            y_pred.fill(T::zero());
            return;
        }

        if n_train == 1 {
            y_pred.fill(y_train[0]);
            return;
        }

        let mut left = 0;
        for i in 0..n_new {
            let xi = x_new[i];

            // Boundary handling: constant extrapolation
            if xi <= x_train[0] {
                y_pred[i] = y_train[0];
                continue;
            }
            if xi >= x_train[n_train - 1] {
                y_pred[i] = y_train[n_train - 1];
                continue;
            }

            // Linear scan forward to find bracket
            while left + 1 < n_train && x_train[left + 1] <= xi {
                left += 1;
            }

            let right = left + 1;
            let x0 = x_train[left];
            let x1 = x_train[right];
            let y0 = y_train[left];
            let y1 = y_train[right];

            let denom = x1 - x0;
            if denom <= T::zero() {
                y_pred[i] = (y0 + y1) / T::from(2.0).unwrap();
            } else {
                let alpha = (xi - x0) / denom;
                y_pred[i] = y0 + alpha * (y1 - y0);
            }
        }
    }

    // ========================================================================
    // Internal Cross-Validation Implementations
    // ========================================================================

    /// Select the best fraction based on cross-validation scores.
    fn select_best_fraction<T: Float>(fractions: &[T], scores: &[T]) -> (T, Vec<T>) {
        if fractions.is_empty() {
            return (T::zero(), Vec::new());
        }

        let best_idx = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        (fractions[best_idx], scores.to_vec())
    }

    /// Perform k-fold cross-validation.
    #[allow(clippy::too_many_arguments)]
    fn kfold_cross_validation<T, F, P>(
        x: &[T],
        y: &[T],
        dims: usize,
        fractions: &[T],
        k: usize,
        seed: Option<u64>,
        smoother: &mut F,
        mut predictor: Option<&mut P>,
        cv_buffer: &mut CVBuffer<T>,
    ) -> (T, Vec<T>)
    where
        T: Float + Debug + Send + Sync + 'static,
        F: FnMut(&[T], &[T], T) -> Vec<T>,
        P: FnMut(&[T], &[T], &[T], T) -> Vec<T>,
    {
        let n = x.len() / dims;
        if n < k || k < 2 {
            return (
                fractions.first().copied().unwrap_or(T::zero()),
                vec![T::zero(); fractions.len()],
            );
        }

        let fold_size = n / k;
        let mut cv_scores = vec![T::zero(); fractions.len()];

        // Generate indices and optionally shuffle if seed is provided
        let mut indices: Vec<usize> = (0..n).collect();
        if let Some(s) = seed {
            let mut rng = SimpleRng::new(s);
            for i in (1..n).rev() {
                let j = (rng.next_u32() as usize) % (i + 1);
                indices.swap(i, j);
            }
        }

        cv_buffer.ensure_capacity(n, dims);
        let mut fold_errors = vec![vec![T::zero(); k]; fractions.len()];
        let mut fold_sizes = vec![0usize; k];

        for fold in 0..k {
            let test_start = fold * fold_size;
            let test_end = if fold == k - 1 {
                n
            } else {
                (fold + 1) * fold_size
            };
            fold_sizes[fold] = test_end - test_start;

            // Build training and test sets once per fold
            let (tx, ty, tex, tey) = (
                &mut cv_buffer.train_x,
                &mut cv_buffer.train_y,
                &mut cv_buffer.test_x,
                &mut cv_buffer.test_y,
            );

            tx.clear();
            ty.clear();
            for &idx in &indices[0..test_start] {
                let offset = idx * dims;
                tx.extend_from_slice(&x[offset..offset + dims]);
                ty.push(y[idx]);
            }
            for &idx in &indices[test_end..n] {
                let offset = idx * dims;
                tx.extend_from_slice(&x[offset..offset + dims]);
                ty.push(y[idx]);
            }

            tex.clear();
            tey.clear();
            for &idx in &indices[test_start..test_end] {
                let offset = idx * dims;
                tex.extend_from_slice(&x[offset..offset + dims]);
                tey.push(y[idx]);
            }

            // Pre-sort training data if no custom predictor is used (1D LOWESS case)
            if predictor.is_none() {
                let mut train_data: Vec<(T, T)> = tx
                    .iter()
                    .zip(ty.iter())
                    .map(|(&xi, &yi)| (xi, yi))
                    .collect();
                train_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Equal));

                cv_buffer.sorted_train_x.clear();
                cv_buffer.sorted_train_y.clear();
                for (xi, yi) in train_data {
                    cv_buffer.sorted_train_x.push(xi);
                    cv_buffer.sorted_train_y.push(yi);
                }
            }

            for (frac_idx, &frac) in fractions.iter().enumerate() {
                let predictions = if let Some(ref mut p_fn) = predictor {
                    p_fn(tx, ty, tex, frac)
                } else {
                    let train_smooth =
                        smoother(&cv_buffer.sorted_train_x, &cv_buffer.sorted_train_y, frac);
                    let mut preds = vec![T::zero(); tex.len() / dims];
                    Self::interpolate_prediction_batch(
                        &cv_buffer.sorted_train_x,
                        &train_smooth,
                        tex,
                        &mut preds,
                    );
                    preds
                };

                let mut error_sum = T::zero();
                for (i, &predicted) in predictions.iter().enumerate() {
                    let actual = tey[i];
                    let error = actual - predicted;
                    error_sum = error_sum + error * error;
                }
                fold_errors[frac_idx][fold] = error_sum;
            }
        }

        // Aggregate results across folds
        for (frac_idx, _) in fractions.iter().enumerate() {
            let mut total_rmse = T::zero();
            let mut count = 0;
            for fold in 0..k {
                if fold_sizes[fold] > 0 {
                    let mse = fold_errors[frac_idx][fold] / T::from(fold_sizes[fold]).unwrap();
                    total_rmse = total_rmse + mse.sqrt();
                    count += 1;
                }
            }
            if count > 0 {
                cv_scores[frac_idx] = total_rmse / T::from(count).unwrap();
            } else {
                cv_scores[frac_idx] = T::infinity();
            }
        }

        Self::select_best_fraction(fractions, &cv_scores)
    }

    /// Perform leave-one-out cross-validation (LOOCV).
    fn leave_one_out_cross_validation<T, F, P>(
        x: &[T],
        y: &[T],
        dims: usize,
        fractions: &[T],
        smoother: &mut F,
        mut predictor: Option<&mut P>,
        cv_buffer: &mut CVBuffer<T>,
    ) -> (T, Vec<T>)
    where
        T: Float + Debug + Send + Sync + 'static,
        F: FnMut(&[T], &[T], T) -> Vec<T>,
        P: FnMut(&[T], &[T], &[T], T) -> Vec<T>,
    {
        let n = x.len() / dims;
        let mut cv_scores = vec![T::zero(); fractions.len()];

        cv_buffer.ensure_capacity(n, dims);

        let mut test_point = vec![T::zero(); dims];

        for (frac_idx, &frac) in fractions.iter().enumerate() {
            let mut total_error = T::zero();

            for i in 0..n {
                // Build training set (all points except i)
                let (tx, ty) = (&mut cv_buffer.train_x, &mut cv_buffer.train_y);

                tx.clear();
                ty.clear();
                for (j, &val) in y.iter().enumerate().take(i) {
                    let offset = j * dims;
                    tx.extend_from_slice(&x[offset..offset + dims]);
                    ty.push(val);
                }
                for (j, &val) in y.iter().enumerate().take(n).skip(i + 1) {
                    let offset = j * dims;
                    tx.extend_from_slice(&x[offset..offset + dims]);
                    ty.push(val);
                }

                let test_offset = i * dims;
                test_point.copy_from_slice(&x[test_offset..test_offset + dims]);

                let predicted = if let Some(ref mut p_fn) = predictor {
                    let preds = p_fn(tx, ty, &test_point, frac);
                    preds[0]
                } else {
                    let train_smooth = smoother(tx, ty, frac);
                    Self::interpolate_prediction(tx, &train_smooth, test_point[0])
                };

                let error = y[i] - predicted;
                total_error = total_error + error * error;
            }

            // Compute RMSE for this fraction
            cv_scores[frac_idx] = (total_error / T::from(n).unwrap()).sqrt();
        }

        Self::select_best_fraction(fractions, &cv_scores)
    }
}
