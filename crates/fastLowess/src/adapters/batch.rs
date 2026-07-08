//! Batch adapter for standard LOWESS smoothing.
//!
//! This module provides the batch execution adapter for LOWESS smoothing.
//! It handles complete datasets in memory with optional parallel processing,
//! making it suitable for small to medium-sized datasets.
//!
//! ## srrstats Compliance
//!
//! @srrstats {G3.0} Rayon-based parallel execution for CPU-bound workloads.
//! @srrstats {G1.5} GPU backend support via feature flag for accelerated fits.

// Internal dependencies
#[cfg(feature = "cpu")]
use crate::engine::executor::smooth_pass_parallel;
#[cfg(feature = "gpu")]
use crate::engine::gpu::{cross_validate_gpu, fit_pass_gpu};
#[cfg(feature = "cpu")]
use crate::evaluation::cv::cv_pass_parallel;
#[cfg(feature = "cpu")]
use crate::evaluation::intervals::interval_pass_parallel;
use crate::input::LowessInput;

// External dependencies
use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// Export dependencies from lowess crate
use crate::parse::IntoEnum;
use lowess::internals::adapters::batch::BatchLowessBuilder;
use lowess::internals::algorithms::regression::WLSSolver;
use lowess::internals::algorithms::regression::ZeroWeightFallback;
use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::engine::output::LowessResult;
use lowess::internals::evaluation::cv::CVKind;
use lowess::internals::evaluation::intervals::IntervalMethod;
use lowess::internals::math::boundary::BoundaryPolicy;
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::primitives::backend::Backend;
use lowess::internals::primitives::errors::LowessError;

// Builder for batch LOWESS processor with parallel support.
#[derive(Debug, Clone)]
pub struct ParallelBatchLowessBuilder<T: Float> {
    // Base builder from the lowess crate
    pub base: BatchLowessBuilder<T>,
    // CV method string for string-based cross-validation API ("kfold" or "loocv").
    pub cv_method_str: Option<String>,
    // K value for K-fold CV (default: 5).
    pub cv_k_val: usize,
}

impl<T: Float> Default for ParallelBatchLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(private_bounds)]
impl<T: Float> ParallelBatchLowessBuilder<T> {
    // Create a new batch LOWESS builder with default parameters.
    fn new() -> Self {
        let mut base = BatchLowessBuilder::default();
        base.parallel = Some(true); // Default to parallel in fastLowess
        Self {
            base,
            cv_method_str: None,
            cv_k_val: 5,
        }
    }

    // Set parallel execution mode.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.base.parallel = Some(parallel);
        self
    }

    // Set the execution backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.base.backend = Some(backend);
        self
    }

    // Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.base.fraction = fraction;
        self
    }

    // Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base.iterations = iterations;
        self
    }

    // Set the delta parameter for interpolation optimization.
    pub fn delta(mut self, delta: T) -> Self {
        self.base.delta = Some(delta);
        self
    }

    // Set the kernel weight function.
    pub fn weight_function(mut self, wf: impl IntoEnum<WeightFunction>) -> Self {
        match wf.into_enum() {
            Ok(w) => self.base.weight_function = w,
            Err(e) => {
                if self.base.deferred_error.is_none() {
                    self.base.deferred_error = Some(e);
                }
            }
        }
        self
    }

    // Set the robustness method for outlier handling.
    pub fn robustness_method(mut self, method: impl IntoEnum<RobustnessMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.robustness_method = m,
            Err(e) => {
                if self.base.deferred_error.is_none() {
                    self.base.deferred_error = Some(e);
                }
            }
        }
        self
    }

    // Set the zero-weight fallback policy.
    pub fn zero_weight_fallback(mut self, fallback: impl IntoEnum<ZeroWeightFallback>) -> Self {
        match fallback.into_enum() {
            Ok(f) => self.base.zero_weight_fallback = f,
            Err(e) => {
                if self.base.deferred_error.is_none() {
                    self.base.deferred_error = Some(e);
                }
            }
        }
        self
    }

    // Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: impl IntoEnum<BoundaryPolicy>) -> Self {
        match policy.into_enum() {
            Ok(p) => self.base.boundary_policy = p,
            Err(e) => {
                if self.base.deferred_error.is_none() {
                    self.base.deferred_error = Some(e);
                }
            }
        }
        self
    }

    // Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    // Enable returning residuals in the output.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base.compute_residuals = enabled;
        self
    }

    // Enable returning robustness weights in the result.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base.return_robustness_weights = enabled;
        self
    }

    // Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base.return_diagnostics = enabled;
        self
    }

    // Enable confidence intervals at the specified level.
    pub fn confidence_intervals(mut self, level: T) -> Self {
        self.base.interval_type = Some(IntervalMethod::confidence(level));
        self
    }

    // Enable prediction intervals at the specified level.
    pub fn prediction_intervals(mut self, level: T) -> Self {
        self.base.interval_type = Some(IntervalMethod::prediction(level));
        self
    }

    // Enable cross-validation with the specified candidate fractions.
    pub fn cv_fractions(mut self, fractions: Vec<T>) -> Self {
        self.base.cv_fractions = Some(fractions);
        self
    }

    // Set the cross-validation method.
    //
    // Accepts case-insensitive strings: "kfold" (or "k_fold", "k-fold"), "loocv" (or "loo_cv", "loo-cv").
    pub fn cv_method(mut self, method: &str) -> Self {
        self.cv_method_str = Some(method.to_string());
        self
    }

    // Set the number of folds for K-fold cross-validation (default: 5).
    pub fn cv_k(mut self, k: usize) -> Self {
        self.cv_k_val = k;
        self
    }

    // Set the random seed for reproducible K-fold fold splitting.
    pub fn cv_seed(mut self, seed: u64) -> Self {
        self.base.cv_seed = Some(seed);
        self
    }

    // Set per-observation case weights applied as `w_ij = custom_weights[j] * K(d_ij / h)`.
    // Must have the same length as the input data and all values must be finite and non-negative.
    pub fn custom_weights(mut self, weights: Vec<T>) -> Self {
        self.base.custom_weights = Some(weights);
        self
    }

    // Build the batch processor.
    pub fn build(mut self) -> Result<ParallelBatchLowess<T>, LowessError> {
        // Resolve string-based CV method
        if self.base.cv_kind.is_none() {
            if let Some(method_str) = self.cv_method_str.take() {
                let lower = method_str.to_lowercase();
                match lower.as_str() {
                    "kfold" | "k_fold" | "k-fold" => {
                        self.base.cv_kind = Some(CVKind::KFold(self.cv_k_val));
                    }
                    "loocv" | "loo_cv" | "loo-cv" => {
                        self.base.cv_kind = Some(CVKind::LOOCV);
                    }
                    _ => {
                        if self.base.deferred_error.is_none() {
                            self.base.deferred_error = Some(LowessError::InvalidOption {
                                option: "cv_method",
                                value: method_str,
                                valid: "kfold, loocv",
                            });
                        }
                    }
                }
            }
        }

        // Check for deferred errors
        if let Some(ref err) = self.base.deferred_error {
            return Err(err.clone());
        }

        // Validate by attempting to build the base processor
        // This reuses the validation logic centralized in the lowess crate
        let _ = self.base.clone().build()?;

        Ok(ParallelBatchLowess { config: self })
    }
}

// Batch LOWESS processor with parallel support.
pub struct ParallelBatchLowess<T: Float> {
    config: ParallelBatchLowessBuilder<T>,
}
#[allow(private_bounds)]
impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> ParallelBatchLowess<T> {
    // Perform LOWESS smoothing on the provided data.
    pub fn fit<I1, I2>(self, x: &I1, y: &I2) -> Result<LowessResult<T>, LowessError>
    where
        I1: LowessInput<T> + ?Sized,
        I2: LowessInput<T> + ?Sized,
    {
        let x_slice = x.as_lowess_slice()?;
        let y_slice = y.as_lowess_slice()?;

        // Configure the base builder with parallel callback if enabled
        let mut builder = self.config.base;

        match builder.backend.unwrap_or(Backend::CPU) {
            Backend::CPU => {
                #[cfg(feature = "cpu")]
                {
                    if builder.parallel.unwrap_or(true) {
                        builder.custom_smooth_pass = Some(smooth_pass_parallel);
                        builder.custom_cv_pass = Some(cv_pass_parallel);
                        builder.custom_interval_pass = Some(interval_pass_parallel);
                    } else {
                        // Resets - though they are None by default
                        // but explicitly clearing just in case
                        builder.custom_smooth_pass = None;
                        builder.custom_cv_pass = None;
                        builder.custom_interval_pass = None;
                    }
                }
                #[cfg(not(feature = "cpu"))]
                {
                    // Fallback to sequential if cpu feature is disabled
                    builder.custom_smooth_pass = None;
                    builder.custom_cv_pass = None;
                    builder.custom_interval_pass = None;
                }
            }
            Backend::GPU => {
                #[cfg(feature = "gpu")]
                {
                    builder.custom_fit_pass = Some(fit_pass_gpu);
                    builder.custom_cv_pass = Some(cross_validate_gpu);
                    builder.delegate_boundary_handling = true;
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(LowessError::UnsupportedFeature {
                        adapter: "Batch",
                        feature: "GPU backend (requires 'gpu' feature)",
                    });
                }
            }
        }

        // Delegate execution to the base implementation
        let processor = builder.build()?;
        processor.fit(x_slice, y_slice)
    }
}
