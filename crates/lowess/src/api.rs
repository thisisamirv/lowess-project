//! High-level API for LOWESS smoothing.
//!
//! This module provides the primary user-facing entry point for LOWESS. It
//! implements a fluent builder pattern for configuring regression parameters
//! and choosing an execution adapter (Batch, Streaming, or Online).

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::string::String;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::marker::PhantomData;
use num_traits::Float;

// Internal dependencies
use crate::adapters::batch::BatchLowessBuilder;
use crate::adapters::online::OnlineLowessBuilder;
use crate::adapters::streaming::StreamingLowessBuilder;
use crate::engine::executor::{CVPassFn, IntervalPassFn, SmoothPassFn};
use crate::evaluation::cv::CVKind;
use crate::evaluation::intervals::IntervalMethod;
use crate::parse::IntoEnum;
use crate::primitives::backend::Backend;

// Publicly re-exported types
pub use crate::adapters::online::UpdateMode;
pub use crate::adapters::streaming::MergeStrategy;
pub use crate::algorithms::regression::ZeroWeightFallback;
pub use crate::algorithms::robustness::RobustnessMethod;
pub use crate::engine::output::LowessResult;

pub use crate::math::boundary::BoundaryPolicy;
pub use crate::math::kernel::WeightFunction;
pub use crate::math::scaling::ScalingMethod;
pub use crate::primitives::errors::LowessError;

// Mode markers for the type-alias-based builder API.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchMode;

#[derive(Debug, Clone, Copy, Default)]
pub struct StreamingMode;

#[derive(Debug, Clone, Copy, Default)]
pub struct OnlineMode;

// Ergonomic entry-point type aliases.
pub type Lowess<T = f64> = LowessBuilder<T, BatchMode>;
pub type StreamingLowess<T = f64> = LowessBuilder<T, StreamingMode>;
pub type OnlineLowess<T = f64> = LowessBuilder<T, OnlineMode>;

// Marker types for selecting execution adapters.
#[allow(non_snake_case)]
pub mod Adapter {
    pub use super::{Batch, Online, Streaming};
}

// Fluent builder for configuring LOWESS parameters and execution modes.
#[derive(Debug, Clone)]
pub struct LowessBuilder<T, Mode = BatchMode> {
    // Smoothing fraction (0..1].
    pub fraction: Option<T>,

    // Robustness iterations.
    pub iterations: Option<usize>,

    // Threshold for skipping fitting (delta-optimization).
    pub delta: Option<T>,

    // Kernel weight function.
    pub weight_function: Option<WeightFunction>,

    // Outlier downweighting method.
    pub robustness_method: Option<RobustnessMethod>,

    // Scaling method for robust scale estimation (MAR/MAD/Mean).
    pub scaling_method: Option<ScalingMethod>,

    // interval estimation configuration.
    pub interval_type: Option<IntervalMethod<T>>,

    // Candidate bandwidths for cross-validation.
    pub cv_fractions: Option<Vec<T>>,

    // CV strategy (K-Fold/LOOCV).
    pub(crate) cv_kind: Option<CVKind>,

    // CV method string for string-based cross-validation API ("kfold" or "loocv").
    pub cv_method_str: Option<String>,

    // K value for K-fold CV (default: 5).
    pub cv_k_val: usize,

    // CV seed for reproducibility.
    pub(crate) cv_seed: Option<u64>,

    // Relative convergence tolerance.
    pub auto_converge: Option<T>,

    // Enable performance/statistical diagnostics.
    pub return_diagnostics: Option<bool>,

    // Return original residuals r_i.
    pub compute_residuals: Option<bool>,

    // Return final robustness weights w_i.
    pub return_robustness_weights: Option<bool>,

    // Policy for handling data boundaries (default: Extend).
    pub boundary_policy: Option<BoundaryPolicy>,

    // Behavior when local neighborhood weights are zero (default: UseLocalMean).
    pub zero_weight_fallback: Option<ZeroWeightFallback>,

    // Merging strategy for overlapping chunks (Streaming only).
    pub merge_strategy: Option<MergeStrategy>,

    // Incremental update mode (Online only).
    pub update_mode: Option<UpdateMode>,

    // Chunk size for streaming (Streaming only).
    pub chunk_size: Option<usize>,

    // Overlap size for streaming chunks (Streaming only).
    pub overlap: Option<usize>,

    // Window capacity for sliding window (Online only).
    pub window_capacity: Option<usize>,

    // Minimum points required for a valid fit (Online only).
    pub min_points: Option<usize>,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    // Custom smooth pass function.
    #[doc(hidden)]
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    // Custom cross-validation pass function.
    #[doc(hidden)]
    pub custom_cv_pass: Option<CVPassFn<T>>,

    // Custom interval estimation pass function.
    #[doc(hidden)]
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    // Execution backend hint.
    #[doc(hidden)]
    pub backend: Option<Backend>,

    // Parallel execution hint.
    #[doc(hidden)]
    pub parallel: Option<bool>,

    // Tracks if any parameter was set multiple times (for validation).
    #[doc(hidden)]
    pub duplicate_param: Option<&'static str>,

    // Accumulated parse errors from string-accepting builder methods.
    #[doc(hidden)]
    pub parse_errors: Vec<LowessError>,

    // Phantom mode marker (zero-sized; determines which build() variant to use).
    #[doc(hidden)]
    pub _mode: PhantomData<Mode>,
}

impl<T: Float, Mode> Default for LowessBuilder<T, Mode> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(private_bounds)]
impl<T: Float, Mode> LowessBuilder<T, Mode> {
    // Select an execution adapter to transition to an execution builder.
    pub fn adapter<A>(self, _adapter: A) -> A::Output
    where
        A: LowessAdapter<T>,
    {
        A::convert(self)
    }

    // Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            fraction: None,
            iterations: None,
            delta: None,
            weight_function: None,
            robustness_method: None,
            scaling_method: None,
            interval_type: None,
            cv_fractions: None,
            cv_kind: None,
            cv_method_str: None,
            cv_k_val: 5,
            cv_seed: None,
            auto_converge: None,
            return_diagnostics: None,
            compute_residuals: None,
            return_robustness_weights: None,
            boundary_policy: None,
            zero_weight_fallback: None,
            merge_strategy: None,
            update_mode: None,
            chunk_size: None,
            overlap: None,
            window_capacity: None,
            min_points: None,
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            backend: None,
            parallel: None,
            duplicate_param: None,
            parse_errors: Vec::new(),
            _mode: PhantomData,
        }
    }

    // Set behavior for handling zero-weight neighborhoods.
    pub fn zero_weight_fallback(mut self, policy: impl IntoEnum<ZeroWeightFallback>) -> Self {
        if self.zero_weight_fallback.is_some() {
            self.duplicate_param = Some("zero_weight_fallback");
        }
        match policy.into_enum() {
            Ok(p) => self.zero_weight_fallback = Some(p),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: impl IntoEnum<BoundaryPolicy>) -> Self {
        if self.boundary_policy.is_some() {
            self.duplicate_param = Some("boundary_policy");
        }
        match policy.into_enum() {
            Ok(p) => self.boundary_policy = Some(p),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the merging strategy for overlapping chunks (Streaming only).
    pub fn merge_strategy(mut self, strategy: impl IntoEnum<MergeStrategy>) -> Self {
        if self.merge_strategy.is_some() {
            self.duplicate_param = Some("merge_strategy");
        }
        match strategy.into_enum() {
            Ok(s) => self.merge_strategy = Some(s),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the incremental update mode (Online only).
    pub fn update_mode(mut self, mode: impl IntoEnum<UpdateMode>) -> Self {
        if self.update_mode.is_some() {
            self.duplicate_param = Some("update_mode");
        }
        match mode.into_enum() {
            Ok(m) => self.update_mode = Some(m),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the chunk size for streaming (Streaming only).
    pub fn chunk_size(mut self, size: usize) -> Self {
        if self.chunk_size.is_some() {
            self.duplicate_param = Some("chunk_size");
        }
        self.chunk_size = Some(size);
        self
    }

    // Set the overlap size for streaming chunks (Streaming only).
    pub fn overlap(mut self, overlap: usize) -> Self {
        if self.overlap.is_some() {
            self.duplicate_param = Some("overlap");
        }
        self.overlap = Some(overlap);
        self
    }

    // Set the window capacity for online processing (Online only).
    pub fn window_capacity(mut self, capacity: usize) -> Self {
        if self.window_capacity.is_some() {
            self.duplicate_param = Some("window_capacity");
        }
        self.window_capacity = Some(capacity);
        self
    }

    // Set the minimum points required for a valid fit (Online only).
    pub fn min_points(mut self, points: usize) -> Self {
        if self.min_points.is_some() {
            self.duplicate_param = Some("min_points");
        }
        self.min_points = Some(points);
        self
    }

    // Set the smoothing fraction (bandwidth alpha).
    pub fn fraction(mut self, fraction: T) -> Self {
        if self.fraction.is_some() {
            self.duplicate_param = Some("fraction");
        }
        self.fraction = Some(fraction);
        self
    }

    // Set the number of robustness iterations (typically 0-4).
    pub fn iterations(mut self, iterations: usize) -> Self {
        if self.iterations.is_some() {
            self.duplicate_param = Some("iterations");
        }
        self.iterations = Some(iterations);
        self
    }

    // Set the delta parameter for interpolation-based optimization.
    pub fn delta(mut self, delta: T) -> Self {
        if self.delta.is_some() {
            self.duplicate_param = Some("delta");
        }
        self.delta = Some(delta);
        self
    }

    // Set the kernel weight function.
    pub fn weight_function(mut self, wf: impl IntoEnum<WeightFunction>) -> Self {
        if self.weight_function.is_some() {
            self.duplicate_param = Some("weight_function");
        }
        match wf.into_enum() {
            Ok(w) => self.weight_function = Some(w),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the robustness weighting method.
    pub fn robustness_method(mut self, rm: impl IntoEnum<RobustnessMethod>) -> Self {
        if self.robustness_method.is_some() {
            self.duplicate_param = Some("robustness_method");
        }
        match rm.into_enum() {
            Ok(r) => self.robustness_method = Some(r),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the scaling method for robust scale estimation.
    pub fn scaling_method(mut self, sm: impl IntoEnum<ScalingMethod>) -> Self {
        if self.scaling_method.is_some() {
            self.duplicate_param = Some("scaling_method");
        }
        match sm.into_enum() {
            Ok(s) => self.scaling_method = Some(s),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Enable standard error computation.
    pub fn return_se(mut self) -> Self {
        if self.interval_type.is_none() {
            self.interval_type = Some(IntervalMethod::se());
        }
        self
    }

    // Enable confidence intervals at the specified level (e.g., 0.95).
    pub fn confidence_intervals(mut self, level: T) -> Self {
        if self.interval_type.as_ref().is_some_and(|it| it.confidence) {
            self.duplicate_param = Some("confidence_intervals");
        }
        self.interval_type = Some(match self.interval_type {
            Some(existing) if existing.prediction => IntervalMethod {
                level,
                confidence: true,
                prediction: true,
                se: true,
            },
            _ => IntervalMethod::confidence(level),
        });
        self
    }

    // Enable prediction intervals at the specified level.
    pub fn prediction_intervals(mut self, level: T) -> Self {
        if self.interval_type.as_ref().is_some_and(|it| it.prediction) {
            self.duplicate_param = Some("prediction_intervals");
        }
        self.interval_type = Some(match self.interval_type {
            Some(existing) if existing.confidence => IntervalMethod {
                level,
                confidence: true,
                prediction: true,
                se: true,
            },
            _ => IntervalMethod::prediction(level),
        });
        self
    }

    // Set the cross-validation method.
    //
    // Accepts case-insensitive strings: "kfold" (or "k_fold", "k-fold"), "loocv" (or "loo_cv", "loo-cv").
    pub fn cv_method(mut self, method: &str) -> Self {
        if self.cv_method_str.is_some() {
            self.duplicate_param = Some("cv_method");
        }
        self.cv_method_str = Some(method.to_string());
        self
    }

    // Set the number of folds for K-fold cross-validation (default: 5).
    pub fn cv_k(mut self, k: usize) -> Self {
        self.cv_k_val = k;
        self
    }

    // Set the candidate bandwidth fractions to evaluate during cross-validation.
    pub fn cv_fractions(mut self, fractions: Vec<T>) -> Self {
        if self.cv_fractions.is_some() {
            self.duplicate_param = Some("cv_fractions");
        }
        self.cv_fractions = Some(fractions);
        self
    }

    // Set the random seed for reproducible K-fold fold splitting.
    pub fn cv_seed(mut self, seed: u64) -> Self {
        self.cv_seed = Some(seed);
        self
    }

    // Enable automatic convergence detection based on relative change.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        if self.auto_converge.is_some() {
            self.duplicate_param = Some("auto_converge");
        }
        self.auto_converge = Some(tolerance);
        self
    }

    // Include statistical diagnostics (Metric, R², etc.) in output.
    pub fn return_diagnostics(mut self) -> Self {
        self.return_diagnostics = Some(true);
        self
    }

    // Include residuals in output.
    pub fn return_residuals(mut self) -> Self {
        self.compute_residuals = Some(true);
        self
    }

    // Include final robustness weights in output.
    pub fn return_robustness_weights(mut self) -> Self {
        self.return_robustness_weights = Some(true);
        self
    }

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++

    // Set a custom smooth pass function for execution (only for dev)
    #[doc(hidden)]
    pub fn custom_smooth_pass(mut self, pass: SmoothPassFn<T>) -> Self {
        self.custom_smooth_pass = Some(pass);
        self
    }

    // Set a custom cross-validation pass function (only for dev)
    #[doc(hidden)]
    pub fn custom_cv_pass(mut self, pass: CVPassFn<T>) -> Self {
        self.custom_cv_pass = Some(pass);
        self
    }

    // Set a custom interval estimation pass function (only for dev)
    #[doc(hidden)]
    pub fn custom_interval_pass(mut self, pass: IntervalPassFn<T>) -> Self {
        self.custom_interval_pass = Some(pass);
        self
    }

    // Set the execution backend hint (only for dev)
    #[doc(hidden)]
    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = Some(backend);
        self
    }

    // Set parallel execution hint (only for dev)
    #[doc(hidden)]
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = Some(parallel);
        self
    }
}

// Trait for transitioning from a generic builder to an execution builder.
pub trait LowessAdapter<T: Float> {
    // The output execution builder.
    type Output;

    // Convert a generic [`LowessBuilder`] into a specialized execution builder.
    fn convert<Mode>(builder: LowessBuilder<T, Mode>) -> Self::Output;
}

// Marker for in-memory batch processing.
#[derive(Debug, Clone, Copy)]
pub struct Batch;

impl<T: Float> LowessAdapter<T> for Batch {
    type Output = BatchLowessBuilder<T>;

    fn convert<Mode>(builder: LowessBuilder<T, Mode>) -> Self::Output {
        let mut result = BatchLowessBuilder::default();

        if let Some(fraction) = builder.fraction {
            result.fraction = fraction;
        }
        if let Some(iterations) = builder.iterations {
            result.iterations = iterations;
        }
        if let Some(delta) = builder.delta {
            result.delta = Some(delta);
        }
        if let Some(wf) = builder.weight_function {
            result.weight_function = wf;
        }
        if let Some(rm) = builder.robustness_method {
            result.robustness_method = rm;
        }
        if let Some(it) = builder.interval_type {
            result.interval_type = Some(it);
        }
        if let Some(cvf) = builder.cv_fractions {
            result.cv_fractions = Some(cvf);
        }
        if let Some(cvk) = builder.cv_kind {
            result.cv_kind = Some(cvk);
        }
        result.cv_seed = builder.cv_seed;
        // Convert string-based CV method (from cv_method()/cv_k() builder methods)
        if result.cv_kind.is_none()
            && let Some(method_str) = builder.cv_method_str
        {
            let lower = method_str.to_lowercase();
            match lower.as_str() {
                "kfold" | "k_fold" | "k-fold" => {
                    result.cv_kind = Some(CVKind::KFold(builder.cv_k_val));
                }
                "loocv" | "loo_cv" | "loo-cv" => {
                    result.cv_kind = Some(CVKind::LOOCV);
                }
                _ => {
                    result.deferred_error = Some(LowessError::InvalidOption {
                        option: "cv_method",
                        value: method_str,
                        valid: "kfold, loocv",
                    });
                }
            }
        }
        if let Some(ac) = builder.auto_converge {
            result.auto_converge = Some(ac);
        }
        if let Some(zwf) = builder.zero_weight_fallback {
            result.zero_weight_fallback = zwf;
        }
        if let Some(bp) = builder.boundary_policy {
            result.boundary_policy = bp;
        }
        if let Some(sm) = builder.scaling_method {
            result.scaling_method = sm;
        }

        if let Some(rw) = builder.return_robustness_weights {
            result.return_robustness_weights = rw;
        }
        if let Some(rd) = builder.return_diagnostics {
            result.return_diagnostics = rd;
        }
        if let Some(cr) = builder.compute_residuals {
            result.compute_residuals = cr;
        }

        // ++++++++++++++++++++++++++++++++++++++
        // +               DEV                  +
        // ++++++++++++++++++++++++++++++++++++++
        if let Some(sp) = builder.custom_smooth_pass {
            result.custom_smooth_pass = Some(sp);
        }
        if let Some(cp) = builder.custom_cv_pass {
            result.custom_cv_pass = Some(cp);
        }
        if let Some(ip) = builder.custom_interval_pass {
            result.custom_interval_pass = Some(ip);
        }
        if let Some(b) = builder.backend {
            result.backend = Some(b);
        }
        if let Some(p) = builder.parallel {
            result.parallel = Some(p);
        }

        result.duplicate_param = builder.duplicate_param;
        if result.deferred_error.is_none() && !builder.parse_errors.is_empty() {
            result.deferred_error = Some(LowessError::ParseErrors(builder.parse_errors));
        }

        result
    }
}

// Marker for chunked streaming processing.
#[derive(Debug, Clone, Copy)]
pub struct Streaming;

impl<T: Float> LowessAdapter<T> for Streaming {
    type Output = StreamingLowessBuilder<T>;

    fn convert<Mode>(builder: LowessBuilder<T, Mode>) -> Self::Output {
        let mut result = StreamingLowessBuilder::default();

        // Override with user-provided values
        if let Some(chunk_size) = builder.chunk_size {
            result.chunk_size = chunk_size;
        }
        if let Some(overlap) = builder.overlap {
            result.overlap = overlap;
        }
        if let Some(fraction) = builder.fraction {
            result.fraction = fraction;
        }
        if let Some(iterations) = builder.iterations {
            result.iterations = iterations;
        }
        if let Some(delta) = builder.delta {
            result.delta = delta;
        }
        if let Some(wf) = builder.weight_function {
            result.weight_function = wf;
        }
        if let Some(bp) = builder.boundary_policy {
            result.boundary_policy = bp;
        }
        if let Some(rm) = builder.robustness_method {
            result.robustness_method = rm;
        }
        if let Some(zwf) = builder.zero_weight_fallback {
            result.zero_weight_fallback = zwf;
        }
        if let Some(ms) = builder.merge_strategy {
            result.merge_strategy = ms;
        }
        if let Some(sm) = builder.scaling_method {
            result.scaling_method = sm;
        }

        if let Some(rw) = builder.return_robustness_weights {
            result.return_robustness_weights = rw;
        }
        if let Some(rd) = builder.return_diagnostics {
            result.return_diagnostics = rd;
        }
        if let Some(cr) = builder.compute_residuals {
            result.compute_residuals = cr;
        }
        if let Some(ac) = builder.auto_converge {
            result.auto_converge = Some(ac);
        }

        // ++++++++++++++++++++++++++++++++++++++
        // +               DEV                  +
        // ++++++++++++++++++++++++++++++++++++++

        if let Some(sp) = builder.custom_smooth_pass {
            result.custom_smooth_pass = Some(sp);
        }
        if let Some(cp) = builder.custom_cv_pass {
            result.custom_cv_pass = Some(cp);
        }
        if let Some(ip) = builder.custom_interval_pass {
            result.custom_interval_pass = Some(ip);
        }
        if let Some(b) = builder.backend {
            result.backend = Some(b);
        }
        if let Some(p) = builder.parallel {
            result.parallel = Some(p);
        }
        result.duplicate_param = builder.duplicate_param;
        if result.deferred_error.is_none() && !builder.parse_errors.is_empty() {
            result.deferred_error = Some(LowessError::ParseErrors(builder.parse_errors));
        }

        result
    }
}

// Marker for incremental online processing.
#[derive(Debug, Clone, Copy)]
pub struct Online;

// Mode-specific build() methods — each delegates to the corresponding adapter.
impl<T: Float> LowessBuilder<T, BatchMode> {
    pub fn build(self) -> Result<crate::adapters::batch::BatchLowess<T>, LowessError> {
        Batch::convert(self).build()
    }
}

impl<T: Float> LowessBuilder<T, StreamingMode> {
    pub fn build(self) -> Result<crate::adapters::streaming::StreamingLowess<T>, LowessError> {
        Streaming::convert(self).build()
    }
}

impl<T: Float> LowessBuilder<T, OnlineMode> {
    pub fn build(self) -> Result<crate::adapters::online::OnlineLowess<T>, LowessError> {
        Online::convert(self).build()
    }
}

impl<T: Float> LowessAdapter<T> for Online {
    type Output = OnlineLowessBuilder<T>;

    fn convert<Mode>(builder: LowessBuilder<T, Mode>) -> Self::Output {
        let mut result = OnlineLowessBuilder::default();

        // Override with user-provided values
        if let Some(window_capacity) = builder.window_capacity {
            result.window_capacity = window_capacity;
        }
        if let Some(min_points) = builder.min_points {
            result.min_points = min_points;
        }
        if let Some(fraction) = builder.fraction {
            result.fraction = fraction;
        }
        if let Some(iterations) = builder.iterations {
            result.iterations = iterations;
        }
        if let Some(delta) = builder.delta {
            result.delta = delta;
        }
        if let Some(wf) = builder.weight_function {
            result.weight_function = wf;
        }
        if let Some(um) = builder.update_mode {
            result.update_mode = um;
        }
        if let Some(rm) = builder.robustness_method {
            result.robustness_method = rm;
        }
        if let Some(bp) = builder.boundary_policy {
            result.boundary_policy = bp;
        }
        if let Some(zwf) = builder.zero_weight_fallback {
            result.zero_weight_fallback = zwf;
        }
        if let Some(sm) = builder.scaling_method {
            result.scaling_method = sm;
        }

        if let Some(cr) = builder.compute_residuals {
            result.compute_residuals = cr;
        }
        if let Some(rw) = builder.return_robustness_weights {
            result.return_robustness_weights = rw;
        }
        if let Some(ac) = builder.auto_converge {
            result.auto_converge = Some(ac);
        }

        // ++++++++++++++++++++++++++++++++++++++
        // +               DEV                  +
        // ++++++++++++++++++++++++++++++++++++++

        if let Some(sp) = builder.custom_smooth_pass {
            result.custom_smooth_pass = Some(sp);
        }
        if let Some(cp) = builder.custom_cv_pass {
            result.custom_cv_pass = Some(cp);
        }
        if let Some(ip) = builder.custom_interval_pass {
            result.custom_interval_pass = Some(ip);
        }
        if let Some(b) = builder.backend {
            result.backend = Some(b);
        }
        if let Some(p) = builder.parallel {
            result.parallel = Some(p);
        }
        result.duplicate_param = builder.duplicate_param;
        if result.deferred_error.is_none() && !builder.parse_errors.is_empty() {
            result.deferred_error = Some(LowessError::ParseErrors(builder.parse_errors));
        }

        result
    }
}
