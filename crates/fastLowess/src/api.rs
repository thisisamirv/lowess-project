//! High-level API for LOWESS smoothing with parallel execution support.
//!
//! This module provides the primary user-facing entry point for LOWESS with
//! heavy-duty parallel execution capabilities. It extends the `lowess` API
//! with adapters that utilize all available CPU cores or GPU hardware.

// Internal dependencies
use crate::adapters::batch::{ParallelBatchLowess, ParallelBatchLowessBuilder};
use crate::adapters::online::{ParallelOnlineLowess, ParallelOnlineLowessBuilder};
use crate::adapters::streaming::{ParallelStreamingLowess, ParallelStreamingLowessBuilder};

// External dependencies
use num_traits::Float;

// Import base marker types for delegation
use lowess::internals::api::Batch as BaseBatch;
use lowess::internals::api::Online as BaseOnline;
use lowess::internals::api::Streaming as BaseStreaming;

// Publicly re-exported types
pub use lowess::internals::api::{LowessAdapter, LowessBuilder};
pub use lowess::internals::engine::output::LowessResult;
pub use lowess::internals::primitives::backend::Backend;
pub use lowess::internals::primitives::errors::LowessError;

// Marker for parallel in-memory batch processing.
#[derive(Debug, Clone, Copy)]
pub struct Batch;

impl<T: Float> LowessAdapter<T> for Batch {
    type Output = ParallelBatchLowessBuilder<T>;

    fn convert<Mode>(builder: LowessBuilder<T, Mode>) -> Self::Output {
        // Determine parallel mode: user choice OR default to true for fastLowess Batch
        let parallel = builder.parallel.unwrap_or(true);

        // Delegate to base implementation to create base builder
        let mut base = <BaseBatch as LowessAdapter<T>>::convert(builder);
        base.parallel = Some(parallel);

        // Wrap with extension fields
        ParallelBatchLowessBuilder {
            base,
            parse_errors: Vec::new(),
            cv_method_str: None,
            cv_k_val: 5,
        }
    }
}

// Marker for parallel chunked streaming processing.
#[derive(Debug, Clone, Copy)]
pub struct Streaming;

impl<T: Float> LowessAdapter<T> for Streaming {
    type Output = ParallelStreamingLowessBuilder<T>;

    fn convert<Mode>(builder: LowessBuilder<T, Mode>) -> Self::Output {
        // Determine parallel mode: user choice OR default to true for fastLowess Streaming
        let parallel = builder.parallel.unwrap_or(true);

        // Delegate to base implementation to create base builder
        let mut base = <BaseStreaming as LowessAdapter<T>>::convert(builder);
        base.parallel = Some(parallel);

        // Wrap with extension fields
        ParallelStreamingLowessBuilder {
            base,
            parse_errors: Vec::new(),
        }
    }
}

// Marker for incremental online processing with parallel support.
#[derive(Debug, Clone, Copy)]
pub struct Online;

impl<T: Float> LowessAdapter<T> for Online {
    type Output = ParallelOnlineLowessBuilder<T>;

    fn convert<Mode>(builder: LowessBuilder<T, Mode>) -> Self::Output {
        // Determine parallel mode: user choice OR default to false for fastLowess Online
        let parallel = builder.parallel.unwrap_or(false);

        // Delegate to base implementation to create base builder
        let mut base = <BaseOnline as LowessAdapter<T>>::convert(builder);
        base.parallel = Some(parallel);

        // Wrap with extension fields
        ParallelOnlineLowessBuilder {
            base,
            parse_errors: Vec::new(),
        }
    }
}

// Entry-point wrapper types: Lowess / StreamingLowess / OnlineLowess
// Mirror the bindings API while defaulting to parallel execution.

// Macro: generate method-forwarding impls common to all three entry-point types.
macro_rules! impl_common_builder {
    ($t:ty) => {
        impl Default for $t {
            fn default() -> Self {
                Self::new()
            }
        }
        impl $t {
            pub fn new() -> Self {
                Self(LowessBuilder::new())
            }
            // string enum options
            pub fn weight_function(mut self, s: &str) -> Self {
                self.0 = self.0.weight_function(s);
                self
            }
            pub fn robustness_method(mut self, s: &str) -> Self {
                self.0 = self.0.robustness_method(s);
                self
            }
            pub fn scaling_method(mut self, s: &str) -> Self {
                self.0 = self.0.scaling_method(s);
                self
            }
            pub fn zero_weight_fallback(mut self, s: &str) -> Self {
                self.0 = self.0.zero_weight_fallback(s);
                self
            }
            pub fn boundary_policy(mut self, s: &str) -> Self {
                self.0 = self.0.boundary_policy(s);
                self
            }
            // numeric options
            pub fn fraction(mut self, f: f64) -> Self {
                self.0 = self.0.fraction(f);
                self
            }
            pub fn iterations(mut self, i: usize) -> Self {
                self.0 = self.0.iterations(i);
                self
            }
            pub fn delta(mut self, d: f64) -> Self {
                self.0 = self.0.delta(d);
                self
            }
            pub fn confidence_intervals(mut self, level: f64) -> Self {
                self.0 = self.0.confidence_intervals(level);
                self
            }
            pub fn prediction_intervals(mut self, level: f64) -> Self {
                self.0 = self.0.prediction_intervals(level);
                self
            }
            pub fn auto_converge(mut self, tol: f64) -> Self {
                self.0 = self.0.auto_converge(tol);
                self
            }
            // flag options (no argument)
            pub fn return_se(mut self) -> Self {
                self.0 = self.0.return_se();
                self
            }
            pub fn return_diagnostics(mut self) -> Self {
                self.0 = self.0.return_diagnostics();
                self
            }
            pub fn return_residuals(mut self) -> Self {
                self.0 = self.0.return_residuals();
                self
            }
            pub fn return_robustness_weights(mut self) -> Self {
                self.0 = self.0.return_robustness_weights();
                self
            }
            // dev options
            #[doc(hidden)]
            pub fn parallel(mut self, p: bool) -> Self {
                self.0 = self.0.parallel(p);
                self
            }
            #[doc(hidden)]
            pub fn backend(mut self, b: Backend) -> Self {
                self.0 = self.0.backend(b);
                self
            }
        }
    };
}

// Parallel batch LOWESS entry point.
pub struct Lowess(LowessBuilder<f64>);
impl_common_builder!(Lowess);
impl Lowess {
    pub fn custom_weights(mut self, w: Vec<f64>) -> Self {
        self.0 = self.0.custom_weights(w);
        self
    }
    pub fn cv_method(mut self, m: &str) -> Self {
        self.0 = self.0.cv_method(m);
        self
    }
    pub fn cv_k(mut self, k: usize) -> Self {
        self.0 = self.0.cv_k(k);
        self
    }
    pub fn cv_fractions(mut self, f: Vec<f64>) -> Self {
        self.0 = self.0.cv_fractions(f);
        self
    }
    pub fn cv_seed(mut self, s: u64) -> Self {
        self.0 = self.0.cv_seed(s);
        self
    }

    pub fn build(self) -> Result<ParallelBatchLowess<f64>, LowessError> {
        Batch::convert(self.0).build()
    }
}

// Parallel streaming LOWESS entry point.
pub struct StreamingLowess(LowessBuilder<f64>);
impl_common_builder!(StreamingLowess);
impl StreamingLowess {
    pub fn chunk_size(mut self, s: usize) -> Self {
        self.0 = self.0.chunk_size(s);
        self
    }
    pub fn overlap(mut self, o: usize) -> Self {
        self.0 = self.0.overlap(o);
        self
    }
    pub fn merge_strategy(mut self, s: &str) -> Self {
        self.0 = self.0.merge_strategy(s);
        self
    }

    pub fn build(self) -> Result<ParallelStreamingLowess<f64>, LowessError> {
        Streaming::convert(self.0).build()
    }
}

// Parallel online LOWESS entry point.
pub struct OnlineLowess(LowessBuilder<f64>);
impl_common_builder!(OnlineLowess);
impl OnlineLowess {
    pub fn window_capacity(mut self, c: usize) -> Self {
        self.0 = self.0.window_capacity(c);
        self
    }
    pub fn min_points(mut self, m: usize) -> Self {
        self.0 = self.0.min_points(m);
        self
    }
    pub fn update_mode(mut self, s: &str) -> Self {
        self.0 = self.0.update_mode(s);
        self
    }

    pub fn build(self) -> Result<ParallelOnlineLowess<f64>, LowessError> {
        Online::convert(self.0).build()
    }
}
