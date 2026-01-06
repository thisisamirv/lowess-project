//! High-level API for LOWESS smoothing with parallel execution support.
//!
//! ## Purpose
//!
//! This module provides the primary user-facing entry point for LOWESS with
//! heavy-duty parallel execution capabilities. It extends the `lowess` API
//! with adapters that utilize all available CPU cores or GPU hardware.
//!
//! ## Design notes
//!
//! * **Fluent Integration**: Re-uses the base `lowess` builder pattern.
//! * **Parallel-First**: Defaults to parallel execution where beneficial.
//! * **Transparent**: Marker types (Batch, Streaming, Online) select the parallel builders.
//!
//! ## Key concepts
//!
//! * **Parallel Support**: Uses `rayon` (CPU) or `wgpu` (GPU) for acceleration.
//! * **Extended Adapters**: Wraps core adapters with parallel implementation logic.
//! * **Feature-Gated**: Parallelism is configurable via crate features.
//!
//! ### Configuration Flow
//!
//! 1. Create a [`LowessBuilder`] via `Lowess::new()`.
//! 2. Chain configuration methods (`.fraction()`, `.iterations()`, etc.).
//! 3. Select an adapter via `.adapter(Batch)` to get a parallel execution builder.

// Feature-gated imports
#[cfg(feature = "cpu")]
use crate::adapters::batch::ParallelBatchLowessBuilder;
#[cfg(feature = "cpu")]
use crate::adapters::online::ParallelOnlineLowessBuilder;
#[cfg(feature = "cpu")]
use crate::adapters::streaming::ParallelStreamingLowessBuilder;

// External dependencies
use num_traits::Float;

// Import base marker types for delegation
use lowess::internals::api::Batch as BaseBatch;
use lowess::internals::api::Online as BaseOnline;
use lowess::internals::api::Streaming as BaseStreaming;

// Publicly re-exported types
pub use lowess::internals::adapters::online::UpdateMode;
pub use lowess::internals::adapters::streaming::MergeStrategy;
pub use lowess::internals::algorithms::regression::ZeroWeightFallback;
pub use lowess::internals::algorithms::robustness::RobustnessMethod;
pub use lowess::internals::api::{LowessAdapter, LowessBuilder};
pub use lowess::internals::engine::output::LowessResult;
pub use lowess::internals::evaluation::cv::{KFold, LOOCV};
pub use lowess::internals::math::boundary::BoundaryPolicy;
pub use lowess::internals::math::kernel::WeightFunction;
pub use lowess::internals::math::scaling::ScalingMethod;
pub use lowess::internals::primitives::backend::Backend;
pub use lowess::internals::primitives::errors::LowessError;

// ============================================================================
// Adapter Module
// ============================================================================

/// Adapter selection namespace.
#[allow(non_snake_case)]
pub mod Adapter {
    pub use super::{Batch, Online, Streaming};
}

// ============================================================================
// Adapter Marker Types
// ============================================================================

/// Marker for parallel in-memory batch processing.
#[derive(Debug, Clone, Copy)]
pub struct Batch;

impl<T: Float> LowessAdapter<T> for Batch {
    type Output = ParallelBatchLowessBuilder<T>;

    fn convert(builder: LowessBuilder<T>) -> Self::Output {
        // Determine parallel mode: user choice OR default to true for fastLowess Batch
        let parallel = builder.parallel.unwrap_or(true);

        // Delegate to base implementation to create base builder
        let mut base = <BaseBatch as LowessAdapter<T>>::convert(builder);
        base = base.parallel(parallel);

        // Wrap with extension fields
        ParallelBatchLowessBuilder { base }
    }
}

/// Marker for parallel chunked streaming processing.
#[derive(Debug, Clone, Copy)]
pub struct Streaming;

impl<T: Float> LowessAdapter<T> for Streaming {
    type Output = ParallelStreamingLowessBuilder<T>;

    fn convert(builder: LowessBuilder<T>) -> Self::Output {
        // Determine parallel mode: user choice OR default to true for fastLowess Streaming
        let parallel = builder.parallel.unwrap_or(true);

        // Delegate to base implementation to create base builder
        let mut base = <BaseStreaming as LowessAdapter<T>>::convert(builder);
        base = base.parallel(parallel);

        // Wrap with extension fields
        ParallelStreamingLowessBuilder { base }
    }
}

/// Marker for incremental online processing with parallel support.
#[derive(Debug, Clone, Copy)]
pub struct Online;

impl<T: Float> LowessAdapter<T> for Online {
    type Output = ParallelOnlineLowessBuilder<T>;

    fn convert(builder: LowessBuilder<T>) -> Self::Output {
        // Determine parallel mode: user choice OR default to false for fastLowess Online
        let parallel = builder.parallel.unwrap_or(false);

        // Delegate to base implementation to create base builder
        let mut base = <BaseOnline as LowessAdapter<T>>::convert(builder);
        base = base.parallel(parallel);

        // Wrap with extension fields
        ParallelOnlineLowessBuilder { base }
    }
}
