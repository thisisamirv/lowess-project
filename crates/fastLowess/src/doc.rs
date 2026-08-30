//! User guide and API reference for the `fastLowess` crate.
//!
//! Browse the submodules below for conceptual guides and worked examples.
//! For the API itself, see [`prelude`](crate::prelude).

/// Conceptual introduction: what LOWESS is and how it works.
#[cfg(doc)]
pub mod concepts {
    #![doc = include_str!("../docs/concepts.md")]
}

/// Installation instructions across build configurations.
#[cfg(doc)]
pub mod installation {
    #![doc = include_str!("../docs/installation.md")]
}

/// Get up and running in minutes.
#[cfg(doc)]
pub mod quickstart {
    #![doc = include_str!("../docs/quickstart.md")]
}

/// Complete parameter reference for all configuration options.
#[cfg(doc)]
pub mod parameters {
    #![doc = include_str!("../docs/parameters.md")]
}

/// Choosing between Batch, Streaming, and Online execution modes.
#[cfg(doc)]
pub mod adapter_choice {
    #![doc = include_str!("../docs/adapter-choice.md")]
}

/// Confidence and prediction intervals.
#[cfg(doc)]
pub mod intervals {
    #![doc = include_str!("../docs/intervals.md")]
}

/// Automated fraction selection via cross-validation.
#[cfg(doc)]
pub mod cross_validation {
    #![doc = include_str!("../docs/cross-validation.md")]
}

/// Distance-weighting kernel functions.
#[cfg(doc)]
pub mod kernels {
    #![doc = include_str!("../docs/kernels.md")]
}

/// Robustness iterations and outlier downweighting.
#[cfg(doc)]
pub mod robustness {
    #![doc = include_str!("../docs/robustness.md")]
}

/// Residual scaling methods (MAD, MAR, Mean).
#[cfg(doc)]
pub mod scaling {
    #![doc = include_str!("../docs/scaling.md")]
}

/// Per-observation custom weights.
#[cfg(doc)]
pub mod custom_weights {
    #![doc = include_str!("../docs/custom-weights.md")]
}

/// Boundary handling strategies for edge bias reduction.
#[cfg(doc)]
pub mod boundary {
    #![doc = include_str!("../docs/boundary.md")]
}

/// Merge strategies for Streaming mode overlap regions.
#[cfg(doc)]
pub mod merge {
    #![doc = include_str!("../docs/merge.md")]
}

/// GPU-accelerated backend via wgpu.
#[cfg(all(doc, feature = "gpu"))]
pub mod gpu_backend {
    #![doc = include_str!("../docs/gpu-backend.md")]
}

/// Application examples: use cases and real-world patterns.
#[cfg(doc)]
pub mod use_cases {
    /// Smoothing genomic data: methylation, ChIP-seq, coverage.
    pub mod genomics {
        #![doc = include_str!("../docs/use-case-genomics.md")]
    }
    /// Time series smoothing and trend extraction.
    pub mod time_series {
        #![doc = include_str!("../docs/use-case-time-series.md")]
    }
    /// Real-time sensor and streaming data processing.
    pub mod real_time {
        #![doc = include_str!("../docs/use-case-real-time.md")]
    }
}

/// Full Rust API reference: structs, builder methods, and result types.
#[cfg(doc)]
pub mod api {
    #![doc = include_str!("../docs/api.md")]
    /// `StreamingLowess` API reference.
    pub mod streaming {
        #![doc = include_str!("../docs/api-streaming.md")]
    }
    /// `OnlineLowess` API reference.
    pub mod online {
        #![doc = include_str!("../docs/api-online.md")]
    }
}

/// Release notes and changelog for this crate.
#[cfg(doc)]
pub mod news {
    #![doc = include_str!("../docs/news.md")]
}
