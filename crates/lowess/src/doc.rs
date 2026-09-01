//! User guide and API reference for the `lowess` crate.
//!
//! Browse the submodules below for conceptual guides and worked examples.
//! For the API itself, see [`prelude`](crate::prelude).

/// Getting started: concepts, installation, and a quick tour.
#[cfg(doc)]
pub mod introduction {
    /// Conceptual introduction: what LOWESS is and how it works.
    pub mod concepts {
        #![doc = include_str!("../docs/introduction/concepts.md")]
    }
    /// Installation instructions across build configurations.
    pub mod installation {
        #![doc = include_str!("../docs/introduction/installation.md")]
    }
    /// Get up and running in minutes.
    pub mod quickstart {
        #![doc = include_str!("../docs/introduction/quickstart.md")]
    }
}

/// Adapter selection, uncertainty quantification, and parameter tuning.
#[cfg(doc)]
pub mod guide {
    /// Choosing between Batch, Streaming, and Online execution modes.
    pub mod adapter_choice {
        #![doc = include_str!("../docs/guide/adapter-choice.md")]
    }
    /// Confidence and prediction intervals.
    pub mod intervals {
        #![doc = include_str!("../docs/guide/intervals.md")]
    }
    /// Automated fraction selection via cross-validation.
    pub mod cross_validation {
        #![doc = include_str!("../docs/guide/cross-validation.md")]
    }
}

/// Weight functions, outlier handling, and residual scaling.
#[cfg(doc)]
pub mod weighting {
    /// Distance-weighting kernel functions.
    pub mod kernels {
        #![doc = include_str!("../docs/weighting/kernels.md")]
    }
    /// Robustness iterations and outlier downweighting.
    pub mod robustness {
        #![doc = include_str!("../docs/weighting/robustness.md")]
    }
    /// Residual scaling methods (MAD, MAR, Mean).
    pub mod scaling {
        #![doc = include_str!("../docs/weighting/scaling.md")]
    }
    /// Per-observation custom weights.
    pub mod custom_weights {
        #![doc = include_str!("../docs/weighting/custom-weights.md")]
    }
}

/// Boundary handling and streaming chunk reconciliation.
#[cfg(doc)]
pub mod advanced {
    /// Boundary handling strategies for edge bias reduction.
    pub mod boundary {
        #![doc = include_str!("../docs/advanced/boundary.md")]
    }
    /// Merge strategies for Streaming mode overlap regions.
    pub mod merge {
        #![doc = include_str!("../docs/advanced/merge.md")]
    }
}

/// Application examples: use cases and real-world patterns.
#[cfg(doc)]
pub mod use_case {
    /// Smoothing genomic data: methylation, ChIP-seq, coverage.
    pub mod genomics {
        #![doc = include_str!("../docs/use-case/use-case-genomics.md")]
    }
    /// Time series smoothing and trend extraction.
    pub mod time_series {
        #![doc = include_str!("../docs/use-case/use-case-time-series.md")]
    }
    /// Real-time sensor and streaming data processing.
    pub mod real_time {
        #![doc = include_str!("../docs/use-case/use-case-real-time.md")]
    }
}

/// Full Rust API reference: structs, builder methods, and result types.
#[cfg(doc)]
pub mod api {
    #![doc = include_str!("../docs/api/api.md")]
    /// `StreamingLowess` API reference.
    pub mod streaming {
        #![doc = include_str!("../docs/api/api-streaming.md")]
    }
    /// `OnlineLowess` API reference.
    pub mod online {
        #![doc = include_str!("../docs/api/api-online.md")]
    }
}

/// Release notes and changelog for this crate.
#[cfg(doc)]
pub mod news {
    #![doc = include_str!("../docs/news.md")]
}
