//! Layer 5: Engine
//!
//! This layer provides the parallel execution engine for LOWESS smoothing.
//! It handles the distribution of compute tasks across CPU cores or GPU hardware.

// Parallel execution engine using CPU threads
pub mod executor;

// GPU-accelerated execution engine using wgpu
#[cfg(feature = "gpu")]
pub mod gpu;
