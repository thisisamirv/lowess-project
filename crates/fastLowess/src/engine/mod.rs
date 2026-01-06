//! Layer 5: Engine
//!
//! ## Purpose
//!
//! This layer provides the parallel execution engine for LOWESS smoothing.
//! It handles the distribution of compute tasks across CPU cores or GPU hardware.
//!
//! ## Architecture
//!
//! ```text
//! Layer 7: API
//!   ↓
//! Layer 6: Adapters
//!   ↓
//! Layer 5: Engine ← You are here
//!   ↓
//! Layer 4: Evaluation
//!   ↓
//! lowess
//! ```

/// Parallel execution engine using CPU threads
pub mod executor;

/// GPU-accelerated execution engine using wgpu
#[cfg(feature = "gpu")]
pub mod gpu;
