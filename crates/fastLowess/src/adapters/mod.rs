//! Layer 6: Adapters
//!
//! ## Purpose
//!
//! This layer provides user-facing APIs that adapt the engine layer for different
//! execution modes and use cases:
//!
//! - **Batch**: Unified adapter for parallel/sequential execution
//! - **Streaming**: Chunked processing for large datasets
//! - **Online**: Incremental updates for real-time data
//!
//! ## Architecture
//!
//! ```text
//! Layer 7: API
//!   ↓
//! Layer 6: Adapters ← You are here
//!   ↓
//! Layer 5: Engine
//!   ↓
//! Layer 4: Evaluation
//!   ↓
//! lowess
//! ```

/// Unified batch adapter for LOWESS smoothing.
pub mod batch;

/// Streaming LOWESS for large datasets.
pub mod streaming;

/// Online LOWESS for real-time data streams.
pub mod online;
