//! Layer 1: Primitives
//!
//! # Purpose
//!
//! This layer provides the primitive abstractions, data structures, and
//! utility functions used throughout the crate. It has zero internal
//! dependencies within the crate.
//!
//! # Architecture
//!
//! ```text
//! Layer 7: API
//!   ↓
//! Layer 6: Adapters
//!   ↓
//! Layer 5: Engine
//!   ↓
//! Layer 4: Evaluation
//!   ↓
//! Layer 3: Algorithms
//!   ↓
//! Layer 2: Math
//!   ↓
//! Layer 1: Primitives ← You are here
//! ```

/// Sorting utilities.
pub mod sorting;

/// Windowing logic.
pub mod window;

/// Shared error types.
pub mod errors;

/// Execution backend configuration.
pub mod backend;

/// Buffer management.
pub mod buffer;
