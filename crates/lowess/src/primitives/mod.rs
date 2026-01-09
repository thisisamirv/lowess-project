//! Layer 1: Primitives
//!
//! This layer provides the primitive abstractions, data structures, and
//! utility functions used throughout the crate. It has zero internal
//! dependencies within the crate.

// Sorting utilities.
pub mod sorting;

// Windowing logic.
pub mod window;

// Shared error types.
pub mod errors;

// Execution backend configuration.
pub mod backend;

// Buffer management.
pub mod buffer;
