//! Layer 1: Primitives
//!
//! This layer provides the primitive abstractions, data structures, and
//! utility functions used throughout the crate. Primitive option parsing lives
//! with the type definitions; cross-file dependencies are limited to errors.

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

// Shared input and update policies.
pub mod policies;
