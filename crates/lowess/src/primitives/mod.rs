//! Layer 1: Primitives
//!
//! This layer provides the primitive abstractions, data structures, and
//! utility functions used throughout the crate. Only `parser` depends on
//! sibling primitive modules, as permitted by the layering rules.

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

// Primitive option parsing.
pub mod parser;
