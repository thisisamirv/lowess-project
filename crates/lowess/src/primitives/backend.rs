//! Execution backend configuration for extension crates.
//!
//! This module defines the `Backend` enum used by extension crates (like `fastLowess`)
//! to select computational backends at runtime. The core `lowess` crate does not
//! implement GPU acceleration directly; this serves as a configuration placeholder
//! for downstream crates.

// Execution backend hint for extension crates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::upper_case_acronyms)]
pub enum Backend {
    // CPU execution (may still use parallelism via rayon).
    #[default]
    CPU,

    // GPU execution (requires extension crate with GPU support).
    GPU,
}
