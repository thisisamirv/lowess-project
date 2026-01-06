//! Execution backend configuration for extension crates.
//!
//! ## Purpose
//!
//! This module defines the `Backend` enum used by extension crates (like `fastLowess`)
//! to select computational backends at runtime. The core `lowess` crate does not
//! implement GPU acceleration directly; this serves as a configuration placeholder
//! for downstream crates.
//!
//! ## Design notes
//!
//! * **Extension-focused**: These types exist to support GPU-accelerated crates.
//! * **Hidden by default**: Fields using `Backend` are `#[doc(hidden)]` in public APIs.
//!
//! ## Key concepts
//!
//! * **CPU**: Default execution mode (standard Rust code).
//! * **GPU**: Hardware accelerated mode (requires external dependencies).
//!
//! ## Invariants
//!
//! * The default backend is always `CPU`.
//!
//! ## Non-goals
//!
//! * This module does not provide GPU implementations (handled by external crates).

/// Execution backend hint for extension crates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::upper_case_acronyms)]
pub enum Backend {
    /// CPU execution (may still use parallelism via rayon).
    #[default]
    CPU,

    /// GPU execution (requires extension crate with GPU support).
    GPU,
}
