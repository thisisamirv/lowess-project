//! Execution backend configuration for extension crates.
//!
//! This module defines the `Backend` enum used by extension crates (like `fastLowess`)
//! to select computational backends at runtime. The core `lowess` crate does not
//! implement GPU acceleration directly; this serves as a configuration placeholder
//! for downstream crates.

#[cfg(not(feature = "std"))]
use alloc::string::ToString;
use core::str::FromStr;

use crate::primitives::errors::LowessError;

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

impl FromStr for Backend {
    type Err = LowessError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "cpu" => Ok(Self::CPU),
            "gpu" => Ok(Self::GPU),
            _ => Err(LowessError::InvalidOption {
                option: "backend",
                value: value.to_string(),
                valid: "cpu, gpu",
            }),
        }
    }
}
