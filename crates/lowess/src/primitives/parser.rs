//! String parsing for primitive LOWESS options.

#[cfg(not(feature = "std"))]
use alloc::string::ToString;
use core::str::FromStr;

use crate::primitives::backend::Backend;
use crate::primitives::errors::LowessError;
use crate::primitives::policies::{MergeStrategy, MissingPolicy, UpdateMode};

impl FromStr for MissingPolicy {
    type Err = LowessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "error" => Ok(Self::Error),
            "drop" => Ok(Self::Drop),
            _ => Err(LowessError::InvalidOption {
                option: "missing",
                value: s.to_string(),
                valid: "error, drop",
            }),
        }
    }
}

impl FromStr for Backend {
    type Err = LowessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Self::CPU),
            "gpu" => Ok(Self::GPU),
            _ => Err(LowessError::InvalidOption {
                option: "backend",
                value: s.to_string(),
                valid: "cpu, gpu",
            }),
        }
    }
}

impl FromStr for MergeStrategy {
    type Err = LowessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "average" | "mean" => Ok(Self::Average),
            "weighted_average" | "weighted" | "weightedaverage" => Ok(Self::WeightedAverage),
            "take_first" | "first" | "takefirst" | "left" => Ok(Self::TakeFirst),
            "take_last" | "last" | "takelast" | "right" => Ok(Self::TakeLast),
            _ => Err(LowessError::InvalidOption {
                option: "merge_strategy",
                value: s.to_string(),
                valid: "average, weighted_average, take_first, take_last",
            }),
        }
    }
}

impl FromStr for UpdateMode {
    type Err = LowessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "full" | "resmooth" => Ok(Self::Full),
            "incremental" | "single" => Ok(Self::Incremental),
            _ => Err(LowessError::InvalidOption {
                option: "update_mode",
                value: s.to_string(),
                valid: "full, incremental",
            }),
        }
    }
}
