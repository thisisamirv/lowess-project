//! Shared input and update policies for LOWESS execution.

#[cfg(not(feature = "std"))]
use alloc::string::ToString;
use core::str::FromStr;

use crate::primitives::errors::LowessError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingPolicy {
    #[default]
    Error,
    Drop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpdateMode {
    Full,
    #[default]
    Incremental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeStrategy {
    Average,
    #[default]
    WeightedAverage,
    TakeFirst,
    TakeLast,
}

impl FromStr for MissingPolicy {
    type Err = LowessError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "error" => Ok(Self::Error),
            "drop" => Ok(Self::Drop),
            _ => Err(LowessError::InvalidOption {
                option: "missing",
                value: value.to_string(),
                valid: "error, drop",
            }),
        }
    }
}

impl FromStr for MergeStrategy {
    type Err = LowessError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "average" | "mean" => Ok(Self::Average),
            "weighted_average" | "weighted" | "weightedaverage" => Ok(Self::WeightedAverage),
            "take_first" | "first" | "takefirst" | "left" => Ok(Self::TakeFirst),
            "take_last" | "last" | "takelast" | "right" => Ok(Self::TakeLast),
            _ => Err(LowessError::InvalidOption {
                option: "merge_strategy",
                value: value.to_string(),
                valid: "average, weighted_average, take_first, take_last",
            }),
        }
    }
}

impl FromStr for UpdateMode {
    type Err = LowessError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "full" | "resmooth" => Ok(Self::Full),
            "incremental" | "single" => Ok(Self::Incremental),
            _ => Err(LowessError::InvalidOption {
                option: "update_mode",
                value: value.to_string(),
                valid: "full, incremental",
            }),
        }
    }
}
