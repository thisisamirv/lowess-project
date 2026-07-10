//! Centralised string-alias maps for all option enums.
//!
//! All `impl FromStr` blocks for option types live here so that accepted
//! aliases and their canonical spellings stay in one place.  Every binding
//! frontend (`fastLowess::binding_support`) delegates to these impls instead
//! of maintaining its own duplicated match arms.
//!
//! ## Canonical names
//!
//! The first alias listed in each match arm is the canonical (round-trip)
//! name.  The `*_str` helpers in `fastLowess::binding_support` always return
//! the canonical name so that `parse → str → parse` round-trips correctly.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::string::ToString;

// External dependencies
use core::str::FromStr;

// Internal dependencies
use crate::adapters::online::UpdateMode;
use crate::adapters::streaming::MergeStrategy;
use crate::algorithms::regression::ZeroWeightFallback;
use crate::algorithms::robustness::RobustnessMethod;
use crate::math::boundary::BoundaryPolicy;
use crate::math::kernel::WeightFunction;
use crate::math::scaling::ScalingMethod;
use crate::primitives::errors::LowessError;

// ── WeightFunction ────────────────────────────────────────────────────────────

impl FromStr for WeightFunction {
    type Err = LowessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(WeightFunction::Cosine),
            "epanechnikov" => Ok(WeightFunction::Epanechnikov),
            "gaussian" => Ok(WeightFunction::Gaussian),
            "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
            "triangle" | "triangular" => Ok(WeightFunction::Triangle),
            "tricube" => Ok(WeightFunction::Tricube),
            "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
            _ => Err(LowessError::InvalidOption {
                option: "weight_function",
                value: s.to_string(),
                valid: "tricube, epanechnikov, gaussian, uniform, biweight, triangle, cosine",
            }),
        }
    }
}

// ── BoundaryPolicy ────────────────────────────────────────────────────────────

impl FromStr for BoundaryPolicy {
    type Err = LowessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "extend" | "pad" => Ok(BoundaryPolicy::Extend),
            "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
            "zero" => Ok(BoundaryPolicy::Zero),
            "noboundary" | "none" => Ok(BoundaryPolicy::NoBoundary),
            _ => Err(LowessError::InvalidOption {
                option: "boundary_policy",
                value: s.to_string(),
                valid: "extend, reflect, zero, noboundary",
            }),
        }
    }
}

// ── ScalingMethod ─────────────────────────────────────────────────────────────

impl FromStr for ScalingMethod {
    type Err = LowessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mar" | "median_absolute_residual" => Ok(ScalingMethod::MAR),
            "mad" | "median_absolute_deviation" => Ok(ScalingMethod::MAD),
            "mean" | "mean_absolute_residual" => Ok(ScalingMethod::Mean),
            _ => Err(LowessError::InvalidOption {
                option: "scaling_method",
                value: s.to_string(),
                valid: "mad, mar, mean",
            }),
        }
    }
}

// ── RobustnessMethod ──────────────────────────────────────────────────────────

impl FromStr for RobustnessMethod {
    type Err = LowessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
            "huber" => Ok(RobustnessMethod::Huber),
            "talwar" => Ok(RobustnessMethod::Talwar),
            _ => Err(LowessError::InvalidOption {
                option: "robustness_method",
                value: s.to_string(),
                valid: "bisquare, huber, talwar",
            }),
        }
    }
}

// ── ZeroWeightFallback ────────────────────────────────────────────────────────

impl FromStr for ZeroWeightFallback {
    type Err = LowessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
            "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
            "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
            _ => Err(LowessError::InvalidOption {
                option: "zero_weight_fallback",
                value: s.to_string(),
                valid: "use_local_mean, return_original, return_none",
            }),
        }
    }
}

// ── MergeStrategy ─────────────────────────────────────────────────────────────

impl FromStr for MergeStrategy {
    type Err = LowessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "average" | "mean" => Ok(MergeStrategy::Average),
            "weighted_average" | "weighted" | "weightedaverage" => {
                Ok(MergeStrategy::WeightedAverage)
            }
            "take_first" | "first" | "takefirst" | "left" => Ok(MergeStrategy::TakeFirst),
            "take_last" | "last" | "takelast" | "right" => Ok(MergeStrategy::TakeLast),
            _ => Err(LowessError::InvalidOption {
                option: "merge_strategy",
                value: s.to_string(),
                valid: "average, weighted_average, take_first, take_last",
            }),
        }
    }
}

// ── UpdateMode ────────────────────────────────────────────────────────────────

impl FromStr for UpdateMode {
    type Err = LowessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "full" | "resmooth" => Ok(UpdateMode::Full),
            "incremental" | "single" => Ok(UpdateMode::Incremental),
            _ => Err(LowessError::InvalidOption {
                option: "update_mode",
                value: s.to_string(),
                valid: "full, incremental",
            }),
        }
    }
}

// ─── Binding helpers (only with the `dev` feature) ───────────────────────────
//
// Parse and canonical-name wrappers used by the binding layer.  Only compiled
// when the `dev` feature is active; re-exported through
// `lowess::internals::alias`.

#[cfg(feature = "dev")]
pub mod helpers {
    use super::{
        BoundaryPolicy, LowessError, MergeStrategy, RobustnessMethod, ScalingMethod, UpdateMode,
        WeightFunction, ZeroWeightFallback,
    };

    // ─── Parse helpers ────────────────────────────────────────────────────────

    pub fn parse_weight_function(s: &str) -> Result<WeightFunction, LowessError> {
        s.parse()
    }

    pub fn parse_robustness_method(s: &str) -> Result<RobustnessMethod, LowessError> {
        s.parse()
    }

    pub fn parse_zero_weight_fallback(s: &str) -> Result<ZeroWeightFallback, LowessError> {
        s.parse()
    }

    pub fn parse_boundary_policy(s: &str) -> Result<BoundaryPolicy, LowessError> {
        s.parse()
    }

    pub fn parse_scaling_method(s: &str) -> Result<ScalingMethod, LowessError> {
        s.parse()
    }

    pub fn parse_update_mode(s: &str) -> Result<UpdateMode, LowessError> {
        s.parse()
    }

    pub fn parse_merge_strategy(s: &str) -> Result<MergeStrategy, LowessError> {
        s.parse()
    }

    // ─── Canonical-name helpers ───────────────────────────────────────────────
    //
    // Round-trip guarantee: `X_str(v).parse::<X>().unwrap() == v` for all `v`.

    pub fn weight_function_str(v: WeightFunction) -> &'static str {
        match v {
            WeightFunction::Tricube => "tricube",
            WeightFunction::Epanechnikov => "epanechnikov",
            WeightFunction::Gaussian => "gaussian",
            WeightFunction::Uniform => "uniform",
            WeightFunction::Biweight => "biweight",
            WeightFunction::Triangle => "triangle",
            WeightFunction::Cosine => "cosine",
        }
    }

    pub fn robustness_method_str(v: RobustnessMethod) -> &'static str {
        match v {
            RobustnessMethod::Bisquare => "bisquare",
            RobustnessMethod::Huber => "huber",
            RobustnessMethod::Talwar => "talwar",
        }
    }

    pub fn scaling_method_str(v: ScalingMethod) -> &'static str {
        match v {
            ScalingMethod::MAD => "mad",
            ScalingMethod::MAR => "mar",
            ScalingMethod::Mean => "mean",
        }
    }

    pub fn zero_weight_fallback_str(v: ZeroWeightFallback) -> &'static str {
        match v {
            ZeroWeightFallback::UseLocalMean => "use_local_mean",
            ZeroWeightFallback::ReturnOriginal => "return_original",
            ZeroWeightFallback::ReturnNone => "return_none",
        }
    }

    pub fn boundary_policy_str(v: BoundaryPolicy) -> &'static str {
        match v {
            BoundaryPolicy::Extend => "extend",
            BoundaryPolicy::Reflect => "reflect",
            BoundaryPolicy::Zero => "zero",
            BoundaryPolicy::NoBoundary => "noboundary",
        }
    }

    pub fn update_mode_str(v: UpdateMode) -> &'static str {
        match v {
            UpdateMode::Full => "full",
            UpdateMode::Incremental => "incremental",
        }
    }

    pub fn merge_strategy_str(v: MergeStrategy) -> &'static str {
        match v {
            MergeStrategy::Average => "average",
            MergeStrategy::WeightedAverage => "weighted_average",
            MergeStrategy::TakeFirst => "take_first",
            MergeStrategy::TakeLast => "take_last",
        }
    }
}
